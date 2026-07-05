import os
import sqlite3
import asyncio
import aiosqlite
import datetime
import time
import logging
import json
import numpy as np
import pandas as pd
import faiss
import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from sentence_transformers import SentenceTransformer

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("axon")



from config_loader import Config, DataSource, load_config

# Engine
class AxonEngine:
    def __init__(self, config: Config):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.engine.max_concurrent_connections)

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set. "
                "Add it to your .env file or set it in your environment."
            )
        self.client = genai.Client(api_key=api_key)

        self._init_mem_palace()
        self._build_faiss_index()
        self._init_audit_log()
        self._init_catalog()
    
    async def _fetch_from_single_source(
        self,
        source_name: str,
        source_type: str,
        source_path: str,
        sql_query: str,
        impersonation_user_id: str = "default_user"
    ) -> dict:

        try:
            if source_type == "enterprise":
                # Enterprise databases use connection pooling and impersonation
                if not hasattr(self, "ent_db"):
                    from enterprise_db import EnterpriseDatabase
                    self.ent_db = EnterpriseDatabase(source_path)

                data = await asyncio.to_thread(
                    self.ent_db.execute_query,
                    sql=sql_query
                )

            elif source_type == "sqlite":
                from connectors.sqlite_connector import SQLiteConnector
                connector = SQLiteConnector(source_path, self.semaphore)
                data = await connector.execute(sql_query)

            elif source_type == "csv":
                from connectors.csv_connector import CSVConnector
                connector = CSVConnector(source_path)
                data = await connector.execute(sql_query)

            elif source_type == "rest":
                from connectors.rest_connector import RESTConnector
                connector = RESTConnector(source_path)
                data = await connector.execute(sql_query)

            else:
                raise ValueError(f"Unknown source type: {source_type}")

            return {
                "database": source_name,
                "status": "success",
                "data": data
            }

        except Exception as e:
            return {
                "database": source_name,
                "status": "error",
                "msg": str(e)
            }

    
    async def execute_federated_query(self, sql_query: str, impersonation_user_id: str = "default_user") -> list:
        """
        Fires sql_query across every registered source using the appropriate connector.
        Source types and paths come from the catalog — no hardcoded folder scanning.
        """
        sources = await self._get_registered_sources()

        tasks = [
            self._fetch_from_single_source(
                s["source_name"], s["source_type"], s["source_path"], sql_query, impersonation_user_id
            )
            for s in sources
        ]

        results = await asyncio.gather(*tasks)
        return list(results)


    async def execute_federated_query_stream(self, sql_query: str, impersonation_user_id: str = "default_user"):
        """Streaming version — yields each result as it arrives."""
        sources = await self._get_registered_sources()

        tasks = [
            asyncio.ensure_future(
                self._fetch_from_single_source(
                    s["source_name"], s["source_type"], s["source_path"], sql_query, impersonation_user_id
                )
            )
            for s in sources
        ]

        for completed_future in asyncio.as_completed(tasks):
            result = await completed_future
            
            # ── SELF-CORRECTION ANOMALY GATE ──────────────────────────────
            if result.get("status") == "success" and isinstance(result.get("data"), list):
                if len(result["data"]) > self.config.engine.max_rows_per_query:
                    logger.warning(f"Anomaly Gate Triggered: {len(result['data'])} rows on {result.get('database')}")
                    result = {
                        "database": result.get("database"),
                        "status": "error",
                        "msg": f"CRITICAL ANOMALY DETECTED: Query returned {len(result['data']):,} rows, exceeding the safety limit of {self.config.engine.max_rows_per_query:,}. The AI hallucinated a bad JOIN or Cartesian product."
                    }
                    
            yield result


    async def _get_registered_sources(self) -> list:
        """Reads all distinct sources from the catalog registry."""
        async with aiosqlite.connect(os.path.join(os.environ.get("OMNI_DATA_DIR", "."), "omni_master_catalog.db")) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT DISTINCT source_name, source_type, source_path FROM schema_registry"
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
    
    async def metadata(self):
        """
        Builds the central schema registry from all configured data sources.
        Supports sqlite, sqlite_folder, csv, and rest source types.
        Incremental — skips sources unchanged since last scan.
        """
        pipeline_start = time.time()
        scanned_count = 0
        skipped_count = 0

        catalog_path = os.path.join(os.environ.get("OMNI_DATA_DIR", "."), "omni_master_catalog.db")

        async with aiosqlite.connect(catalog_path) as master_db:

            # ── PROCESS EACH CONFIGURED SOURCE ───────────────────────
            for source in self.config.data_sources:

                if source.type == "sqlite_folder":
                    # Expand into individual sqlite sources
                    folder = source.path
                    if not os.path.exists(folder):
                        logger.error(f"sqlite_folder path not found: {folder}")
                        continue

                    db_files = [
                        f for f in os.listdir(folder)
                        if f.endswith(".db")
                        and f not in ("omni_master_catalog.db", "mem_palace_vault.db", "omni_audit.db")
                    ]

                    for db_file in db_files:
                        full_path = os.path.join(folder, db_file)
                        source_name = f"{source.name}/{db_file}"
                        scanned, skipped = await self._scan_single_source(
                            master_db, source_name, "sqlite", full_path,
                            source.schema_notes, source.auto_schema_notes
                        )
                        scanned_count += scanned
                        skipped_count += skipped

                elif source.type in ("sqlite", "csv"):
                    scanned, skipped = await self._scan_single_source(
                        master_db, source.name, source.type,
                        source.path, source.schema_notes, source.auto_schema_notes
                    )
                    scanned_count += scanned
                    skipped_count += skipped

                elif source.type == "rest":
                    # REST sources are registered without mtime (they're live)
                    # Schema is inferred from a sample response
                    scanned, skipped = await self._scan_rest_source(
                        master_db, source
                    )
                    scanned_count += scanned
                    skipped_count += skipped

                else:
                    logger.warning(f"Unknown source type '{source.type}' for '{source.name}' — skipping")

            await master_db.commit()

        duration_ms = int((time.time() - pipeline_start) * 1000)
        await self._log_pipeline_run(scanned_count, skipped_count, duration_ms)
        logger.info(
            f"Catalog refresh: {scanned_count} scanned, "
            f"{skipped_count} skipped, {duration_ms}ms"
        )


    async def _scan_single_source(
        self,
        master_db,
        source_name: str,
        source_type: str,
        source_path: str,
        schema_notes: str,
        auto_schema_notes: bool = True
    ) -> tuple[int, int]:
        """
        Scans one SQLite or CSV source.
        Populates schema_registry (table level) and column_registry (column level).
        Returns (scanned_count, skipped_count).
        """

        if not source_path or not os.path.exists(source_path):
            logger.error(f"Source path not found: {source_path}")
            return 0, 0

        current_mtime = self._get_source_mtime(source_path)

        # ── CHECK IF UNCHANGED ────────────────────────────────────────
        cursor = await master_db.execute(
            "SELECT last_scanned_mtime FROM schema_registry WHERE source_name = ? LIMIT 1",
            (source_name,)
        )
        existing = await cursor.fetchone()
        if existing and existing[0] == current_mtime:
            return 0, 1   # skipped — nothing changed

        # ── DELETE STALE ENTRIES ──────────────────────────────────────
        await master_db.execute(
            "DELETE FROM schema_registry WHERE source_name = ?", (source_name,)
        )
        await master_db.execute(
            "DELETE FROM column_registry WHERE source_name = ?", (source_name,)
        )

        try:
            if source_type == "sqlite":

                async with aiosqlite.connect(source_path) as db:

                    # ── GET ALL TABLES ────────────────────────────────
                    cursor = await db.execute(
                        "SELECT name, sql FROM sqlite_master WHERE type='table';"
                    )
                    tables = await cursor.fetchall()

                    for table_name, raw_schema in tables:
                        if not table_name or table_name.startswith("sqlite_"):
                            continue

                        # ── ROW COUNT ─────────────────────────────────
                        try:
                            count_cursor = await db.execute(
                                f"SELECT COUNT(*) FROM \"{table_name}\""
                            )
                            row_count = (await count_cursor.fetchone())[0]
                        except Exception:
                            row_count = 0

                        # ── COLUMN INFO VIA PRAGMA ────────────────────
                        # PRAGMA table_info returns:
                        # cid | name | type | notnull | dflt_value | pk
                        pragma_cursor = await db.execute(
                            f"PRAGMA table_info(\"{table_name}\")"
                        )
                        pragma_rows = await pragma_cursor.fetchall()

                        column_data = []
                        for prow in pragma_rows:
                            col_id, col_name, col_type, not_null, default_val, is_pk = prow

                            # ── SAMPLE VALUES ─────────────────────────
                            sample_values = []
                            null_count    = 0
                            distinct_count = 0

                            try:
                                # Get up to 10 distinct non-null sample values
                                sample_cursor = await db.execute(
                                    f"SELECT DISTINCT \"{col_name}\" "
                                    f"FROM \"{table_name}\" "
                                    f"WHERE \"{col_name}\" IS NOT NULL "
                                    f"LIMIT 10"
                                )
                                samples = await sample_cursor.fetchall()
                                sample_values = [str(row[0]) for row in samples]

                                # Distinct count
                                dist_cursor = await db.execute(
                                    f"SELECT COUNT(DISTINCT \"{col_name}\") "
                                    f"FROM \"{table_name}\""
                                )
                                distinct_count = (await dist_cursor.fetchone())[0]

                                # Null count
                                null_cursor = await db.execute(
                                    f"SELECT COUNT(*) FROM \"{table_name}\" "
                                    f"WHERE \"{col_name}\" IS NULL"
                                )
                                null_count = (await null_cursor.fetchone())[0]

                            except Exception as e:
                                logger.debug(f"Sample extraction failed for {table_name}.{col_name}: {e}")

                            column_data.append({
                                "column_name":   col_name,
                                "column_type":   col_type or "TEXT",
                                "is_primary_key": int(is_pk),
                                "is_nullable":    0 if not_null else 1,
                                "sample_values":  sample_values,
                                "distinct_count": distinct_count,
                                "null_count":     null_count,
                            })

                            # Write to column_registry
                            await master_db.execute(
                                """INSERT OR REPLACE INTO column_registry
                                (source_name, table_name, column_name, column_type,
                                    is_primary_key, is_nullable, sample_values,
                                    distinct_count, null_count)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                (
                                    source_name,
                                    table_name,
                                    col_name,
                                    col_type or "TEXT",
                                    int(is_pk),
                                    0 if not_null else 1,
                                    json.dumps(sample_values),
                                    distinct_count,
                                    null_count,
                                )
                            )

                        # ── TABLE DESCRIPTION ─────────────────────────
                        table_description = self._describe_table(
                            table_name, column_data, row_count
                        )

                        # ── AUTO-GENERATE SCHEMA NOTES ────────────────
                        effective_notes = schema_notes
                        if not effective_notes and auto_schema_notes:
                            effective_notes = self._auto_generate_schema_notes(
                                table_name, column_data, row_count
                            )

                        # ── WRITE TO SCHEMA REGISTRY ──────────────────
                        await master_db.execute(
                            """INSERT OR REPLACE INTO schema_registry
                            (source_name, source_type, source_path, table_name,
                                raw_schema, schema_notes, row_count,
                                table_description, last_scanned_mtime)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                source_name,
                                source_type,
                                source_path,
                                table_name,
                                raw_schema or "",
                                effective_notes,
                                row_count,
                                table_description,
                                current_mtime,
                            )
                        )

            elif source_type == "csv":
                import pandas as pd

                df = pd.read_csv(source_path, nrows=500)   # sample for stats
                table_name = "df"

                columns_sql = ", ".join([f'"{col}" TEXT' for col in df.columns])
                raw_schema = f"CREATE TABLE df ({columns_sql})"
                row_count = len(pd.read_csv(source_path, usecols=[0]))

                column_data = []
                for col in df.columns:
                    sample_values  = [str(v) for v in df[col].dropna().unique()[:10].tolist()]
                    distinct_count = int(df[col].nunique())
                    null_count     = int(df[col].isna().sum())
                    col_type       = str(df[col].dtype)

                    column_data.append({
                        "column_name":    col,
                        "column_type":    col_type,
                        "is_primary_key": 0,
                        "is_nullable":    1,
                        "sample_values":  sample_values,
                        "distinct_count": distinct_count,
                        "null_count":     null_count,
                    })

                    await master_db.execute(
                        """INSERT OR REPLACE INTO column_registry
                        (source_name, table_name, column_name, column_type,
                            is_primary_key, is_nullable, sample_values,
                            distinct_count, null_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            source_name, table_name, col, col_type,
                            0, 1,
                            json.dumps(sample_values),
                            distinct_count, null_count,
                        )
                    )

                table_description = self._describe_table(table_name, column_data, row_count)

                # ── AUTO-GENERATE SCHEMA NOTES ────────────────────
                effective_notes = schema_notes
                if not effective_notes and auto_schema_notes:
                    effective_notes = self._auto_generate_schema_notes(
                        table_name, column_data, row_count
                    )

                await master_db.execute(
                    """INSERT OR REPLACE INTO schema_registry
                    (source_name, source_type, source_path, table_name,
                        raw_schema, schema_notes, row_count,
                        table_description, last_scanned_mtime)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        source_name, source_type, source_path, table_name,
                        raw_schema, effective_notes, row_count,
                        table_description, current_mtime,
                    )
                )

            return 1, 0   # scanned

        except Exception as e:
            logger.error(f"Scan failed for {source_name}: {e}")
            return 0, 0


    async def _scan_rest_source(self, master_db, source: DataSource) -> tuple[int, int]:
        """Registers a REST source. Fetches a sample to infer schema."""
        try:
            async with httpx.AsyncClient() as http_client:
                headers = {}
                if source.auth_token:
                    headers["Authorization"] = f"Bearer {source.auth_token}"
                response = await http_client.get(source.url, headers=headers, timeout=10.0)
                response.raise_for_status()
                data = response.json()

            if isinstance(data, list) and len(data) > 0:
                sample = data[0]
            elif isinstance(data, dict):
                sample = data
            else:
                sample = {}

            columns_raw = list(sample.keys())
            raw_schema = f"CREATE TABLE df ({', '.join([f'{k} TEXT' for k in columns_raw])})"

            # ── DELETE STALE ENTRIES ──────────────────────────────────
            await master_db.execute(
                "DELETE FROM schema_registry WHERE source_name = ?", (source.name,)
            )
            await master_db.execute(
                "DELETE FROM column_registry WHERE source_name = ?", (source.name,)
            )

            # ── WRITE COLUMN REGISTRY ─────────────────────────────────
            for col_name in columns_raw:
                sample_val = str(sample.get(col_name, ""))
                await master_db.execute(
                    """INSERT OR REPLACE INTO column_registry
                    (source_name, table_name, column_name, column_type,
                        is_primary_key, is_nullable, sample_values, distinct_count, null_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        source.name, "df", col_name, "TEXT",
                        0, 1,
                        json.dumps([sample_val]) if sample_val else json.dumps([]),
                        0, 0
                    )
                )

            # ── WRITE SCHEMA REGISTRY ─────────────────────────────────
            column_data = [{"column_name": k, "column_type": "TEXT", "sample_values": [str(sample.get(k,""))]} for k in columns_raw]
            table_description = self._describe_table("df", column_data, 0)

            await master_db.execute(
                """INSERT OR REPLACE INTO schema_registry
                (source_name, source_type, source_path, table_name,
                    raw_schema, schema_notes, row_count, table_description, last_scanned_mtime)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    source.name, "rest", source.url, "df",
                    raw_schema, source.schema_notes or "", 0,
                    table_description, 0.0
                )
            )
            return 1, 0

        except Exception as e:
            logger.error(f"REST source scan failed for {source.name}: {e}")
            return 0, 0
    
    async def _discover_schema(self, table_name: str) -> tuple[str, str]:
        """
        Returns (rich_schema_string, schema_notes).

        rich_schema_string is formatted so Agent 2 knows:
        - Column names and types
        - Whether each column is a primary key
        - Actual sample values (eliminates casing bugs entirely)
        - Row count

        Example output:
            Table: transactions (9,000 rows)
            Columns:
            - id        INTEGER  [primary key]
            - amount    INTEGER  sample values: 100, 250, 3400, 780
            - status    TEXT     sample values: 'Success', 'Failed', 'Pending'
        """

        catalog_path = os.path.join(os.environ.get("OMNI_DATA_DIR", "."), "omni_master_catalog.db")

        try:
            async with aiosqlite.connect(catalog_path) as db:
                db.row_factory = aiosqlite.Row

                # Get table-level info
                table_cursor = await db.execute(
                    """SELECT raw_schema, schema_notes, row_count, table_description
                    FROM schema_registry
                    WHERE table_name = ?
                    LIMIT 1""",
                    (table_name,)
                )
                table_row = await table_cursor.fetchone()

                if not table_row:
                    return f"ERROR: Table '{table_name}' not found in catalog.", ""

                row_count    = table_row["row_count"] or 0
                schema_notes = table_row["schema_notes"] or ""

                # Get column-level info
                col_cursor = await db.execute(
                    """SELECT column_name, column_type, is_primary_key,
                            is_nullable, sample_values, distinct_count
                    FROM column_registry
                    WHERE table_name = ?
                    ORDER BY rowid""",
                    (table_name,)
                )
                columns = await col_cursor.fetchall()

                if not columns:
                    # Fall back to raw schema if column registry is empty
                    return table_row["raw_schema"], schema_notes

                # ── BUILD RICH SCHEMA STRING ──────────────────────────
                lines = [f"Table: {table_name} ({row_count:,} rows)"]
                lines.append("Columns:")

                for col in columns:
                    col_name  = col["column_name"]
                    col_type  = col["column_type"]
                    is_pk     = col["is_primary_key"]
                    samples   = json.loads(col["sample_values"] or "[]")
                    distinct  = col["distinct_count"] or 0

                    parts = [f"  - {col_name:<20} {col_type}"]

                    if is_pk:
                        parts.append("[primary key]")

                    if samples and not self.config.privacy.disable_sample_values:
                        # Format samples: strings get quotes, numbers don't
                        formatted = []
                        for s in samples[:8]:
                            try:
                                float(s)
                                formatted.append(s)
                            except ValueError:
                                formatted.append(f"'{s}'")
                        sample_str = ", ".join(formatted)
                        parts.append(f"values: {sample_str}")
                        if distinct > len(samples):
                            parts.append(f"({distinct:,} distinct)")

                    lines.append(" ".join(parts))

                rich_schema = "\n".join(lines)
                return rich_schema, schema_notes

        except Exception as e:
            return f"ERROR: Catalog query failed: {str(e)}", ""
    
    async def translate_intent_to_sql(self, user_prompt: str, user_id: str = "default_user") -> tuple[str, int, int, float]:

        router_prompt = (
            f"You are a SQL table extractor. Given a natural language query, "
            f"return only the table name as a single word. No explanation.\n\nQuery: {user_prompt}"
        )

        try:
            router_response = await self.client.aio.models.generate_content(
                model=self.config.llm.router_model,
                contents=router_prompt
            )
        except Exception as e:
            logger.error(f"Agent 1 failed: {str(e)}")
            return "-- ERROR: Agent 1 failed", 0, 0, 0.0

        table_name = router_response.text.strip().lower()
        logger.debug(f"Agent 1 identified table: '{table_name}'")

        try:
            agent1_input = router_response.usage_metadata.prompt_token_count
            agent1_output = router_response.usage_metadata.candidates_token_count
        except Exception:
            agent1_input = 0
            agent1_output = 0

        # Discover schema — now returns schema_notes too
        rich_schema, schema_notes = await self._discover_schema(table_name)
        if not rich_schema or rich_schema.startswith("ERROR"):
            logger.error(f"Schema lookup failed for table '{table_name}'")
            return f"-- ERROR: Schema lookup failed for '{table_name}'", agent1_input + agent1_output, 0, 0.0

        schema = rich_schema
        logger.debug(f"Scout retrieved schema: {schema}")

        # Build prompt — inject schema_notes from config if available
        notes_section = f"\nNote: {schema_notes}" if schema_notes else ""

        sql_prompt = (
            f"Table Schema:\n{schema}{notes_section}\n\n"
            f"User Intent:\n{user_prompt}\n\n"
            f"Return ONLY the raw SQL query. No markdown, no explanation.\n\n"
            f"SQL Query:"
        )

        # ── INVISIBLE CONTEXT & PII MASKING ──────────────────────────────
        context = self.config.privacy.user_contexts.get(user_id)
        if context:
            sql_prompt += f"\n\nCRITICAL CONTEXT FOR THIS USER: {context}"
            
        if self.config.privacy.masked_columns:
            masked_list = ", ".join(self.config.privacy.masked_columns)
            sql_prompt += f"\n\nSECURITY MANDATE: You must wrap the following PII columns in a SHA256() function (or equivalent hashing function for the database type) if they are selected: {masked_list}."

        try:
            sql_response = await self.client.aio.models.generate_content(
                model=self.config.llm.sql_model,
                contents=sql_prompt
            )
        except Exception as e:
            logger.error(f"Agent 2 failed: {str(e)}")
            return "-- ERROR: Agent 2 failed", agent1_input + agent1_output, 0, 0.0

        try:
            agent2_input = sql_response.usage_metadata.prompt_token_count
            agent2_output = sql_response.usage_metadata.candidates_token_count
        except Exception:
            agent2_input = 0
            agent2_output = 0

        clean_sql = sql_response.text.replace("```sql", "").replace("```", "").strip()
        logger.debug(f"Agent 2 generated SQL: {clean_sql}")

        input_cost_rate = self.config.llm.input_cost_per_million_tokens / 1_000_000
        output_cost_rate = self.config.llm.output_cost_per_million_tokens / 1_000_000
        total_cost = ((agent1_input + agent2_input) * input_cost_rate +
                      (agent1_output + agent2_output) * output_cost_rate)
        agent1_tokens_total = agent1_input + agent1_output
        agent2_tokens_total = agent2_input + agent2_output

        return clean_sql, agent1_tokens_total, agent2_tokens_total, total_cost
    
    def _init_mem_palace(self) -> None:
        con = sqlite3.connect(os.path.join(os.environ.get("OMNI_DATA_DIR", "."), "mem_palace_vault.db"))
        cur = con.cursor()
    
        cur.execute("CREATE TABLE IF NOT EXISTS vault (intent_key TEXT PRIMARY KEY, approved_sql TEXT, hit_count INTEGER DEFAULT 0, last_used TEXT, embedding BLOB)"
        )

        try:
            cur.execute("ALTER TABLE vault ADD COLUMN embedding BLOB")
        except Exception:
            pass

        con.commit()
        con.close()

        self.embedding_model = SentenceTransformer(self.config.cache.embedding_model)
        self.faiss_index = faiss.IndexFlatIP(self.config.cache.embedding_dim)
        self.faiss_key_map = []

        logger.info("MemPalace Vault initialized")

    def _compute_embedding(self, text: str) -> np.ndarray:

        embedding = self.embedding_model.encode([text], convert_to_numpy=True, normalize_embeddings=True)

        return embedding.astype(np.float32)
    
    def _build_faiss_index(self):
        con = sqlite3.connect(os.path.join(os.environ.get("OMNI_DATA_DIR", "."), "mem_palace_vault.db"))
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        rows = cur.execute(
            "SELECT intent_key, embedding FROM vault WHERE embedding IS NOT NULL"
        ).fetchall()
        con.close()

        for row in rows:
            embedding = np.frombuffer(
                row["embedding"], dtype=np.float32
            ).reshape(1, -1)
            self.faiss_index.add(embedding)
            self.faiss_key_map.append(row["intent_key"])

        logger.info(f"FAISS index rebuilt with {self.faiss_index.ntotal} entries")

    async def learn_correction(self, user_prompt: str, corrected_sql: str):
        """
        Active Learning: When an operator manually corrects an AI-generated SQL query,
        we embed the prompt and the corrected SQL into the MemPalace Vault.
        Subsequent identical or semantically similar queries will now hit this corrected SQL.
        """
        intent_key = user_prompt.strip().lower()
        embedding = self._compute_embedding(intent_key)
        
        async with aiosqlite.connect(os.path.join(os.environ.get("OMNI_DATA_DIR", "."), "mem_palace_vault.db")) as db:
            # Check if it exists
            cursor = await db.execute("SELECT intent_key FROM vault WHERE intent_key = ?", (intent_key,))
            row = await cursor.fetchone()
            
            if row:
                await db.execute(
                    """UPDATE vault 
                       SET approved_sql = ?, embedding = ?, last_used = ? 
                       WHERE intent_key = ?""",
                    (corrected_sql, embedding.tobytes(), datetime.datetime.now().isoformat(), intent_key)
                )
            else:
                await db.execute(
                    """INSERT INTO vault (intent_key, approved_sql, embedding, hit_count, last_used)
                       VALUES (?, ?, ?, 1, ?)""",
                    (intent_key, corrected_sql, embedding.tobytes(), datetime.datetime.now().isoformat())
                )
            await db.commit()
            
        logger.info(f"Active Learning: Saved corrected SQL for intent '{intent_key}'")
        
        # Rebuild FAISS index to include the new embedding
        self.faiss_index = faiss.IndexFlatIP(self.config.cache.embedding_dim)
        self.faiss_key_map = []
        self._build_faiss_index()

    async def _check_mem_palace(self, intent_key: str) -> str | None:

        # ── EXACT MATCH (cheap O(1) lookup) ──────────────────────────
        async with aiosqlite.connect(os.path.join(os.environ.get("OMNI_DATA_DIR", "."), "mem_palace_vault.db")) as db:
            cursor = await db.execute(
                "SELECT approved_sql FROM vault WHERE intent_key = ?",
                (intent_key,)
            )
            row = await cursor.fetchone()
            if row is not None:
                await db.execute(
                    "UPDATE vault SET hit_count = hit_count + 1, last_used = ? WHERE intent_key = ?",
                    (datetime.datetime.now().isoformat(), intent_key)
                )
                await db.commit()
                logger.info(f"Exact cache HIT for: '{intent_key}'")
                return row[0]

        # ── SEMANTIC SEARCH (FAISS fallback) ─────────────────────────
        if self.faiss_index.ntotal > 0:
            embedding = self._compute_embedding(intent_key)
            distances, indices = self.faiss_index.search(embedding, k=1)

            similarity = float(distances[0][0])

            if similarity >= self.config.cache.semantic_similarity_threshold:
                matched_key = self.faiss_key_map[int(indices[0][0])]
                logger.info(
                    f"Semantic cache HIT (similarity={similarity:.3f}): "
                    f"'{intent_key}' → '{matched_key}'"
                )
                # Fetch the SQL for the matched key
                async with aiosqlite.connect(os.path.join(os.environ.get("OMNI_DATA_DIR", "."), "mem_palace_vault.db")) as db:
                    cursor = await db.execute(
                        "SELECT approved_sql FROM vault WHERE intent_key = ?",
                        (matched_key,)
                    )
                    row = await cursor.fetchone()
                    if row:
                        await db.execute(
                            "UPDATE vault SET hit_count = hit_count + 1, last_used = ? WHERE intent_key = ?",
                            (datetime.datetime.now().isoformat(), matched_key)
                        )
                        await db.commit()
                        return row[0]

        return None

    async def _find_similar_queries(self, user_prompt: str, threshold: float = 0.4) -> list:

        # Fetch all stored intent keys from vault
        async with aiosqlite.connect(os.path.join(os.environ.get("OMNI_DATA_DIR", "."), "mem_palace_vault.db")) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT intent_key, approved_sql FROM vault"
            )
            rows = await cursor.fetchall()

        # Not enough data for meaningful TF-IDF comparison
        if len(rows) < 3:
            return []

        stored_keys = [row["intent_key"] for row in rows]
        stored_sql  = [row["approved_sql"] for row in rows]

        # Fit TF-IDF on all stored intent keys
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(stored_keys)

        try:
            prompt_vector = vectorizer.transform([user_prompt])
        except Exception:
            return []

        # Cosine similarity between the new prompt and all stored keys
        similarities = cosine_similarity(prompt_vector, tfidf_matrix)[0]

        # Collect entries in the middle band:
        results = []
        for i, sim in enumerate(similarities):
            if threshold <= sim < self.config.cache.semantic_similarity_threshold:
                results.append({
                    "intent_key": stored_keys[i],
                    "sql": stored_sql[i],
                    "similarity": round(float(sim), 4)
                })

        # Sort by similarity descending — best match first
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

    # 

    async def _store_in_mem_palace(self, intent_key: str, approved_sql: str) -> None:

        embedding = self._compute_embedding(intent_key)
        embedding_bytes = embedding.tobytes()

        async with aiosqlite.connect(os.path.join(os.environ.get("OMNI_DATA_DIR", "."), "mem_palace_vault.db")) as db:
            await db.execute(
                "INSERT OR REPLACE INTO vault VALUES (?, ?, 0, ?, ?)",
                (
                    intent_key,
                    approved_sql,
                    datetime.datetime.now().isoformat(),
                    embedding_bytes
                )
            )
            await db.commit()

        if intent_key not in self.faiss_key_map:
            self.faiss_index.add(embedding)
            self.faiss_key_map.append(intent_key)

        logger.debug(f"Stored in MemPalace + FAISS: '{intent_key}'")

    # Legacy method — not called in API deployment. Kept for local testing only.
    
    # def _suggest_cached_query(self, similar_queries: list) -> str | None:

    #     if not similar_queries:
    #         return None
    #     top = similar_queries[0]
    #     # Present the suggestion to the operator
    #     print("\n--- SIMILAR QUERY FOUND IN MEMPALACE ---")
    #     print(f"  Previous intent: {top['intent_key']}")
    #     print(f"  Cached SQL:      {top['sql']}")
    #     print(f"  Similarity:      {top['similarity']:.2%}")
    #     print("----------------------------------------")
    #     decision = input("Use this cached query instead of calling AI? [Y/N]: ").strip()
    #     # Log the suggestion and decision to audit
    #     try:
    #         con = sqlite3.connect(os.path.join(os.environ.get("OMNI_DATA_DIR", "."), "omni_audit.db"))
    #         cur = con.cursor()
    #         # Add suggestion_log table if it doesn't exist
    #         cur.execute("""
    #             CREATE TABLE IF NOT EXISTS suggestion_log (
    #                 log_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    #                 timestamp       TEXT,
    #                 user_prompt     TEXT,
    #                 suggested_key   TEXT,
    #                 similarity      REAL,
    #                 accepted        INTEGER
    #             )
    #         """)
    #         cur.execute(
    #             "INSERT INTO suggestion_log VALUES (NULL, ?, ?, ?, ?, ?)",
    #             (
    #                 datetime.datetime.now().isoformat(),
    #                 top["intent_key"],
    #                 top["intent_key"],
    #                 top["similarity"],
    #                 1 if decision == "Y" else 0
    #             )
    #         )
    #         con.commit()
    #         con.close()
    #     except Exception as e:
    #         logger.error(f"Suggestion log write failed: {str(e)}")

    #     if decision == "Y":
    #         logger.info(
    #             f"Suggestion accepted (similarity={top['similarity']:.3f}): "
    #             f"'{top['intent_key']}'"
    #         )
    #         return top["sql"]

    #     logger.info("Suggestion rejected — proceeding to AI")
    #     return None    
    
    # def _conscious_gate(self, generated_sql: str) -> bool:

    #     print("=x=x=x=x=x=")
    #     print("GENERATED SQL — AWAITING APPROVAL")
    #     print(generated_sql)
    #     print("=x=x=x=x=x=")

    #     decision = input("Approve this query? [Y/N]: ").strip()
        
    #     return decision == "Y"

    
    def _validate_input(self, user_prompt: str) -> tuple[bool, str]:
        if not user_prompt or not user_prompt.strip():
            return False, "Prompt cannot be empty"

        if len(user_prompt) > self.config.engine.max_prompt_length:
            return False, f"Prompt exceeds {self.config.engine.max_prompt_length} characters"

        if any(signal in user_prompt.upper() for signal in self.config.engine.sql_injection_signals):
            return False, "Prompt contains disallowed SQL fragments"

        return True, ""
    
    def _init_audit_log(self) -> None:
        con = sqlite3.connect(os.path.join(os.environ.get("OMNI_DATA_DIR", "."), "omni_audit.db"))
        cur = con.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS query_logs (
                log_id               INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp            TEXT,
                user_prompt          TEXT,
                generated_sql        TEXT,
                latency_ms           INTEGER,
                ai_latency_ms        INTEGER,
                execution_latency_ms INTEGER,
                agent1_tokens        INTEGER,
                agent2_tokens        INTEGER,
                query_cost_usd       REAL,
                from_cache           INTEGER,
                is_automated         INTEGER DEFAULT 0
            )
        """)

        new_columns = [
            ("ai_latency_ms",        "INTEGER"),
            ("execution_latency_ms", "INTEGER"),
            ("agent1_tokens",        "INTEGER"),
            ("agent2_tokens",        "INTEGER"),
            ("query_cost_usd",       "REAL"),
            ("from_cache",           "INTEGER"),
            ("is_automated",         "INTEGER DEFAULT 0"),
        ]
        for col_name, col_type in new_columns:
            try:
                cur.execute(f"ALTER TABLE query_logs ADD COLUMN {col_name} {col_type}")
            except Exception:
                pass

        cur.execute("""
            CREATE TABLE IF NOT EXISTS suggestion_log (
                log_id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT,
                user_prompt     TEXT,
                suggested_key   TEXT,
                similarity      REAL,
                accepted        INTEGER
            )
        """)

        con.commit()
        con.close()
        logger.info("Audit log initialized")

    def _run_ml_analysis(self, results: list) -> dict:

        # ── EXTRACT NUMERIC VALUES ────────────────────────────────────
        extracted = []
        for res in results:
            if res["status"] == "success" and res.get("data"):
                try:
                    value = float(res["data"][0][0])
                    extracted.append({"database": res["database"], "value": value})
                except (TypeError, ValueError, IndexError):
                    pass

        if len(extracted) < 3:
            return {"error": "Insufficient numeric data for analysis",}

        df = pd.DataFrame(extracted)


        # ── ISOLATION FOREST ─────────────────────────────────────────
        model = IsolationForest(contamination=0.1, random_state=42)
        predictions = model.fit_predict(df[["value"]])
        anomalous_dbs = df[predictions == -1]["database"].tolist()


        # ── LINEAR REGRESSION TREND ──────────────────────────────────
        X = df.index.values.reshape(-1, 1)
        y = df["value"].values
        reg = LinearRegression().fit(X, y)
        slope = reg.coef_[0]
        if abs(slope) < 0.01: direction = "flat"
        elif slope > 0: direction = "up"
        else: direction = "down"


        # ── DBSCAN CLUSTERING ────────────────────────────────────────
        clustering = DBSCAN(eps=50, min_samples=3).fit(df[["value"]])
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)


        # ── RETURN SUMMARY ───────────────────────────────────────────
        return {
            "total_sources": len(df),
            "anomalous_sources": anomalous_dbs,
            "anomaly_count": len(anomalous_dbs),
            "trend_slope": round(slope, 4),
            "trend_direction": direction,
            "cluster_count": n_clusters,
            "mean_value": round(df["value"].mean(), 2),
            "max_value": float(df["value"].max()),
            "min_value": float(df["value"].min()),
            "std_value": round(df["value"].std(), 2)
        }
    
    async def _insight_agent(self, aggregated_data: dict, ml_analysis: dict) -> str:

        INSIGHT_SYSTEM = (
            "You are a data intelligence analyst. "
            "Given query result statistics and ML analysis, "
            "write a concise 3-5 sentence intelligence report. "
            "Highlight anomalies, trends, and anything operationally significant. "
            "Be specific with numbers. Do not use markdown formatting."
        )

        insight_prompt = (
            f"Query Result Statistics:\n{aggregated_data}\n\n"
            f"ML Analysis:\n{ml_analysis}\n\n"
            f"Intelligence Report:"
        )

        try:
            response = await self.client.aio.models.generate_content(
                model=self.config.llm.model,
                contents=insight_prompt,
                config=types.GenerateContentConfig(system_instruction=INSIGHT_SYSTEM)
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Insight Agent failed: {str(e)}")
            return "-- Insight Agent unavailable"

    async def _run_drift_check(self) -> None:
        # ── FETCH RECENT QUERY LATENCIES FROM AUDIT LOG ──────────────
        async with aiosqlite.connect(os.path.join(os.environ.get("OMNI_DATA_DIR", "."), "omni_audit.db")) as con:
            con.row_factory = aiosqlite.Row
            cursor = con.cursor()
            cursor = await con.execute(
                """SELECT latency_ms, ai_latency_ms, execution_latency_ms, query_cost_usd
                FROM query_logs
                WHERE latency_ms IS NOT NULL
                ORDER BY log_id DESC
                LIMIT ?""",
                (self.config.monitoring.drift_window_size,)
            )
            rows = await cursor.fetchall()

        if len(rows) < 10:
            logger.info("Drift check skipped — insufficient data (need 10+ queries)")
            return

        current_df = pd.DataFrame(
            [dict(row) for row in rows],
            columns=["latency_ms", "ai_latency_ms", "execution_latency_ms", "query_cost_usd"]
        ).fillna(0)


        # ── BASELINE HANDLING ────────────────────────────────────────
        if not os.path.exists("baseline_snapshot.json"):
            # First run — save current stats as baseline and exit
            baseline_stats = current_df.describe().to_dict()
            with open("baseline_snapshot.json", "w") as f:
                json.dump(baseline_stats, f, indent=2)
            logger.info("Drift baseline snapshot saved to baseline_snapshot.json")
            return

        # Load baseline and reconstruct a reference DataFrame from its statistics
        with open("baseline_snapshot.json", "r") as f:
            baseline_stats = json.load(f)

        # Build reference DataFrame from baseline mean and std
        reference_data = {}
        for col in current_df.columns:
            if col in baseline_stats:
                mean = baseline_stats[col].get("mean", 0)
                std  = baseline_stats[col].get("std", 1)
                reference_data[col] = np.random.normal(mean, std, len(current_df))

        reference_df = pd.DataFrame(reference_data)

        # ── EVIDENTLY DRIFT REPORT ───────────────────────────────────
        try:
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=reference_df, current_data=current_df)

            report_dict = report.as_dict()
            drift_detected = report_dict["metrics"][0]["result"]["dataset_drift"]

            if drift_detected:
                logger.error("DATA DRIFT DETECTED in query performance metrics")
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                report_path = f"drift_report_{timestamp}.html"
                report.save_html(report_path)
                logger.info(f"Drift report saved: {report_path}")
            else:
                logger.info("Drift check passed — no significant distribution shift")

        except ImportError:
            logger.warning("Evidently not installed — skipping drift report generation")
        except Exception as e:
            logger.error(f"Drift check failed: {str(e)}")

    def _check_latency_threshold(self, ai_latency_ms: float) -> None:
        if ai_latency_ms > self.config.llm.ai_latency_threshold_ms:
            logger.error(
                f"AI latency alert: {ai_latency_ms}ms exceeded "
                f"threshold of {self.config.llm.ai_latency_threshold_ms}ms"
            )

    def _get_source_mtime(self, source_path: str) -> float:
        try:
            return os.path.getmtime(source_path)
        except OSError:
            return 0.0
        
    async def _log_pipeline_run(self, scanned: int, skipped: int, duration_ms: int) -> None:
        try:
            async with aiosqlite.connect(os.path.join(os.environ.get("OMNI_DATA_DIR", "."), "omni_master_catalog.db")) as db:
                await db.execute(
                    """INSERT INTO pipeline_runs
                    (timestamp, sources_scanned, sources_skipped, duration_ms)
                    VALUES (?, ?, ?, ?)""",
                    (datetime.datetime.now().isoformat(), scanned, skipped, duration_ms)
                )
                await db.commit()
                logger.info(
                    f"Pipeline run: {scanned} scanned, {skipped} skipped, {duration_ms}ms"
                )
        except Exception as e:
            logger.error(f"Pipeline log write failed: {str(e)}")

    async def _send_alert(self, report: str) -> None:

        if not self.config.monitoring.alert_webhook_url:
            logger.warning("Alert webhook URL not configured — skipping alert")
            return

        payload = {
            "text": f"*Axon Anomaly Alert*\n{report}",
            "timestamp": datetime.datetime.now().isoformat()
        }

        try:
            async with httpx.AsyncClient() as http_client:
                response = await http_client.post(
                    self.config.monitoring.alert_webhook_url,
                    json=payload,
                    timeout=10.0
                )
                response.raise_for_status()
                logger.info("Alert sent successfully to webhook")
        except Exception as e:
            logger.error(f"Alert webhook failed: {str(e)}")

    async def _run_health_checks(self) -> None:
        logger.info("Scheduled health check starting")
        health_start = time.time()

        for query in self.config.monitoring.health_check_queries:
            try:
                # Fire the query across all sources
                results = await self.execute_federated_query(query)

                # Run ML analysis on the results
                ml_analysis = self._run_ml_analysis(results)

                # Only alert if anomalies were detected
                if ml_analysis.get("anomaly_count", 0) > 0:

                    success_count = len([r for r in results if r["status"] == "success"])
                    aggregated = {
                        "query": query,
                        "total_databases": len(results),
                        "successful": success_count,
                        "automated": True
                    }

                    report = await self._insight_agent(aggregated, ml_analysis)
                    logger.error(
                        f"Anomaly detected in health check: {ml_analysis['anomaly_count']} "
                        f"anomalous sources for query: {query}"
                    )
                    await self._send_alert(report)

                # Log this automated run to the audit log
                latency_ms = int((time.time() - health_start) * 1000)
                async with aiosqlite.connect(os.path.join(os.environ.get("OMNI_DATA_DIR", "."), "omni_audit.db")) as audit_db:
                    await audit_db.execute(
                        """INSERT INTO query_logs
                        (timestamp, user_prompt, generated_sql, latency_ms, is_automated)
                        VALUES (?, ?, ?, ?, 1)""",
                        (
                            datetime.datetime.now().isoformat(),
                            "AUTOMATED_HEALTH_CHECK",
                            query,
                            latency_ms
                        )
                    )
                    await audit_db.commit()

            except Exception as e:
                logger.error(f"Health check failed for query '{query}': {str(e)}")

        total_ms = int((time.time() - health_start) * 1000)
        logger.info(f"Health check complete in {total_ms}ms")

    def _init_catalog(self) -> None:
        con = sqlite3.connect(os.path.join(os.environ.get("OMNI_DATA_DIR", "."), "omni_master_catalog.db"))
        cur = con.cursor()

        # ── TABLE-LEVEL REGISTRY ──────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS schema_registry (
                source_name        TEXT,
                source_type        TEXT,
                source_path        TEXT,
                table_name         TEXT,
                raw_schema         TEXT,
                schema_notes       TEXT,
                row_count          INTEGER,
                table_description  TEXT,
                last_scanned_mtime REAL,
                PRIMARY KEY (source_name, table_name)
            )
        """)

        # ── COLUMN-LEVEL REGISTRY ─────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS column_registry (
                source_name      TEXT,
                table_name       TEXT,
                column_name      TEXT,
                column_type      TEXT,
                is_primary_key   INTEGER DEFAULT 0,
                is_nullable      INTEGER DEFAULT 1,
                sample_values    TEXT,      -- JSON array of up to 10 distinct values
                distinct_count   INTEGER,
                null_count       INTEGER,
                PRIMARY KEY (source_name, table_name, column_name)
            )
        """)

        # ── PIPELINE RUN LOG ─────────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT,
                sources_scanned INTEGER,
                sources_skipped INTEGER,
                duration_ms     INTEGER
            )
        """)

        # Safe additions for existing catalogs
        safe_additions = [
            ("schema_registry", "source_name",        "TEXT"),
            ("schema_registry", "row_count",         "INTEGER"),
            ("schema_registry", "table_description",  "TEXT"),
            ("schema_registry", "source_type",        "TEXT"),
            ("schema_registry", "source_path",        "TEXT"),
            ("schema_registry", "schema_notes",       "TEXT"),
            ("schema_registry", "last_scanned_mtime", "REAL"),
        ]
        for table, col, col_type in safe_additions:
            try:
                cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
            except Exception:
                pass

        con.commit()
        con.close()
        logger.info("Catalog initialized with schema_registry + column_registry")

    def _describe_table(
        self,
        table_name: str,
        columns: list,
        row_count: int
    ) -> str:
        """
        Generates a plain-English description of what a table stores
        by reading its column names, types, and sample values.
        No AI call — deterministic and instant.
        """
        col_summaries = []
        for col in columns:
            name  = col["column_name"]
            dtype = col["column_type"]
            samples = col["sample_values"]

            if samples:
                sample_str = ", ".join(f"'{v}'" for v in samples[:5])
                col_summaries.append(f"{name} ({dtype}, e.g. {sample_str})")
            else:
                col_summaries.append(f"{name} ({dtype})")

        cols_text = " | ".join(col_summaries)
        return (
            f"Table '{table_name}' stores {row_count:,} rows with columns: {cols_text}."
        )

    def _auto_generate_schema_notes(
        self,
        table_name: str,
        column_data: list,
        row_count: int
    ) -> str:
        """
        Auto-generates schema notes from discovered column metadata.
        Tells the AI about enum-like columns, numeric ranges, and null patterns
        so it can generate accurate SQL without guessing.
        """
        notes = []

        for col in column_data:
            name     = col["column_name"]
            dtype    = col["column_type"].upper() if col["column_type"] else "TEXT"
            samples  = col["sample_values"]
            distinct = col["distinct_count"]
            nulls    = col["null_count"]

            # ── ENUM-LIKE COLUMNS (low cardinality) ──────────────────
            # If a column has ≤ 20 distinct values, list them all
            # This is the most useful hint for the AI
            if 1 < distinct <= 20 and samples:
                values_str = ", ".join(f"'{v}'" for v in samples)
                notes.append(
                    f"The '{name}' column has {distinct} distinct values: {values_str}. "
                    f"Use exact casing in queries."
                )

            # ── BOOLEAN-LIKE COLUMNS ─────────────────────────────────
            elif distinct == 2 and samples:
                notes.append(
                    f"The '{name}' column is boolean-like with values: "
                    f"'{samples[0]}' and '{samples[1]}'."
                )

            # ── NUMERIC RANGE ────────────────────────────────────────
            elif dtype in ("INTEGER", "REAL", "NUMERIC", "FLOAT", "DOUBLE",
                           "int64", "float64") and samples:
                try:
                    numeric_vals = [float(v) for v in samples if v not in (None, "None", "")]
                    if numeric_vals:
                        lo, hi = min(numeric_vals), max(numeric_vals)
                        notes.append(
                            f"The '{name}' column ({dtype}) has sample range "
                            f"{lo:g} to {hi:g} across {distinct} distinct values."
                        )
                except (ValueError, TypeError):
                    pass

            # ── HIGH NULL RATE ───────────────────────────────────────
            if row_count > 0 and nulls > 0:
                null_pct = (nulls / row_count) * 100
                if null_pct > 30:
                    notes.append(
                        f"The '{name}' column is {null_pct:.0f}% NULL — "
                        f"use COALESCE or filter NULLs if needed."
                    )

        if not notes:
            return ""

        return " ".join(notes)

    async def generate_sql(self, user_prompt: str, bypass_cache: bool = False, user_id: str = "default_user") -> dict:
        """
        Phase 1: Validates input, checks cache (unless bypassed), generates SQL.
        Returns the SQL and metadata without executing anything.
        Caller stores this as a pending approval.
        """
        start_time = time.time()

        valid, error_msg = self._validate_input(user_prompt)
        if not valid:
            return {"status": "rejected", "sql": None, "reason": error_msg}

        final_sql = None
        from_cache = False
        agent1_tokens = 0
        agent2_tokens = 0
        query_cost_usd = 0.0
        ai_latency_ms = 0

        intent_key = user_prompt.strip().lower()

        # ── SPINAL REFLEX ────────────────────────────────────────────
        if user_prompt.strip().upper().startswith(("SELECT", "INSERT", "UPDATE", "DELETE")):
            final_sql = user_prompt.strip()
            logger.info("Spinal reflex: raw SQL detected, bypassing AI")

        # ── MEM PALACE CHECK ─────────────────────────────────────────
        if final_sql is None and not bypass_cache:
            cached_sql = await self._check_mem_palace(intent_key)
            if cached_sql is not None:
                final_sql = cached_sql
                from_cache = True
                logger.info(f"MemPalace HIT for: '{intent_key}'")

        # ── SIMILARITY SUGGESTION ─────────────────────────────────────
        if final_sql is None and not bypass_cache:
            similar = await self._find_similar_queries(intent_key)
            if similar:
                top = similar[0]
                logger.info(
                    f"Similar query found (similarity={top['similarity']:.2%}): "
                    f"'{top['intent_key']}'"
                )
                # Return suggestion info — dashboard will show it to operator
                return {
                    "status": "suggestion_available",
                    "suggestion": top,
                    "intent_key": intent_key,
                    "user_prompt": user_prompt,
                    "from_cache": False,
                }

        # ── COGNITIVE BRAIN ──────────────────────────────────────────
        if final_sql is None:
            ai_start = time.time()
            final_sql, agent1_tokens, agent2_tokens, query_cost_usd = \
                await self.translate_intent_to_sql(user_prompt, user_id=user_id)
            ai_latency_ms = int((time.time() - ai_start) * 1000)
            self._check_latency_threshold(ai_latency_ms)

            if final_sql.startswith("-- ERROR:"):
                return {"status": "error", "sql": None, "reason": final_sql}

        generation_latency_ms = int((time.time() - start_time) * 1000)

        return {
            "status": "pending_approval",
            "user_id": user_id,
            "sql": final_sql,
            "intent_key": intent_key,
            "user_prompt": user_prompt,
            "from_cache": from_cache,
            "agent1_tokens": agent1_tokens,
            "agent2_tokens": agent2_tokens,
            "query_cost_usd": query_cost_usd,
            "ai_latency_ms": ai_latency_ms,
            "generation_latency_ms": generation_latency_ms,
        }


    async def execute_approved_query(self, pending: dict) -> dict:
        """
        Phase 2: Executes a previously generated and human-approved SQL query.
        Called after the operator approves via the dashboard or API.
        """
        execution_start = time.time()

        final_sql   = pending["sql"]
        intent_key  = pending["intent_key"]
        from_cache  = pending["from_cache"]
        agent1_tokens = pending.get("agent1_tokens", 0)
        agent2_tokens = pending.get("agent2_tokens", 0)
        query_cost_usd = pending.get("query_cost_usd", 0.0)
        ai_latency_ms = pending.get("ai_latency_ms", 0)

        # ── MEM PALACE WRITE ─────────────────────────────────────────
        if not from_cache:
            await self._store_in_mem_palace(intent_key, final_sql)

        # ── MOTOR SYSTEM ─────────────────────────────────────────────
        motor_start = time.time()
        user_id = pending.get("user_id", "default_user")
        results = await self.execute_federated_query(final_sql, impersonation_user_id=user_id)
        execution_latency_ms = int((time.time() - motor_start) * 1000)

        # ── ML ANALYSIS ──────────────────────────────────────────────
        success_results = [r for r in results if r["status"] == "success"]
        error_count = len(results) - len(success_results)

        aggregated_data = {
            "total_databases": len(results),
            "successful_databases": len(success_results),
            "failed_databases": error_count,
            "sql_used": final_sql
        }

        ml_analysis = self._run_ml_analysis(results)
        logger.info(f"ML analysis: {ml_analysis.get('anomaly_count', 0)} anomalies")

        # ── INSIGHT AGENT ─────────────────────────────────────────────
        insight_report = await self._insight_agent(aggregated_data, ml_analysis)
        logger.info("Insight Agent report generated")

        # ── AUDIT LOG ────────────────────────────────────────────────
        total_latency_ms = int((time.time() - execution_start) * 1000)
        async with aiosqlite.connect(os.path.join(os.environ.get("OMNI_DATA_DIR", "."), "omni_audit.db")) as audit_db:
            await audit_db.execute(
                """INSERT INTO query_logs
                (timestamp, user_prompt, generated_sql, latency_ms,
                    ai_latency_ms, execution_latency_ms,
                    agent1_tokens, agent2_tokens, query_cost_usd, from_cache)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.datetime.now().isoformat(),
                    intent_key,
                    final_sql,
                    total_latency_ms,
                    ai_latency_ms,
                    execution_latency_ms,
                    agent1_tokens,
                    agent2_tokens,
                    query_cost_usd,
                    1 if from_cache else 0,
                )
            )
            await audit_db.commit()

        return {
            "status": "success",
            "sql": final_sql,
            "from_cache": from_cache,
            "latency_ms": total_latency_ms,
            "ai_latency_ms": ai_latency_ms,
            "execution_latency_ms": execution_latency_ms,
            "agent1_tokens": agent1_tokens,
            "agent2_tokens": agent2_tokens,
            "query_cost_usd": query_cost_usd,
            "results": results,
            "ml_analysis": ml_analysis,
            "insight_report": insight_report,
        }