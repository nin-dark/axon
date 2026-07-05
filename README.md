# Axon — Federated Cognitive Data Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Gemini](https://img.shields.io/badge/Gemini-2.5-orange?logo=google)](https://ai.google.dev/)

Axon is a production-grade, federated text-to-SQL cognitive engine built from scratch — no LangChain, no monolithic wrappers. You describe a question in plain English. Axon routes it through a multi-agent AI pipeline, generates executable SQL, gates every execution through a human approval step, fires the query concurrently across every registered data source, and returns a natural language intelligence report alongside ML-analyzed results.

Point it at any combination of SQLite files, CSV datasets, REST APIs, or enterprise databases. Everything is configured in a single `config.yaml`.

---

## Architecture

```text
Natural Language Prompt
        │
        ▼
┌───────────────────────┐
│   Input Validation    │  SQL injection detection, length limits
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│   Semantic MemPalace  │  FAISS vector search → exact match fallback
│   Cache Check         │  TF-IDF suggestion for moderate similarity
└──────────┬────────────┘
           │ MISS
           ▼
┌───────────────────────┐
│   Agent 1 — Router    │  Gemini 2.5 Flash — extracts table name only
│   (Cognitive Router)  │
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│   O(1) Scout          │  Catalog lookup — fetches rich schema with
│   Schema Discovery    │  column profiles, sample values, auto-notes
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│   Agent 2 — SQL Gen   │  Gemini 2.5 Pro — generates SQL from schema
│   + PII Masking       │
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│   Human-in-the-Loop   │  Dashboard approval (approve / reject / correct)
│   Conscious Gate      │  Corrections feed back into MemPalace (active learning)
└──────────┬────────────┘
           │ Approved
           ▼
┌───────────────────────┐
│   Motor System        │  Async scatter-gather across all registered sources
│   Federated Execution │  Streams results via WebSocket as each source responds
│   + Anomaly Gate      │  Blocks Cartesian products > configured row limit
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│   ML Analysis Layer   │  IsolationForest anomaly detection
│                       │  DBSCAN behavioral clustering
│                       │  Linear regression trend detection
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│   Agent 3 — Insight   │  Gemini — synthesizes a 3–5 sentence
│   Intelligence Report │  intelligence report from ML output
└──────────┬────────────┘
           │
           ▼
    Structured Response
    + Audit Log + Cost Tracking
```

---

## Key Features

### Multi-Agent SQL Pipeline

- **Agent 1 (Router)** runs on `gemini-2.5-flash` — extracts only the table name, keeping the prompt narrow to eliminate hallucination
- **Agent 2 (SQL Generator)** runs on `gemini-2.5-pro` — receives a rich schema profile with actual column values, not just a CREATE TABLE string
- **Agent 3 (Insight)** runs on `gemini-2.5-flash-lite` — generates a natural language intelligence report from the ML analysis output
- Each agent uses a different model, configurable per-role in `config.yaml`

### Semantic MemPalace Cache

- **FAISS vector index** with `all-MiniLM-L6-v2` embeddings for semantic similarity matching — "show failed transactions" and "how many transactions failed" resolve to the same cached SQL
- **Exact match O(1) lookup** runs before the FAISS search for instant retrieval of previously approved queries
- **TF-IDF suggestion layer** catches moderate-similarity matches that fall below the FAISS threshold and surfaces them to the operator for manual review
- Cache hits have provably zero AI cost — tracked in real-time via `/metrics`

### Active Learning

When an operator corrects an AI-generated SQL query during approval, the corrected SQL is embedded and stored in the MemPalace Vault. All future semantically similar queries will return this human-verified SQL instead of calling the AI. The system improves with every correction.

### Human-in-the-Loop Approval Gate

Every SQL query — whether AI-generated, cache-retrieved, or raw SQL — requires explicit operator approval before execution. The gate operates entirely through the REST API and WebSocket interface, with no terminal dependency. Operators can approve, reject, or edit the SQL before it runs.

### Federated Multi-Source Execution

Axon federates queries across heterogeneous data sources simultaneously using `asyncio.gather` and `asyncio.as_completed` for streaming. Supported source types:

| Type            | Description                                                                                            |
| --------------- | ------------------------------------------------------------------------------------------------------ |
| `sqlite`        | Single `.db` file                                                                                      |
| `sqlite_folder` | Scans a directory for all `.db` files                                                                  |
| `csv`           | CSV file queried in-memory via DuckDB                                                                  |
| `rest`          | REST API returning JSON, schema inferred from sample response                                          |
| `enterprise`    | Production databases via SQLAlchemy connection pool (supports enterprise/industrial database backends) |

Sources are auto-detected from path/URL if `type` is not specified. `name` is auto-derived from the filename.

### Rich Schema Catalog

The metadata pipeline builds a two-level catalog at startup:

- **`schema_registry`** — one row per table with row count, raw DDL, and a plain-English auto-generated description
- **`column_registry`** — one row per column with data type, primary key flag, up to 10 distinct sample values, distinct value count, and null count

The catalog is **incremental** — sources unchanged since the last scan (compared by `mtime`) are skipped entirely. Subsequent restarts complete in milliseconds.

### ML Analysis on Aggregated Results

After federated execution, three scikit-learn analyses run on the aggregated numerical output:

- **IsolationForest** — identifies databases returning statistically anomalous values
- **DBSCAN** — clusters databases by behavioral similarity
- **LinearRegression** — computes trend direction (up / flat / down) across sources

### MLOps Observability

- **Evidently drift detection** runs on a scheduled interval, comparing recent query performance metrics against a saved baseline snapshot and generating an HTML report when drift is detected
- **APScheduler** runs health check queries, drift checks, and expired approval cleanup on configurable intervals
- **Webhook alerts** fire when anomalies are detected during automated health checks
- **`/metrics`** exposes live operational statistics including cache hit rate, p90 latency, AI spend, and cache savings

### Privacy and Security

- **PII column masking** — configurable list of column names that Agent 2 is instructed to wrap in `SHA256()` when selected
- **SQL injection prevention** — configurable signal list blocks injection patterns in natural language prompts
- **Anomaly gate** — blocks queries returning more rows than a configured maximum, catching AI-hallucinated Cartesian products before they reach the application
- **API key authentication** via `X-API-Key` header on all endpoints, with WebSocket key support via query string

---

## Project Structure

```text
axon/
├── engine.py              # AxonEngine — all cognitive and data logic
├── main.py                # FastAPI application, endpoints, WebSocket handler
├── config.yaml            # Single configuration file — edit this to get started
├── config_loader.py       # Typed config parsing with auto-detection
├── models.py              # Pydantic request/response schemas
├── enterprise_db.py       # SQLAlchemy connection pool for enterprise databases
├── connectors/
│   ├── base_connector.py  # Abstract connector interface
│   ├── sqlite_connector.py
│   ├── csv_connector.py
│   └── rest_connector.py
├── dashboard/
│   ├── index.html         # Live streaming approval dashboard
│   ├── app.js             # WebSocket client, approval flow, metrics panel
│   └── style.css
├── requirements.txt
└── .env.example
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/)

### Installation

```bash
git clone https://github.com/nin-dark/axon.git
cd axon

pip install -r requirements.txt

cp .env.example .env
# Add your GEMINI_API_KEY to .env
```

### Configuration

Edit `config.yaml` to point Axon at your data. The minimal configuration is:

```yaml
data_sources:
  - path: /path/to/your/database.db
```

Type and name are auto-detected. For a folder of databases:

```yaml
data_sources:
  - path: /path/to/your/databases/
```

For a CSV file:

```yaml
data_sources:
  - path: /path/to/data.csv
```

### Run

```bash
python main.py
```

The server starts on `http://localhost:8000`. Open the interactive API docs at:

```
http://localhost:8000/docs
```

Open the live dashboard at `dashboard/index.html` in your browser.

---

## Configuration Reference

All system behaviour is controlled through `config.yaml`. No code changes are required to switch models, add data sources, or tune thresholds.

```yaml
# ── LLM ──────────────────────────────────────────────────────────────────────
llm:
  router_model: gemini-2.5-flash # Agent 1 — table name extraction
  sql_model: gemini-2.5-pro # Agent 2 — SQL generation
  model: gemini-2.5-flash-lite # Agent 3 — insight reports
  input_cost_per_million_tokens: 0.10
  output_cost_per_million_tokens: 0.40
  ai_latency_threshold_ms: 3000 # Alert threshold for slow AI responses

# ── CACHE ─────────────────────────────────────────────────────────────────────
cache:
  embedding_model: all-MiniLM-L6-v2
  embedding_dim: 384
  semantic_similarity_threshold: 0.70 # FAISS similarity threshold (0–1)
  tfidf_threshold: 0.40 # TF-IDF suggestion threshold

# ── ENGINE ────────────────────────────────────────────────────────────────────
engine:
  max_concurrent_connections: 50 # Semaphore limit for DB connections
  max_prompt_length: 500
  approval_timeout_seconds: 300 # Pending approvals expire after this
  max_rows_per_query: 5000 # Anomaly gate row limit

# ── MONITORING ────────────────────────────────────────────────────────────────
monitoring:
  health_check_interval_hours: 1
  drift_window_size: 100
  alert_webhook_url: "" # Slack/Discord/custom webhook URL
  health_check_queries:
    - "SELECT COUNT(*) FROM transactions WHERE status = 'Failed'"

# ── PRIVACY ───────────────────────────────────────────────────────────────────
privacy:
  disable_sample_values: true # Hide column samples from schema view
  masked_columns: # Agent 2 wraps these in SHA256()
    - credit_card
    - ssn
    - email
  user_contexts: # Per-user SQL generation constraints
    analyst_nyc: "Only generate SQL for region='NYC' data."

# ── DATA SOURCES ──────────────────────────────────────────────────────────────
data_sources:
  # Auto-detected sqlite folder:
  - path: ./databases/

  # Explicit single SQLite database:
  - name: sales
    type: sqlite
    path: /var/data/sales.db
    schema_notes: "The region column values are: 'North', 'South', 'East', 'West'"

  # CSV file:
  - name: quarterly_report
    type: csv
    path: /var/data/Q1_2026.csv

  # REST API:
  - name: live_metrics
    type: rest
    url: https://api.internal.com/v1/metrics
    auth_token: your_token_here

  # Enterprise:
  - name: production_db
    type: enterprise
    path: "postgresql+psycopg2://user:pass@host:5432/dbname"
```

---

## API Reference

All endpoints require `X-API-Key: <your_key>` when `OMNI_API_KEY` is set in the environment.

### Query Lifecycle

Queries follow a two-phase lifecycle. Phase 1 generates SQL and returns a pending approval ID. Phase 2 executes after the operator approves.

#### `POST /query`

Submit a natural language prompt. Returns immediately with either `pending_approval`, `suggestion_available`, `rejected`, or `error`.

```json
// Request
{ "prompt": "count failed transactions by region" }

// Response — pending_approval
{
  "status": "pending_approval",
  "approval_id": "3f9e2a1b-...",
  "sql": "SELECT region, COUNT(*) FROM transactions WHERE status = 'Failed' GROUP BY region",
  "from_cache": false,
  "query_cost_usd": 0.00004812,
  "ai_latency_ms": 1840
}
```

#### `POST /approve/{approval_id}`

Approve or reject a pending query.

```json
// Approve
{ "approved": true }

// Response — full execution result
{
  "status": "success",
  "sql": "SELECT region, COUNT(*) ...",
  "from_cache": false,
  "latency_ms": 2340,
  "execution_latency_ms": 498,
  "results": [...],
  "insight_report": "Across 180 databases, 18% of transactions failed. ...",
  "ml_analysis": {
    "anomaly_count": 7,
    "trend_direction": "up",
    "cluster_count": 3
  }
}
```

#### `POST /accept-suggestion/{approval_id}`

Respond to a TF-IDF cache suggestion.

```json
{
  "use_suggestion": true,
  "suggestion_sql": "SELECT ...",
  "intent_key": "count failed transactions"
}
```

#### `GET /pending-approvals`

List all queries currently awaiting operator approval.

### WebSocket — `/ws/query`

The WebSocket endpoint streams results as each data source responds, eliminating the wait for all sources to finish.

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/query?api_key=your_key");
ws.onopen = () =>
  ws.send(JSON.stringify({ prompt: "count failed transactions" }));

ws.onmessage = ({ data }) => {
  const msg = JSON.parse(data);
  // msg.type === "approval_required"  →  show SQL to operator
  // msg.type === "db_result"          →  one source responded
  // msg.type === "complete"           →  all done, insight_report available
  // msg.type === "suggestion"         →  TF-IDF match found
};

// Approve:
ws.send(JSON.stringify({ approved: true }));

// Correct and approve (active learning):
ws.send(JSON.stringify({ approved: true, corrected_sql: "SELECT ..." }));
```

### Observability Endpoints

| Endpoint                       | Description                                      |
| ------------------------------ | ------------------------------------------------ |
| `GET /health`                  | Server status, active source count, vault size   |
| `GET /metrics`                 | Cache hit rate, p90 latency, AI spend, savings   |
| `GET /vault`                   | All cached queries sorted by hit count           |
| `GET /logs`                    | Recent query audit log                           |
| `GET /schema`                  | All sources, tables, and row counts              |
| `GET /schema/{source}/{table}` | Full column profile with types and sample values |
| `GET /sources`                 | All registered data sources with table counts    |
| `GET /sources/{source_name}`   | Tables in a specific source                      |
| `GET /pipeline-status`         | Catalog scan history                             |
| `GET /scheduler-status`        | Recent automated health check runs               |

---

## Environment Variables

| Variable            | Required | Description                                                                |
| ------------------- | -------- | -------------------------------------------------------------------------- |
| `GEMINI_API_KEY`    | **Yes**  | Google Gemini API key                                                      |
| `OMNI_API_KEY`      | No       | API key for endpoint authentication. If unset, all requests are allowed    |
| `OMNI_DATA_DIR`     | No       | Directory for SQLite system databases. Defaults to `.` (current directory) |
| `ALERT_WEBHOOK_URL` | No       | Overrides `monitoring.alert_webhook_url` in config                         |
| `GCP_PROJECT_ID`    | No       | GCP project ID for Secret Manager integration                              |

---

## Tech Stack

| Layer          | Technology                                                             |
| -------------- | ---------------------------------------------------------------------- |
| AI             | Google Gemini 2.5 Flash, 2.5 Flash Lite, 2.5 Pro via `google-genai`    |
| API            | FastAPI, Uvicorn, WebSockets                                           |
| Data           | aiosqlite, DuckDB, SQLAlchemy (enterprise), pandas, httpx              |
| Semantic Cache | FAISS, sentence-transformers (`all-MiniLM-L6-v2`), scikit-learn TF-IDF |
| ML Analysis    | scikit-learn (IsolationForest, DBSCAN, LinearRegression)               |
| MLOps          | Evidently, APScheduler                                                 |
| Infra          | Docker, GCP Cloud Run, GCP Secret Manager                              |
| Config         | PyYAML, python-dotenv                                                  |

---

## How the Schema Intelligence Works

Unlike single-call text-to-SQL approaches that hallucinate column names and value casing, Axon grounds Agent 2 in real data before generating a single character of SQL.

During the catalog scan, Axon queries each table directly to extract:

- All column names and declared types
- Up to 10 distinct sample values per column
- Distinct value count and null percentage

For low-cardinality columns (≤20 distinct values), `_auto_generate_schema_notes` automatically produces notes like:

> _The 'status' column has 3 distinct values: 'Success', 'Failed', 'Pending'. Use exact casing in queries._

These notes are stored in the catalog and injected into the Agent 2 prompt at query time. The result is that Agent 2 always knows the exact casing of enum values, numeric ranges, and null patterns — without any manual configuration.

The `privacy.disable_sample_values` flag controls whether sample values appear in the public `/schema` endpoint. It does not affect what Agent 2 receives — schema notes generated at scan time are always passed to the AI.

---

## Under Development

These features are partially implemented and actively being worked on for the next major release:

- **Per-User Access Control (NL-RLS)**: User context injection (e.g., restrict to a region, hide compensation columns) is scaffolded in the config and injected as a critical system instruction to Agent 2. Full end-to-end RBAC integration across the REST/WebSocket layer is pending.
- **Enterprise DB Session Impersonation**: Per-user session impersonation (`SET ROLE` / `RESET ROLE`) is pending a mature authentication integration.

---

## Roadmap

- [ ] Connect `user_id` to a real authentication provider (OAuth / LDAP).
- [ ] Add daily AI spend caps with automatic circuit-breaking when the budget is exceeded.
- [ ] Implement query result caching with a configurable TTL to bypass the database execution phase for identical approved queries.
- [ ] Add native connectors for specialized industrial databases (e.g., via ODBC).
- [ ] Export Prometheus-compatible metrics for external monitoring dashboards.

---

## Author

Built by Nikesh — [GitHub](https://github.com/nin-dark) · [LinkedIn](https://www.linkedin.com/in/nikesh-patel-200452362/)
