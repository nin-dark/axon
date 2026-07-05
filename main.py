import os
import uuid
import sqlite3
import logging
import asyncio
import datetime
import time
import aiosqlite
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

from config_loader import load_config
from engine import AxonEngine
from models import (
    QueryRequest, ApprovalRequest, AcceptSuggestionRequest,
    PendingApprovalResponse, QueryResponse, SingleDBResult
)

logger = logging.getLogger("axon.api")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)):
    expected_key = os.environ.get("OMNI_API_KEY")
    if expected_key and api_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid X-API-Key")
    return api_key

# ── GLOBAL STATE ──────────────────────────────────────────────────────────────

engine: AxonEngine = None
config = None

# In-memory approval store
# Maps approval_id → pending dict from generate_sql()
# In production with multiple instances, replace with Redis
pending_approvals: Dict[str, dict] = {}

async def _cleanup_expired_approvals():
    """Remove pending approvals that have exceeded the timeout."""
    if not config:
        return
    timeout = config.engine.approval_timeout_seconds
    now = time.time()
    expired = [
        aid for aid, data in pending_approvals.items()
        if now - data.get("_created_at", 0) > timeout
    ]
    for aid in expired:
        pending_approvals.pop(aid, None)
    if expired:
        logger.info(f"Cleaned up {len(expired)} expired pending approvals")


# ── LIFESPAN ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, config

    config = load_config("config.yaml")
    engine = AxonEngine(config)
    await engine.metadata()
    logger.info("Axon Engine online")

    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        engine._run_health_checks,
        trigger="interval",
        hours=config.monitoring.health_check_interval_hours,
        id="health_check"
    )
    scheduler.add_job(
        engine._run_drift_check,
        trigger="interval",
        hours=config.monitoring.health_check_interval_hours,
        id="drift_check"
    )
    scheduler.add_job(
        _cleanup_expired_approvals,
        trigger="interval",
        minutes=1,
        id="approval_cleanup"
    )
    scheduler.start()
    logger.info("Health check scheduler started")

    yield

    scheduler.shutdown(wait=False)
    logger.info("Axon Engine shutting down")


app = FastAPI(
    title="Axon",
    description="Federated Cognitive Data Intelligence Platform",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── HELPER ────────────────────────────────────────────────────────────────────

def db_path(filename: str) -> str:
    return os.path.join(os.environ.get("OMNI_DATA_DIR", "."), filename)

async def sqlite_rows_to_dicts(db_path: str, query: str, params: tuple = ()) -> list:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


# ── POST /query — Phase 1: Generate SQL, Return Pending Approval ──────────────

@app.post("/query", response_model=PendingApprovalResponse, dependencies=[Depends(verify_api_key)])
async def submit_query(request: QueryRequest):
    """
    Generates SQL from the prompt but does not execute it.
    Returns an approval_id. Call POST /approve/{approval_id} to execute.
    """
    result = await engine.generate_sql(request.prompt)

    if result["status"] in ("rejected", "error"):
        return PendingApprovalResponse(
            status=result["status"],
            reason=result.get("reason")
        )

    if result["status"] == "suggestion_available":
        # Return suggestion to operator — they decide via /accept-suggestion
        approval_id = str(uuid.uuid4())
        result["_created_at"] = time.time()
        pending_approvals[approval_id] = result
        return PendingApprovalResponse(
            status="suggestion_available",
            approval_id=approval_id,
            suggestion=result.get("suggestion"),
            intent_key=result.get("intent_key"),
        )

    # Normal case — SQL generated, store as pending approval
    approval_id = str(uuid.uuid4())
    result["_created_at"] = time.time()
    pending_approvals[approval_id] = result

    return PendingApprovalResponse(
        status="pending_approval",
        approval_id=approval_id,
        sql=result["sql"],
        from_cache=result["from_cache"],
        query_cost_usd=result.get("query_cost_usd", 0.0),
        ai_latency_ms=result.get("ai_latency_ms", 0),
    )


# ── POST /approve/{id} — Phase 2: Human Approves, Engine Executes ─────────────

@app.post("/approve/{approval_id}", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
async def approve_query(approval_id: str, request: ApprovalRequest):
    """
    Approves or rejects a pending query.
    If approved, executes across all data sources and returns full results.
    """
    pending = pending_approvals.pop(approval_id, None)
    if not pending:
        raise HTTPException(
            status_code=404,
            detail="Approval ID not found or expired. The query may have timed out."
        )

    if not request.approved:
        logger.info(f"Query rejected by operator: {pending.get('sql', '')[:60]}...")
        return QueryResponse(status="rejected", sql=pending.get("sql"))

    result = await engine.execute_approved_query(pending)
    db_results = [
        SingleDBResult(**r) for r in result.get("results", [])
    ]

    return QueryResponse(
        status=result["status"],
        sql=result["sql"],
        from_cache=result["from_cache"],
        latency_ms=result["latency_ms"],
        ai_latency_ms=result["ai_latency_ms"],
        execution_latency_ms=result["execution_latency_ms"],
        agent1_tokens=result["agent1_tokens"],
        agent2_tokens=result["agent2_tokens"],
        query_cost_usd=result["query_cost_usd"],
        results=db_results,
        insight_report=result["insight_report"],
        ml_analysis=result["ml_analysis"],
    )


# ── POST /accept-suggestion — Operator Accepts a TF-IDF Suggestion ───────────

@app.post("/accept-suggestion/{approval_id}", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
async def accept_suggestion(approval_id: str, request: AcceptSuggestionRequest):
    """
    Operator accepts or declines a TF-IDF cache suggestion.
    If accepted, uses suggestion SQL. If declined, AI is called.
    """
    pending = pending_approvals.pop(approval_id, None)
    if not pending:
        raise HTTPException(status_code=404, detail="Approval ID not found")

    if request.use_suggestion and request.suggestion_sql:
        # Operator accepted suggestion — build pending with suggestion SQL
        exec_pending = {
            "sql": request.suggestion_sql,
            "intent_key": request.intent_key,
            "from_cache": True,
            "agent1_tokens": 0,
            "agent2_tokens": 0,
            "query_cost_usd": 0.0,
            "ai_latency_ms": 0,
        }
    else:
        # Operator declined — call the AI
        ai_result = await engine.generate_sql(
            pending.get("user_prompt") or pending.get("intent_key", "")
        )
        if ai_result["status"] in ("rejected", "error"):
            return QueryResponse(
                status=ai_result["status"],
                reason=ai_result.get("reason")
            )
        exec_pending = ai_result

    # Log suggestion decision
    try:
        async with aiosqlite.connect(db_path("omni_audit.db")) as audit_db:
            await audit_db.execute(
                "INSERT INTO suggestion_log VALUES (NULL, ?, ?, ?, ?, ?)",
                (
                    datetime.datetime.now().isoformat(),
                    pending.get("user_prompt") or pending.get("intent_key", ""),
                    request.intent_key if request.use_suggestion else "",
                    0.0,
                    1 if request.use_suggestion else 0,
                )
            )
            await audit_db.commit()
    except Exception as e:
        logger.error(f"Suggestion log write failed: {str(e)}")

    result = await engine.execute_approved_query(exec_pending)
    db_results = [SingleDBResult(**r) for r in result.get("results", [])]

    return QueryResponse(
        status=result["status"],
        sql=result["sql"],
        from_cache=result["from_cache"],
        latency_ms=result["latency_ms"],
        results=db_results,
        insight_report=result["insight_report"],
        ml_analysis=result["ml_analysis"],
    )


# ── GET /pending-approvals — List All Waiting Approvals ──────────────────────

@app.get("/pending-approvals", dependencies=[Depends(verify_api_key)])
async def list_pending_approvals():
    return [
        {
            "approval_id": aid,
            "sql": data.get("sql"),
            "from_cache": data.get("from_cache"),
            "cost_usd": data.get("query_cost_usd", 0),
            "intent": data.get("intent_key"),
        }
        for aid, data in pending_approvals.items()
    ]


# ── WEBSOCKET /ws/query — Streaming with Approval Flow ───────────────────────

@app.websocket("/ws/query")
async def ws_query(
    websocket: WebSocket, 
    api_key: str = None, 
    user_id: str = "default_user"
):
    # Enforce API Key for WebSocket
    expected_key = os.environ.get("OMNI_API_KEY")
    # Browsers can't easily send headers in WebSocket, so we allow it in query string too
    client_key = websocket.headers.get("X-API-Key") or api_key
    if expected_key and client_key != expected_key:
        await websocket.close(code=1008, reason="Invalid API Key")
        return

    await websocket.accept()
    approval_timeout = config.engine.approval_timeout_seconds if config else 300

    try:
        data = await websocket.receive_json()
        prompt = data.get("prompt", "").strip()

        # Phase 1: Generate SQL
        result = await engine.generate_sql(prompt, user_id=user_id)

        if result["status"] in ("rejected", "error"):
            await websocket.send_json({
                "type": "error",
                "message": result.get("reason", "Query failed")
            })
            return

        if result["status"] == "suggestion_available":
            # Send suggestion to dashboard for operator decision
            await websocket.send_json({
                "type": "suggestion",
                "suggestion": result["suggestion"],
                "intent_key": result["intent_key"],
            })
            # Wait for operator response
            decision = await asyncio.wait_for(
                websocket.receive_json(),
                timeout=approval_timeout
            )
            if decision.get("use_suggestion"):
                final_sql = decision["suggestion_sql"]
                exec_pending = {
                    "sql": final_sql,
                    "intent_key": result["intent_key"],
                    "from_cache": True,
                    "agent1_tokens": 0,
                    "agent2_tokens": 0,
                    "query_cost_usd": 0.0,
                    "ai_latency_ms": 0,
                }
            else:
                # Operator wants AI — generate fresh
                result = await engine.generate_sql(result["intent_key"], bypass_cache=True, user_id=user_id)
                if result["status"] != "pending_approval":
                    await websocket.send_json({"type": "error", "message": "AI generation failed"})
                    return
                exec_pending = result
        else:
            exec_pending = result

        # Send SQL to client for approval
        await websocket.send_json({
            "type": "approval_required",
            "sql": exec_pending["sql"],
            "from_cache": exec_pending["from_cache"],
            "cost_usd": exec_pending.get("query_cost_usd", 0.0),
            "ai_latency_ms": exec_pending.get("ai_latency_ms", 0),
        })

        # Wait for approval from client
        approval_msg = await asyncio.wait_for(
            websocket.receive_json(),
            timeout=approval_timeout
        )

        if not approval_msg.get("approved"):
            await websocket.send_json({"type": "rejected"})
            return

        # Handle Active Learning (Human-in-the-Loop Correction)
        if approval_msg.get("corrected_sql"):
            custom_sql = approval_msg["corrected_sql"]
            is_valid, _ = engine._validate_input(custom_sql)
            
            if not is_valid and not approval_msg.get("force_destructive"):
                await websocket.send_json({"type": "destructive_warning"})
                return
                
            exec_pending["sql"] = custom_sql
            await engine.learn_correction(exec_pending["intent_key"], exec_pending["sql"])
        elif not exec_pending["from_cache"]:
            # Store in vault before streaming if it wasn't from cache or corrected
            await engine._store_in_mem_palace(
                exec_pending["intent_key"], exec_pending["sql"]
            )

        ws_exec_start = time.time()
        all_results = []
        completed = 0
        total = len(await engine._get_registered_sources())

        async for result in engine.execute_federated_query_stream(exec_pending["sql"], impersonation_user_id=user_id):
            completed += 1
            all_results.append(result)
            await websocket.send_json({
                "type": "db_result",
                "database": result["database"],
                "status": result["status"],
                "data": result.get("data"),
                "msg": result.get("msg"),
                "completed": completed,
                "total": total,
                "progress_pct": round((completed / total) * 100, 1),
            })

        # ML analysis and insight
        success_count = len([r for r in all_results if r["status"] == "success"])
        ml_analysis = engine._run_ml_analysis(all_results)
        insight_report = await engine._insight_agent(
            {"total": total, "successful": success_count, "sql": exec_pending["sql"]},
            ml_analysis
        )

        await websocket.send_json({
            "type": "complete",
            "ml_analysis": ml_analysis,
            "insight_report": insight_report,
            "total_completed": completed,
            "completed": True,
        })

        # Log to audit
        ws_latency_ms = int((time.time() - ws_exec_start) * 1000)
        async with aiosqlite.connect(db_path("omni_audit.db")) as audit_db:
            await audit_db.execute(
                """INSERT INTO query_logs
                   (timestamp, user_prompt, generated_sql, latency_ms,
                    ai_latency_ms, execution_latency_ms,
                    agent1_tokens, agent2_tokens, query_cost_usd, from_cache)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.datetime.now().isoformat(),
                    exec_pending["intent_key"],
                    exec_pending["sql"],
                    ws_latency_ms,
                    exec_pending.get("ai_latency_ms", 0),
                    ws_latency_ms - exec_pending.get("ai_latency_ms", 0),
                    exec_pending.get("agent1_tokens", 0),
                    exec_pending.get("agent2_tokens", 0),
                    exec_pending.get("query_cost_usd", 0.0),
                    1 if exec_pending["from_cache"] else 0,
                )
            )
            await audit_db.commit()

    except asyncio.TimeoutError:
        await websocket.send_json({
            "type": "timeout",
            "message": f"Approval timed out after {approval_timeout} seconds"
        })
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass   


# ── READ-ONLY GET ENDPOINTS ───────────────────────────────────────────────────

@app.get("/vault", dependencies=[Depends(verify_api_key)])
async def get_vault():
    return await sqlite_rows_to_dicts(
        db_path("mem_palace_vault.db"),
        "SELECT intent_key, approved_sql, hit_count, last_used FROM vault ORDER BY hit_count DESC"
    )

@app.get("/logs", dependencies=[Depends(verify_api_key)])
async def get_audit_logs(limit: int = 50):
    return await sqlite_rows_to_dicts(
        db_path("omni_audit.db"),
        "SELECT * FROM query_logs ORDER BY log_id DESC LIMIT ?",
        (limit,)
    )

@app.get("/health", dependencies=[Depends(verify_api_key)])
async def health_check():
    return {
        "status": "alive",
        "version": "2.0.0",
        "sources": len(await engine._get_registered_sources()) if engine else 0,
        "vault_size": (await sqlite_rows_to_dicts(
            db_path("mem_palace_vault.db"), "SELECT COUNT(*) as c FROM vault"
        ))[0]["c"],
    }

# ── GET /metrics ──────────────────────────────────────────────────────────────
@app.get("/metrics", dependencies=[Depends(verify_api_key)])
async def get_metrics():
    result = {}

    try:
        # ── AUDIT DB ─────────────────────────────────────────────
        async with aiosqlite.connect(db_path("omni_audit.db")) as con:
            con.row_factory = aiosqlite.Row

            # ── TOTAL QUERIES ────────────────────────────────────────
            cursor = await con.execute(
                "SELECT COUNT(*) FROM query_logs"
            )
            total = (await cursor.fetchone())[0]

            if total == 0:
                return {"message": "No queries logged yet"}

            # ── CACHE STATS ──────────────────────────────────────────
            cursor = await con.execute(
                "SELECT COUNT(*) FROM query_logs WHERE from_cache = 1"
            )
            cache_hits = (await cursor.fetchone())[0]

            cache_hit_rate = round((cache_hits / total) * 100, 2)

            # ── LATENCY STATS ────────────────────────────────────────
            cursor = await con.execute(
                "SELECT latency_ms, ai_latency_ms FROM query_logs WHERE latency_ms IS NOT NULL"
            )
            latency_rows = await cursor.fetchall()

            all_latencies    = [r["latency_ms"] for r in latency_rows if r["latency_ms"]]
            all_ai_latencies = [r["ai_latency_ms"] for r in latency_rows if r["ai_latency_ms"]]

            avg_latency    = round(sum(all_latencies) / len(all_latencies), 2) if all_latencies else 0
            avg_ai_latency = round(sum(all_ai_latencies) / len(all_ai_latencies), 2) if all_ai_latencies else 0

            p90_latency = sorted(all_latencies)[int(len(all_latencies) * 0.90)] if all_latencies else 0

            # ── COST STATS ───────────────────────────────────────────
            cursor = await con.execute(
                """SELECT SUM(query_cost_usd), AVG(query_cost_usd)
                   FROM query_logs
                   WHERE from_cache = 0 AND query_cost_usd IS NOT NULL"""
            )
            cost_row = await cursor.fetchone()

            total_spend      = round(cost_row[0] or 0, 8)
            avg_cost         = round(cost_row[1] or 0, 8)
            cache_savings    = round(cache_hits * avg_cost, 8)

            # ── ERROR RATE ───────────────────────────────────────────
            cursor = await con.execute(
                "SELECT COUNT(*) FROM query_logs WHERE generated_sql LIKE '-- ERROR%'"
            )
            error_count = (await cursor.fetchone())[0]

            error_rate = round((error_count / total) * 100, 2) if total > 0 else 0

            # ── SUGGESTION STATS ─────────────────────────────────────
            # Only available if suggestion_log table exists
            try:
                cursor = await con.execute(
                    "SELECT COUNT(*), SUM(accepted) FROM suggestion_log"
                )
                suggestion_rows = await cursor.fetchone()
                total_suggestions = suggestion_rows[0] or 0
                accepted_suggestions = suggestion_rows[1] or 0
                suggestion_acceptance_rate = (
                    round((accepted_suggestions / total_suggestions) * 100, 2)
                    if total_suggestions > 0 else 0
                )
            except Exception:
                total_suggestions = 0
                suggestion_acceptance_rate = 0


        # ── VAULT STATS ──────────────────────────────────────────
        async with aiosqlite.connect(db_path("mem_palace_vault.db")) as con2:
            cursor = await con2.execute("SELECT COUNT(*) FROM vault")
            vault_size  = (await cursor.fetchone())[0]
            cursor = await con2.execute("SELECT SUM(hit_count) FROM vault")
            total_hits  = (await cursor.fetchone())[0] or 0


        result = {
            # Volume
            "total_queries":            total,
            "cache_hits":               cache_hits,
            "cache_hit_rate_pct":       cache_hit_rate,

            # Latency
            "avg_latency_ms":           avg_latency,
            "avg_ai_latency_ms":        avg_ai_latency,
            "p90_latency_ms":           p90_latency,

            # Cost
            "total_ai_spend_usd":       total_spend,
            "avg_cost_per_query_usd":   avg_cost,
            "cache_savings_usd":        cache_savings,

            # Quality
            "error_rate_pct":           error_rate,

            # Vault
            "vault_size":               vault_size,
            "total_vault_hits":         total_hits,

            # Suggestions
            "total_suggestions":            total_suggestions,
            "suggestion_acceptance_rate_pct": suggestion_acceptance_rate,
        }

    except Exception as e:
        logger.error(f"Metrics computation failed: {str(e)}")
        return {"error": str(e)}

    return result

@app.get("/pipeline-status")
async def get_pipeline_status():
    return await sqlite_rows_to_dicts(
        db_path("omni_master_catalog.db"),
        "SELECT * FROM pipeline_runs ORDER BY run_id DESC LIMIT 10"
    )

@app.get("/scheduler-status")
async def get_scheduler_status():
    return await sqlite_rows_to_dicts(
        db_path("omni_audit.db"),
        "SELECT timestamp, generated_sql, latency_ms FROM query_logs WHERE is_automated = 1 ORDER BY log_id DESC LIMIT 10"
    )

@app.get("/schema")
async def get_schema():
    """High-level overview: all sources, tables, and row counts."""
    return await sqlite_rows_to_dicts(
        db_path("omni_master_catalog.db"),
        """SELECT
               source_name,
               source_type,
               table_name,
               row_count,
               table_description
           FROM schema_registry
           ORDER BY source_name, table_name"""
    )


@app.get("/schema/{source_name}/{table_name}")
async def get_table_schema(source_name: str, table_name: str):
    """
    Full column profile for one specific table.
    Returns column names, types, sample values, distinct counts, null counts.
    """
    import json

    columns = await sqlite_rows_to_dicts(
        db_path("omni_master_catalog.db"),
        """SELECT
               column_name,
               column_type,
               is_primary_key,
               is_nullable,
               sample_values,
               distinct_count,
               null_count
           FROM column_registry
           WHERE source_name = ? AND table_name = ?
           ORDER BY rowid""",
        (source_name, table_name)
    )

    # Parse sample_values JSON string back to list
    for col in columns:
        try:
            col["sample_values"] = json.loads(col["sample_values"] or "[]")
        except Exception:
            col["sample_values"] = []

    table_info = await sqlite_rows_to_dicts(
        db_path("omni_master_catalog.db"),
        """SELECT source_name, source_type, source_path, row_count,
                  table_description, schema_notes
           FROM schema_registry
           WHERE source_name = ? AND table_name = ?""",
        (source_name, table_name)
    )

    return {
        "table": table_info[0] if table_info else {},
        "columns": columns,
    }


@app.get("/sources")
async def get_sources():
    """Lists all registered data sources with their table counts."""
    return await sqlite_rows_to_dicts(
        db_path("omni_master_catalog.db"),
        """SELECT
               source_name,
               source_type,
               source_path,
               COUNT(DISTINCT table_name) as table_count,
               SUM(row_count) as total_rows
           FROM schema_registry
           GROUP BY source_name, source_type, source_path
           ORDER BY source_name"""
    )


@app.get("/sources/{source_name}")
async def get_source_detail(source_name: str):
    """All tables in one specific source."""
    return await sqlite_rows_to_dicts(
        db_path("omni_master_catalog.db"),
        """SELECT table_name, row_count, table_description
           FROM schema_registry
           WHERE source_name = ?
           ORDER BY table_name""",
        (source_name,)
    )

if __name__ == "__main__":
    import uvicorn
    cfg = load_config("config.yaml")
    uvicorn.run("main:app", host=cfg.api.host, port=cfg.api.port, reload=True)