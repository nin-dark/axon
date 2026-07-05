import yaml
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LLMConfig:
    model: str
    router_model: str
    sql_model: str
    input_cost_per_million_tokens: float
    output_cost_per_million_tokens: float
    ai_latency_threshold_ms: int


@dataclass
class CacheConfig:
    embedding_model: str
    embedding_dim: int
    semantic_similarity_threshold: float
    tfidf_threshold: float


@dataclass
class EngineConfig:
    max_rows_per_query: int
    max_concurrent_connections: int
    max_prompt_length: int
    approval_timeout_seconds: int
    sql_injection_signals: List[str]


@dataclass
class MonitoringConfig:
    drift_window_size: int
    health_check_interval_hours: int
    alert_webhook_url: str
    health_check_queries: List[str]



@dataclass
class PrivacyConfig:
    disable_sample_values: bool
    masked_columns: List[str]
    user_contexts: dict


@dataclass
class APIConfig:
    host: str
    port: int
    cors_origins: List[str]


@dataclass
class DataSource:
    name: str
    type: str                    # sqlite | sqlite_folder | csv | rest
    path: Optional[str] = None   # for sqlite, sqlite_folder, csv
    url:  Optional[str] = None   # for rest
    auth_token: Optional[str] = None
    schema_notes: str = ""
    auto_schema_notes: bool = True  # auto-generate schema_notes from discovered data


@dataclass
class Config:
    llm: LLMConfig
    cache: CacheConfig
    engine: EngineConfig
    monitoring: MonitoringConfig
    api: APIConfig
    privacy: PrivacyConfig
    data_sources: List[DataSource]


def load_config(path: str = "config.yaml") -> Config:
    """
    Loads config.yaml and returns a typed Config object.
    Environment variables override YAML values where relevant.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"config.yaml not found at '{path}'. "
            f"Copy config.yaml to your working directory and fill in your data sources."
        )

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    llm_raw = raw.get("llm", {})
    cache_raw = raw.get("cache", {})
    engine_raw = raw.get("engine", {})
    monitoring_raw = raw.get("monitoring", {})
    api_raw = raw.get("api", {})
    privacy_raw = raw.get("privacy", {})
    sources_raw = raw.get("data_sources", [])

    # Environment variables override YAML for sensitive values
    alert_webhook = (
        os.environ.get("ALERT_WEBHOOK_URL")
        or monitoring_raw.get("alert_webhook_url", "")
    )

    llm = LLMConfig(
        model=llm_raw.get("model", "gemini-2.5-flash-lite"),
        router_model=llm_raw.get("router_model", "gemini-2.5-flash"),
        sql_model=llm_raw.get("sql_model", "gemini-2.5-pro"),
        input_cost_per_million_tokens=llm_raw.get("input_cost_per_million_tokens", 0.10),
        output_cost_per_million_tokens=llm_raw.get("output_cost_per_million_tokens", 0.40),
        ai_latency_threshold_ms=llm_raw.get("ai_latency_threshold_ms", 3000),
    )

    cache = CacheConfig(
        embedding_model=cache_raw.get("embedding_model", "all-MiniLM-L6-v2"),
        embedding_dim=cache_raw.get("embedding_dim", 384),
        semantic_similarity_threshold=cache_raw.get("semantic_similarity_threshold", 0.85),
        tfidf_threshold=cache_raw.get("tfidf_threshold", 0.4),
    )

    engine_cfg = EngineConfig(
        max_rows_per_query=engine_raw.get("max_rows_per_query", 5000),
        max_concurrent_connections=engine_raw.get("max_concurrent_connections", 50),
        max_prompt_length=engine_raw.get("max_prompt_length", 500),
        approval_timeout_seconds=engine_raw.get("approval_timeout_seconds", 300),
        sql_injection_signals=engine_raw.get("sql_injection_signals", [
            "DROP TABLE", "DELETE FROM", "--", "/*", "*/", "UNION SELECT"
        ]),
    )

    monitoring = MonitoringConfig(
        drift_window_size=monitoring_raw.get("drift_window_size", 100),
        health_check_interval_hours=monitoring_raw.get("health_check_interval_hours", 1),
        alert_webhook_url=alert_webhook,
        health_check_queries=monitoring_raw.get("health_check_queries", []),
    )

    api = APIConfig(
        host=api_raw.get("host", "0.0.0.0"),
        port=api_raw.get("port", 8000),
        cors_origins=api_raw.get("cors_origins", ["*"]),
    )

    privacy = PrivacyConfig(
        disable_sample_values=privacy_raw.get("disable_sample_values", True),
        masked_columns=privacy_raw.get("masked_columns", []),
        user_contexts=privacy_raw.get("user_contexts", {}),
    )

    data_sources = []
    for s in sources_raw:
        src_path = s.get("path")
        src_url = s.get("url")
        src_type = s.get("type")
        src_name = s.get("name")

        # ── AUTO-DETECT TYPE ──────────────────────────────────────
        if not src_type:
            if src_url:
                src_type = "rest"
            elif src_path:
                if os.path.isdir(src_path):
                    src_type = "sqlite_folder"
                elif src_path.lower().endswith(".csv"):
                    src_type = "csv"
                elif src_path.lower().endswith(".db"):
                    src_type = "sqlite"
                else:
                    # Default to sqlite for any other file
                    src_type = "sqlite"
            else:
                src_type = "sqlite"

        # ── AUTO-DERIVE NAME ──────────────────────────────────────
        if not src_name:
            if src_path:
                # Use basename without extension, e.g. "sales.db" → "sales"
                basename = os.path.basename(src_path.rstrip("/\\"))
                src_name = os.path.splitext(basename)[0] if "." in basename else basename
            elif src_url:
                src_name = src_url.split("//")[-1].split("/")[0]  # domain as name
            else:
                src_name = "unnamed"

        data_sources.append(DataSource(
            name=src_name,
            type=src_type,
            path=src_path,
            url=src_url,
            auth_token=s.get("auth_token"),
            schema_notes=s.get("schema_notes", ""),
            auto_schema_notes=s.get("auto_schema_notes", True),
        ))

    if not data_sources:
        raise ValueError(
            "No data_sources defined in config.yaml. "
            "Add at least one source under the data_sources key."
        )

    return Config(
        llm=llm,
        cache=cache,
        engine=engine_cfg,
        monitoring=monitoring,
        api=api,
        privacy=privacy,
        data_sources=data_sources,
    )