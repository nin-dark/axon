import pytest
from engine import AxonEngine
from config_loader import load_config
config = load_config("config.yaml")



@pytest.fixture
async def ready_engine():
    engine = AxonEngine(config)
    await engine.metadata()
    return engine


@pytest.mark.asyncio
async def test_full_pipeline_success(ready_engine, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "Y")

    result = await ready_engine.process_query("count all failed transactions")

    assert result["status"] == "success"
    assert result["sql"] is not None
    assert result["sql"].strip().upper().startswith("SELECT")
    assert isinstance(result["results"], list)
    assert len(result["results"]) == 180
    assert result["latency_ms"] > 0
    

@pytest.mark.asyncio
async def test_full_pipeline_rejection(ready_engine, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "N")

    result = await ready_engine.process_query("count all failed transactions")

    assert result["status"] == "rejected"
    assert "results" not in result


@pytest.mark.asyncio
async def test_cache_hit_on_second_query(ready_engine, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "Y")

    result1 = await ready_engine.process_query("count all failed transactions")
    assert result1["from_cache"] == False

    result2 = await ready_engine.process_query("count all failed transactions")
    assert result2["from_cache"] == True

    assert result1["sql"] == result2["sql"]


@pytest.mark.asyncio
async def test_validation_blocks_before_ai(ready_engine):
    result = await ready_engine.process_query("")
    assert result["status"] == "rejected"
    assert result["sql"] is None
    pass


@pytest.mark.asyncio
async def test_raw_sql_bypasses_ai(ready_engine, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "Y")
    raw = "SELECT COUNT(*) FROM transactions WHERE status = 'Failed'"
    result = await ready_engine.process_query(raw)
    assert result["status"] == "success"
    assert result["sql"] == raw