import pytest
import pytest_asyncio
from engine import AxonEngine
from config_loader import load_config
config = load_config("config.yaml")


@pytest.fixture
def engine():
    return AxonEngine(config)


# ── CONSCIOUS GATE ────────────────────────────────────────────────────────────

def test_gate_approves_on_Y(engine, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "Y")
    result = engine._conscious_gate("SELECT 1")
    assert result == True


def test_gate_rejects_on_N(engine, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "N")
    result = engine._conscious_gate("SELECT 1")
    assert result == False

def test_gate_rejects_on_lowercase_y(engine, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "y")
    result = engine._conscious_gate("SELECT 1")
    assert result == False


# ── INPUT VALIDATION ──────────────────────────────────────────────────────────

def test_validation_rejects_empty_string(engine):
    valid, msg = engine._validate_input("")
    assert valid == False
    assert msg != ""


def test_validation_rejects_whitespace_only(engine):
    valid, msg = engine._validate_input("   ")
    assert valid == False


def test_validation_rejects_oversized_prompt(engine):

    long_prompt = "a" * 501
    valid, msg = engine._validate_input(long_prompt)
    assert valid == False

def test_validation_rejects_sql_injection(engine):
    valid, msg = engine._validate_input("DROP TABLE transactions")
    assert valid == False

def test_validation_rejects_comment_injection(engine):
    valid, msg = engine._validate_input("show me data -- ignore above")
    assert valid == False

def test_validation_accepts_clean_prompt(engine):
    valid, msg = engine._validate_input("show me total failed transactions")
    assert valid == True
    assert msg == ""


# ── MEM PALACE ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_mem_palace_returns_none_on_miss(engine):
    result = await engine._check_mem_palace("a key that has never been stored")
    assert result is None

@pytest.mark.asyncio
async def test_mem_palace_stores_and_retrieves(engine):
    await engine._store_in_mem_palace("test intent", "SELECT 1")
    result = await engine._check_mem_palace("test intent")
    assert result == "SELECT 1"

@pytest.mark.asyncio
async def test_mem_palace_increments_hit_count(engine):
    await engine._store_in_mem_palace("hit count intent", "SELECT 2")
    await engine._check_mem_palace("hit count intent")
    await engine._check_mem_palace("hit count intent")

    import sqlite3
    con = sqlite3.connect("mem_palace_vault.db")
    row = con.execute("SELECT hit_count FROM vault WHERE intent_key = ?", ("hit count intent",)).fetchone()
    con.close()
    assert row[0] == 2

@pytest.mark.asyncio
async def test_mem_palace_insert_or_replace(engine):
    await engine._store_in_mem_palace("duplicate key", "SELECT 1")
    await engine._store_in_mem_palace("duplicate key", "SELECT 99")
    result = await engine._check_mem_palace("duplicate key")
    assert result == "SELECT 99"


# ── SPINAL REFLEX ─────────────────────────────────────────────────────────────

def test_spinal_reflex_detects_select(engine):
    prompt = "SELECT * FROM transactions"
    assert prompt.strip().upper().startswith(("SELECT", "INSERT", "UPDATE", "DELETE"))

def test_spinal_reflex_ignores_english(engine):
    prompt = "show me failed transactions"
    assert not prompt.strip().upper().startswith(("SELECT", "INSERT", "UPDATE", "DELETE"))