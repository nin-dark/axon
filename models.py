from pydantic import BaseModel
from typing import Optional, List, Any, Dict


class QueryRequest(BaseModel):
    prompt: str


class ApprovalRequest(BaseModel):
    approved: bool


class AcceptSuggestionRequest(BaseModel):
    """Sent when operator accepts a TF-IDF suggestion."""
    use_suggestion: bool
    suggestion_sql: Optional[str] = None  # if use_suggestion=True
    intent_key: str


class SingleDBResult(BaseModel):
    database: str
    status: str
    data: Optional[List[Any]] = None
    msg: Optional[str] = None


class PendingApprovalResponse(BaseModel):
    """Returned from POST /query when SQL is ready for approval."""
    status: str                          # "pending_approval" | "suggestion_available" | "rejected" | "error"
    approval_id: Optional[str] = None
    sql: Optional[str] = None
    from_cache: bool = False
    query_cost_usd: Optional[float] = None
    ai_latency_ms: Optional[int] = None
    reason: Optional[str] = None
    suggestion: Optional[Dict] = None   # populated when status == "suggestion_available"
    intent_key: Optional[str] = None


class QueryResponse(BaseModel):
    """Returned from POST /approve/{id} after execution."""
    status: str
    sql: Optional[str] = None
    from_cache: bool = False
    latency_ms: float = 0.0
    ai_latency_ms: Optional[float] = None
    execution_latency_ms: Optional[float] = None
    agent1_tokens: Optional[int] = None
    agent2_tokens: Optional[int] = None
    query_cost_usd: Optional[float] = None
    reason: Optional[str] = None
    results: Optional[List[SingleDBResult]] = None
    insight_report: Optional[str] = None
    ml_analysis: Optional[dict] = None