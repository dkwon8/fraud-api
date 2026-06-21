from uuid import UUID
from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


class ScoreResponse(BaseModel):
    transaction_id: UUID
    if_score: float
    ae_score: float
    final_score: float
    predicted_label: int
    threshold_used: float
    latency_ms: float


class LabelRequest(BaseModel):
    transaction_id: UUID
    true_label: int = Field(ge=0, le=1)


class LabelResponse(BaseModel):
    transaction_id: UUID
    true_label: int
    message: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float
