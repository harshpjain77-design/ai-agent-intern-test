from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, ConfigDict, Field


class DecisionState(str, Enum):
    ANSWER = "ANSWER"
    CLARIFY = "CLARIFY"
    ABSTAIN = "ABSTAIN"
    HANDOFF = "HANDOFF"
    REFUSE = "REFUSE"


class Citation(BaseModel):
    filename: str
    heading: str


class PublicItem(BaseModel):
    name: str
    quantity: int
    final_sale: bool


class PublicOrderStatus(BaseModel):
    """The sole order schema that can cross the tool boundary."""
    model_config = ConfigDict(extra="forbid")
    order_id: str
    membership_tier: str
    items: list[PublicItem]
    placed_at: str
    status: str
    status_updated_at: str
    shipped_at: str | None = None
    delivered_at: str | None = None
    carrier: str | None = None
    tracking_number: str | None = None
    estimated_delivery: str | None = None
    customer_safe_message: str


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, str]


class ScoredChunk(BaseModel):
    filename: str
    heading: str
    bm25: float
    cosine: float
    authority_score: float
    total_score: float
    eligible: bool
    exclusion_reason: str | None = None


class EvidencePack(BaseModel):
    query: str
    rewritten_query: str | None = None
    candidates: list[ScoredChunk] = Field(default_factory=list)
    selected_citations: list[Citation] = Field(default_factory=list)
    excluded_sources: list[dict[str, str]] = Field(default_factory=list)
    conflict_detected: bool = False
    conflict_sources: list[str] = Field(default_factory=list)
    conflict_reason: str | None = None
    answerable: bool = True
    insufficiency_reason: str | None = None


class AgentResponse(BaseModel):
    state: DecisionState
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    handoff_recommended: bool = False
    tool_calls: list[ToolCall] = Field(default_factory=list)
    trace_id: str

