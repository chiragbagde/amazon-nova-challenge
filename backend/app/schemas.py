from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class TicketCategory(str, Enum):
    billing = "billing"
    delivery = "delivery"
    technical = "technical"
    account = "account"
    refund = "refund"
    other = "other"


class TicketPriority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class EscalationLevel(str, Enum):
    none = "none"
    supervisor = "supervisor"
    specialist = "specialist"
    incident = "incident"


class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class CustomerInfo(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None


class TranscriptAnalyzeRequest(BaseModel):
    transcript: str = Field(min_length=10, max_length=8000)
    customer: CustomerInfo | None = None
    language: str = Field(default="en", max_length=16)


class IntakeAnswer(BaseModel):
    question_key: str = Field(min_length=2, max_length=64)
    question_label: str = Field(min_length=2, max_length=120)
    answer: str = Field(min_length=1, max_length=1000)


class IntakeQuestion(BaseModel):
    key: str = Field(min_length=2, max_length=64)
    label: str = Field(min_length=2, max_length=120)
    prompt: str = Field(min_length=5, max_length=300)
    help_text: str = Field(min_length=5, max_length=240)


class IntakeNextRequest(BaseModel):
    history: list[IntakeAnswer] = Field(default_factory=list, max_length=12)
    question_key: str | None = Field(default=None, max_length=64)
    answer: str | None = Field(default=None, max_length=1000)
    customer: CustomerInfo | None = None
    language: str = Field(default="en", max_length=16)


class IntakeNextResponse(BaseModel):
    history: list[IntakeAnswer]
    next_question: IntakeQuestion | None = None
    is_complete: bool
    compiled_transcript: str | None = None
    context_summary: str = Field(min_length=5, max_length=1200)
    missing_fields: list[str] = Field(default_factory=list, max_length=8)


class TicketDraft(BaseModel):
    title: str = Field(min_length=5, max_length=120)
    summary: str = Field(min_length=10, max_length=2000)
    category: TicketCategory
    priority: TicketPriority
    sentiment: str = Field(min_length=3, max_length=32)
    route_to: str = Field(min_length=3, max_length=64)
    escalation_level: EscalationLevel
    sla_risk: RiskLevel
    churn_risk: RiskLevel
    requires_manager_review: bool
    refund_requested: bool = False
    customer_impact: str = Field(min_length=5, max_length=500)
    resolution_path: str = Field(min_length=5, max_length=280)
    next_actions: list[str] = Field(default_factory=list, max_length=6)
    suggested_reply: str = Field(min_length=5, max_length=1200)


class AnalyzeResponse(BaseModel):
    draft: TicketDraft
    confidence: float = Field(ge=0.0, le=1.0)
    source: Literal["nova", "fallback"]
    warnings: list[str] = Field(default_factory=list)


class CreateTicketRequest(BaseModel):
    transcript: str = Field(min_length=10, max_length=8000)
    draft: TicketDraft


class TicketRecord(BaseModel):
    id: str
    created_at: str
    transcript: str
    draft: TicketDraft


class TicketListResponse(BaseModel):
    items: list[TicketRecord]


# ── Nova Sonic intake ─────────────────────────────────────────────────────────

class SonicIntakeRequest(BaseModel):
    """Six agent answers for the Sonic-led intake interview."""
    answers: list[str] = Field(min_length=6, max_length=6,
                               description="One answer per intake question, in order.")
    language: str = Field(default="en", max_length=16)


class SonicAnswerItem(BaseModel):
    key: str
    label: str
    answer: str
    nova_ack: str = ""


class SonicIntakeResponse(BaseModel):
    answers: list[SonicAnswerItem]
    transcript: str          # compiled case brief for Nova Lite
    full_text: str            # raw Sonic text output (acks concatenated)
    success: bool
    errors: list[str] = Field(default_factory=list)
