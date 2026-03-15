from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .schemas import (
    CreateTicketRequest,
    EscalationLevel,
    RiskLevel,
    TicketCategory,
    TicketDraft,
    TicketListResponse,
    TicketPriority,
    TicketRecord,
)


class TicketStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if not self.path.exists():
            self._write_items([])

    def _read_items(self) -> list[dict]:
        try:
            with self.path.open("r", encoding="utf-8") as file:
                data = json.load(file)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _write_items(self, items: list[dict]) -> None:
        tmp_path = self.path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as file:
            json.dump(items, file, indent=2)
        tmp_path.replace(self.path)

    def create_ticket(self, payload: CreateTicketRequest) -> TicketRecord:
        with self.lock:
            items = self._read_items()

            ticket = TicketRecord(
                id=f"TKT-{uuid4().hex[:8].upper()}",
                created_at=datetime.now(UTC).isoformat(),
                transcript=payload.transcript,
                draft=payload.draft,
            )

            items.insert(0, ticket.model_dump())
            self._write_items(items)
            return ticket

    def list_tickets(self, limit: int = 20) -> TicketListResponse:
        with self.lock:
            items = self._read_items()[: max(1, min(limit, 100))]
            parsed = [TicketRecord.model_validate(self._normalize_legacy_record(item)) for item in items]
            return TicketListResponse(items=parsed)

    def _normalize_legacy_record(self, item: dict) -> dict:
        draft = item.get("draft", {})
        if not isinstance(draft, dict):
            draft = {}

        category_value = draft.get("category", TicketCategory.other.value)
        try:
            category = TicketCategory(category_value)
        except ValueError:
            category = TicketCategory.other

        priority_value = draft.get("priority", TicketPriority.medium.value)
        try:
            priority = TicketPriority(priority_value)
        except ValueError:
            priority = TicketPriority.medium

        summary = str(draft.get("summary", item.get("transcript", "Customer issue"))).strip() or "Customer issue"
        route_to = draft.get("route_to") or {
            TicketCategory.billing: "Billing Operations",
            TicketCategory.delivery: "Logistics Desk",
            TicketCategory.account: "Identity Support",
            TicketCategory.technical: "Technical Support",
            TicketCategory.refund: "Refund Desk",
            TicketCategory.other: "Customer Care",
        }[category]

        normalized_draft = TicketDraft(
            title=str(draft.get("title", "Support ticket")).strip() or "Support ticket",
            summary=summary,
            category=category,
            priority=priority,
            sentiment=str(draft.get("sentiment", "neutral")).strip() or "neutral",
            route_to=str(route_to),
            escalation_level=EscalationLevel(draft.get("escalation_level", EscalationLevel.none.value)),
            sla_risk=RiskLevel(draft.get("sla_risk", RiskLevel.medium.value if priority in {TicketPriority.medium, TicketPriority.high} else RiskLevel.low.value)),
            churn_risk=RiskLevel(draft.get("churn_risk", RiskLevel.low.value)),
            requires_manager_review=bool(draft.get("requires_manager_review", priority == TicketPriority.critical)),
            refund_requested=bool(draft.get("refund_requested", category == TicketCategory.refund)),
            customer_impact=str(draft.get("customer_impact", "Customer requires follow-up from support.")).strip()
            or "Customer requires follow-up from support.",
            resolution_path=str(draft.get("resolution_path", f"Route case to {route_to} and validate missing details.")).strip()
            or f"Route case to {route_to} and validate missing details.",
            next_actions=[
                str(action).strip()
                for action in draft.get(
                    "next_actions",
                    [f"Send ticket to {route_to}", "Validate customer details and continue support workflow."],
                )
                if str(action).strip()
            ][:6],
            suggested_reply=str(
                draft.get(
                    "suggested_reply",
                    "We have logged your issue and our support team is reviewing it now.",
                )
            ).strip()
            or "We have logged your issue and our support team is reviewing it now.",
        )

        normalized = dict(item)
        normalized["draft"] = normalized_draft.model_dump()
        return normalized
