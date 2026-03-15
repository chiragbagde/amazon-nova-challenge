from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..schemas import (
    AnalyzeResponse,
    EscalationLevel,
    RiskLevel,
    TicketCategory,
    TicketDraft,
    TicketPriority,
    TranscriptAnalyzeRequest,
)
from .bedrock_lite_client import BedrockLiteClient


@dataclass
class TicketAnalyzer:
    llm: BedrockLiteClient

    def analyze(self, payload: TranscriptAnalyzeRequest) -> AnalyzeResponse:
        if self.llm.is_configured():
            try:
                return self._analyze_with_nova(payload)
            except Exception as exc:
                fallback = self._analyze_with_rules(payload)
                fallback.warnings.append(f"Nova failed, used fallback: {exc}")
                return fallback

        fallback = self._analyze_with_rules(payload)
        fallback.warnings.append("AWS_BEDROCK_KEY missing, used fallback analyzer")
        return fallback

    def _analyze_with_nova(self, payload: TranscriptAnalyzeRequest) -> AnalyzeResponse:
        prompt = self._build_prompt(payload)
        raw = self.llm.converse(prompt)
        parsed = self._extract_json(raw)

        draft = TicketDraft.model_validate(parsed)
        confidence = float(parsed.get("confidence", 0.83))

        return AnalyzeResponse(
            draft=draft,
            confidence=max(0.0, min(confidence, 1.0)),
            source="nova",
            warnings=[],
        )

    def _analyze_with_rules(self, payload: TranscriptAnalyzeRequest) -> AnalyzeResponse:
        text = payload.transcript.strip()
        lower = text.lower()

        category = TicketCategory.other
        if any(word in lower for word in ["refund", "charge", "payment", "invoice"]):
            category = TicketCategory.refund if "refund" in lower else TicketCategory.billing
        elif any(word in lower for word in ["late", "delivery", "courier", "shipment", "package"]):
            category = TicketCategory.delivery
        elif any(word in lower for word in ["password", "login", "account", "locked"]):
            category = TicketCategory.account
        elif any(word in lower for word in ["error", "bug", "crash", "app", "not working"]):
            category = TicketCategory.technical

        priority = TicketPriority.medium
        if any(word in lower for word in ["urgent", "immediately", "cannot", "can't", "blocked", "asap"]):
            priority = TicketPriority.high
        if any(word in lower for word in ["legal", "outage", "payment failed repeatedly", "critical"]):
            priority = TicketPriority.critical

        sentiment = "neutral"
        if any(word in lower for word in ["angry", "frustrated", "upset", "terrible", "worst"]):
            sentiment = "negative"
        elif any(word in lower for word in ["thanks", "thank you", "great", "happy"]):
            sentiment = "positive"

        refund_requested = any(word in lower for word in ["refund", "charged twice", "double charge", "duplicate charge"])
        requires_manager_review = priority == TicketPriority.critical or any(
            word in lower for word in ["supervisor", "manager", "complaint", "cancel", "legal"]
        )

        route_to = {
            TicketCategory.billing: "Billing Operations",
            TicketCategory.delivery: "Logistics Desk",
            TicketCategory.account: "Identity Support",
            TicketCategory.technical: "Technical Support",
            TicketCategory.refund: "Refund Desk",
            TicketCategory.other: "Customer Care",
        }[category]

        escalation_level = EscalationLevel.none
        if priority == TicketPriority.high or requires_manager_review:
            escalation_level = EscalationLevel.supervisor
        if category in {TicketCategory.technical, TicketCategory.account} and priority in {
            TicketPriority.high,
            TicketPriority.critical,
        }:
            escalation_level = EscalationLevel.specialist
        if priority == TicketPriority.critical:
            escalation_level = EscalationLevel.incident

        sla_risk = RiskLevel.medium if priority in {TicketPriority.medium, TicketPriority.high} else RiskLevel.low
        if priority == TicketPriority.critical or any(word in lower for word in ["today", "1 hour", "immediately"]):
            sla_risk = RiskLevel.high

        churn_risk = RiskLevel.low
        if sentiment == "negative" or any(word in lower for word in ["cancel", "switch", "never again", "bad service"]):
            churn_risk = RiskLevel.medium
        if any(word in lower for word in ["cancel account", "close account", "lawsuit", "fraud"]):
            churn_risk = RiskLevel.high

        title_prefix = {
            TicketCategory.billing: "Billing issue",
            TicketCategory.delivery: "Delivery issue",
            TicketCategory.account: "Account access issue",
            TicketCategory.technical: "Technical issue",
            TicketCategory.refund: "Refund request",
            TicketCategory.other: "Customer support issue",
        }[category]

        summary = text[:500]
        if len(text) > 500:
            summary += "..."

        draft = TicketDraft(
            title=f"{title_prefix}: {summary[:64]}",
            summary=summary,
            category=category,
            priority=priority,
            sentiment=sentiment,
            route_to=route_to,
            escalation_level=escalation_level,
            sla_risk=sla_risk,
            churn_risk=churn_risk,
            requires_manager_review=requires_manager_review,
            refund_requested=refund_requested,
            customer_impact="Customer is blocked and requires support follow-up.",
            resolution_path=self._resolution_path(category, escalation_level, refund_requested),
            next_actions=[
                f"Send ticket to {route_to}",
                "Validate account or order details with the customer",
                "Resolve immediately or escalate based on risk flags",
            ],
            suggested_reply="Thanks for sharing the details. We have created your support ticket and our team is working on it.",
        )

        return AnalyzeResponse(
            draft=draft,
            confidence=0.62,
            source="fallback",
            warnings=[],
        )

    def _build_prompt(self, payload: TranscriptAnalyzeRequest) -> str:
        customer = payload.customer.model_dump() if payload.customer else {}
        transcript = payload.transcript.strip()

        return (
            "You are a support operations assistant. Convert the customer transcript into a ticket draft. "
            "Return strict JSON only, no markdown and no extra text.\n"
            "Required JSON shape:\n"
            "{\n"
            "  \"title\": \"string\",\n"
            "  \"summary\": \"string\",\n"
            "  \"category\": \"billing|delivery|technical|account|refund|other\",\n"
            "  \"priority\": \"low|medium|high|critical\",\n"
            "  \"sentiment\": \"positive|neutral|negative\",\n"
            "  \"route_to\": \"string\",\n"
            "  \"escalation_level\": \"none|supervisor|specialist|incident\",\n"
            "  \"sla_risk\": \"low|medium|high\",\n"
            "  \"churn_risk\": \"low|medium|high\",\n"
            "  \"requires_manager_review\": true,\n"
            "  \"refund_requested\": false,\n"
            "  \"customer_impact\": \"string\",\n"
            "  \"resolution_path\": \"string\",\n"
            "  \"next_actions\": [\"string\", \"string\"],\n"
            "  \"suggested_reply\": \"string\",\n"
            "  \"confidence\": 0.0\n"
            "}\n"
            "Think like a support triage lead. Optimize for route accuracy, escalation judgment, SLA risk, and retention risk.\n"
            f"Language: {payload.language}\n"
            f"Customer context: {json.dumps(customer)}\n"
            f"Transcript: {json.dumps(transcript)}\n"
        )

    def _resolution_path(
        self,
        category: TicketCategory,
        escalation_level: EscalationLevel,
        refund_requested: bool,
    ) -> str:
        if escalation_level == EscalationLevel.incident:
            return "Open incident bridge, assign owner, and update customer every 30 minutes."
        if category == TicketCategory.refund or refund_requested:
            return "Verify payment records, confirm duplicate charge, and queue refund approval."
        if category == TicketCategory.delivery:
            return "Check courier status, confirm ETA, and offer compensation if SLA is breached."
        if category == TicketCategory.account:
            return "Validate identity, restore account access, and reset blocked credentials."
        if category == TicketCategory.technical:
            return "Reproduce issue, attach logs, and route to product support if unresolved."
        return "Acknowledge issue, gather missing context, and route to the correct desk."

    def _extract_json(self, text: str) -> dict:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Model output did not contain JSON object")

        raw_json = cleaned[start : end + 1]
        return json.loads(raw_json)
