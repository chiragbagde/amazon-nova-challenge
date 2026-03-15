from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..schemas import CustomerInfo, IntakeAnswer, IntakeNextRequest, IntakeNextResponse, IntakeQuestion
from .bedrock_lite_client import BedrockLiteClient


@dataclass(frozen=True)
class IntakeStep:
    key: str
    label: str
    prompt: str
    help_text: str


STEPS: tuple[IntakeStep, ...] = (
    IntakeStep(
        key="issue_summary",
        label="Issue Summary",
        prompt="What exactly is the customer reporting?",
        help_text="Capture the main issue in one or two sentences.",
    ),
    IntakeStep(
        key="reference_details",
        label="Reference Details",
        prompt="Which order, account, product, or transaction is affected?",
        help_text="Ask for any order ID, account email, product name, or transaction detail.",
    ),
    IntakeStep(
        key="timeline",
        label="Timeline",
        prompt="When did this issue start, and what has already happened so far?",
        help_text="Include dates, repeated attempts, or prior support interactions.",
    ),
    IntakeStep(
        key="customer_impact",
        label="Customer Impact",
        prompt="How is this affecting the customer right now?",
        help_text="Capture business impact, emotional impact, or blocked work.",
    ),
    IntakeStep(
        key="urgency",
        label="Urgency",
        prompt="Is there a deadline, urgency, escalation threat, or risk if this is not resolved soon?",
        help_text="Look for time pressure, cancellation risk, or manager escalation.",
    ),
    IntakeStep(
        key="desired_resolution",
        label="Desired Resolution",
        prompt="What outcome does the customer want from support?",
        help_text="Refund, replacement, callback, account recovery, technical fix, or status update.",
    ),
)


@dataclass
class IntakeService:
    llm: BedrockLiteClient | None = None

    def next_step(self, payload: IntakeNextRequest) -> IntakeNextResponse:
        history = list(payload.history)
        history = self._merge_latest_answer(history, payload.question_key, payload.answer)

        answered_keys = {item.question_key for item in history}
        next_step = next((step for step in STEPS if step.key not in answered_keys), None)
        compiled_transcript = self._build_compiled_transcript(history, payload.customer) if next_step is None else None

        return IntakeNextResponse(
            history=history,
            next_question=self._build_next_question(next_step, history, payload.language) if next_step else None,
            is_complete=next_step is None,
            compiled_transcript=compiled_transcript,
            context_summary=self._build_context_summary(history, payload.customer),
            missing_fields=[step.label for step in STEPS if step.key not in answered_keys],
        )

    def _merge_latest_answer(
        self,
        history: list[IntakeAnswer],
        question_key: str | None,
        answer: str | None,
    ) -> list[IntakeAnswer]:
        if not question_key or not answer or not answer.strip():
            return history

        step = next((item for item in STEPS if item.key == question_key), None)
        if step is None:
            return history

        normalized = answer.strip()
        updated = [item for item in history if item.question_key != question_key]
        updated.append(
            IntakeAnswer(
                question_key=step.key,
                question_label=step.label,
                answer=normalized,
            )
        )
        return updated

    def _to_question(self, step: IntakeStep) -> IntakeQuestion:
        return IntakeQuestion(
            key=step.key,
            label=step.label,
            prompt=step.prompt,
            help_text=step.help_text,
        )

    def _build_next_question(
        self,
        step: IntakeStep,
        history: list[IntakeAnswer],
        language: str,
    ) -> IntakeQuestion:
        if self.llm and self.llm.is_configured():
            try:
                return self._build_question_with_nova(step, history, language)
            except Exception:
                pass
        return self._to_question(step)

    def _build_context_summary(self, history: list[IntakeAnswer], customer: CustomerInfo | None) -> str:
        if not history:
            return "No intake answers captured yet."

        parts: list[str] = []
        if customer and any([customer.name, customer.email, customer.phone]):
            customer_bits = [value for value in [customer.name, customer.email, customer.phone] if value]
            parts.append(f"Customer: {', '.join(customer_bits)}.")

        for item in history:
            parts.append(f"{item.question_label}: {item.answer}")

        summary = " ".join(parts).strip()
        return summary[:1200]

    def _build_compiled_transcript(self, history: list[IntakeAnswer], customer: CustomerInfo | None) -> str:
        by_key = {item.question_key: item.answer for item in history}
        customer_line = ""
        if customer and any([customer.name, customer.email, customer.phone]):
            customer_bits = [value for value in [customer.name, customer.email, customer.phone] if value]
            customer_line = f"Customer details: {', '.join(customer_bits)}.\n"

        transcript = (
            f"{customer_line}"
            f"Issue summary: {by_key.get('issue_summary', 'Not provided')}.\n"
            f"Reference details: {by_key.get('reference_details', 'Not provided')}.\n"
            f"Timeline: {by_key.get('timeline', 'Not provided')}.\n"
            f"Customer impact: {by_key.get('customer_impact', 'Not provided')}.\n"
            f"Urgency and risk: {by_key.get('urgency', 'Not provided')}.\n"
            f"Desired resolution: {by_key.get('desired_resolution', 'Not provided')}."
        )
        return transcript[:8000]

    def _build_question_with_nova(
        self,
        step: IntakeStep,
        history: list[IntakeAnswer],
        language: str,
    ) -> IntakeQuestion:
        history_json = json.dumps([item.model_dump() for item in history])
        prompt = (
            "You are helping a support agent gather missing context from a customer.\n"
            "Return strict JSON only with this shape:\n"
            "{\n"
            '  "prompt": "string",\n'
            '  "help_text": "string"\n'
            "}\n"
            "Write one concise follow-up question for the next missing slot.\n"
            "Keep the wording simple, direct, and suitable for a live support call.\n"
            "Do not ask for information already captured.\n"
            f"Language: {language}\n"
            f"Next slot label: {step.label}\n"
            f"Default prompt: {step.prompt}\n"
            f"Default help text: {step.help_text}\n"
            f"Existing intake history: {history_json}\n"
        )
        raw = self.llm.converse(prompt, max_tokens=180, temperature=0.2)
        parsed = self._extract_json(raw)
        refined_prompt = str(parsed.get("prompt", step.prompt)).strip() or step.prompt
        refined_help = str(parsed.get("help_text", step.help_text)).strip() or step.help_text
        return IntakeQuestion(
            key=step.key,
            label=step.label,
            prompt=refined_prompt[:300],
            help_text=refined_help[:240],
        )

    def _extract_json(self, text: str) -> dict[str, object]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Model output did not contain JSON object")

        return json.loads(cleaned[start : end + 1])
