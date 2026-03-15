"""
Nova Sonic intake service.

Runs a bidirectional Sonic session for the 6-question support intake interview.
The model speaks each question aloud and accepts the agent's spoken answer.
On completion it returns a compiled transcript ready for Nova Lite analysis.

Two modes:
  • text_mode=True  – no real audio; sends text turns via the Sonic stream.
                      Used for the REST fallback and testing.
  • text_mode=False – full audio pipeline; caller supplies raw PCM chunks via
                      the `audio_in` async-generator and receives PCM bytes
                      back via the `audio_out` async-generator.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import uuid
from contextlib import suppress
from typing import Any, AsyncGenerator, Dict, List, Optional

from dotenv import load_dotenv

# ── AWS Bedrock Sonic SDK ──────────────────────────────────────────────────────
try:
    from aws_sdk_bedrock_runtime.client import (
        BedrockRuntimeClient,
        InvokeModelWithBidirectionalStreamOperationInput,
    )
    from aws_sdk_bedrock_runtime.config import Config
    from aws_sdk_bedrock_runtime.models import (
        BidirectionalInputPayloadPart,
        InvokeModelWithBidirectionalStreamInputChunk,
    )
    from smithy_aws_core.identity.environment import EnvironmentCredentialsResolver

    SONIC_SDK_AVAILABLE = True
except ImportError:
    SONIC_SDK_AVAILABLE = False


# ── Intake question script ─────────────────────────────────────────────────────

INTAKE_QUESTIONS: List[Dict[str, str]] = [
    {
        "key": "issue_summary",
        "label": "Issue Summary",
        "spoken": (
            "Hello! I'm your Nova support assistant. "
            "Let's walk through this case together. "
            "First, can you briefly describe what the customer is reporting?"
        ),
    },
    {
        "key": "reference_details",
        "label": "Reference Details",
        "spoken": (
            "Got it. Which order, account, product, or transaction is affected? "
            "Please share any reference numbers or identifiers you have."
        ),
    },
    {
        "key": "timeline",
        "label": "Timeline",
        "spoken": (
            "When did this issue first start, and what has already happened so far? "
            "Include any prior support interactions if relevant."
        ),
    },
    {
        "key": "customer_impact",
        "label": "Customer Impact",
        "spoken": (
            "How is this affecting the customer right now? "
            "Tell me about any business impact, blocked work, or emotional distress."
        ),
    },
    {
        "key": "urgency",
        "label": "Urgency",
        "spoken": (
            "Is there a deadline, escalation threat, or serious risk if this isn't resolved quickly? "
            "Any cancellation or legal concerns?"
        ),
    },
    {
        "key": "desired_resolution",
        "label": "Desired Resolution",
        "spoken": (
            "Finally, what outcome does the customer want from support? "
            "For example: a refund, replacement, callback, or account recovery."
        ),
    },
]

SYSTEM_PROMPT = (
    "You are Nova, a professional support intake agent. "
    "Your job is to gather structured information about a customer support case "
    "from the support agent who is assisting the customer. "
    "Ask each question clearly and concisely. "
    "Acknowledge every answer briefly before moving to the next question. "
    "Keep your tone calm, professional, and friendly. "
    "Do not add extra commentary beyond short acknowledgements."
)


# ── Low-level event builders ───────────────────────────────────────────────────

def _chunk(payload: Dict[str, Any]) -> "InvokeModelWithBidirectionalStreamInputChunk":
    data = json.dumps(payload).encode("utf-8")
    return InvokeModelWithBidirectionalStreamInputChunk(
        value=BidirectionalInputPayloadPart(bytes_=data)
    )


def _ev_session_start(max_tokens: int = 512) -> Dict:
    return {"event": {"sessionStart": {"inferenceConfiguration": {
        "maxTokens": max_tokens, "topP": 0.9, "temperature": 0.3,
    }}}}


def _ev_prompt_start(pname: str) -> Dict:
    return {"event": {"promptStart": {
        "promptName": pname,
        "textOutputConfiguration": {"mediaType": "text/plain"},
        "audioOutputConfiguration": {
            "mediaType": "audio/lpcm",
            "sampleRateHertz": 24000,
            "sampleSizeBits": 16,
            "channelCount": 1,
            "voiceId": "matthew",
            "encoding": "base64",
            "audioType": "SPEECH",
        },
    }}}


def _ev_content_start(pname: str, cname: str, role: str, ctype: str, interactive: bool) -> Dict:
    ev: Dict[str, Any] = {
        "promptName": pname,
        "contentName": cname,
        "role": role,
        "type": ctype,
        "interactive": interactive,
    }
    if ctype == "TEXT":
        ev["textInputConfiguration"] = {"mediaType": "text/plain"}
    elif ctype == "AUDIO":
        ev["audioInputConfiguration"] = {
            "mediaType": "audio/lpcm",
            "sampleRateHertz": 16000,
            "sampleSizeBits": 16,
            "channelCount": 1,
            "audioType": "SPEECH",
            "encoding": "base64",
        }
    return {"event": {"contentStart": ev}}


def _ev_text_input(pname: str, cname: str, content: str) -> Dict:
    return {"event": {"textInput": {"promptName": pname, "contentName": cname, "content": content}}}


def _ev_audio_input(pname: str, cname: str, b64: str) -> Dict:
    return {"event": {"audioInput": {"promptName": pname, "contentName": cname, "content": b64}}}


def _ev_content_end(pname: str, cname: str) -> Dict:
    return {"event": {"contentEnd": {"promptName": pname, "contentName": cname}}}


def _ev_prompt_end(pname: str) -> Dict:
    return {"event": {"promptEnd": {"promptName": pname}}}


def _ev_session_end() -> Dict:
    return {"event": {"sessionEnd": {}}}


# ── Main service class ─────────────────────────────────────────────────────────

class SonicIntakeService:
    """
    Runs one complete Nova Sonic intake interview (6 questions) in text mode.

    Returns a dict with:
      • answers   – list of {key, label, answer} dicts
      • transcript – compiled case brief string
      • full_text  – raw concatenated Sonic text output
      • success    – bool
      • errors     – list of error strings
    """

    def __init__(
        self,
        model_id: str = "amazon.nova-2-sonic-v1:0",
        region: str = "us-east-1",
        timeout_per_turn: int = 30,
    ) -> None:
        self.model_id = model_id
        self.region = region
        self.timeout_per_turn = timeout_per_turn

    # ── public API ─────────────────────────────────────────────────────────────

    async def run_text_interview(
        self,
        agent_answers: List[str],
        language: str = "en",
    ) -> Dict[str, Any]:
        """
        Conduct the intake interview in text mode.
        `agent_answers` must have exactly 6 strings (one per intake question).
        """
        if not SONIC_SDK_AVAILABLE:
            return self._fallback_text_interview(agent_answers)

        load_dotenv()

        if len(agent_answers) != len(INTAKE_QUESTIONS):
            raise ValueError(f"Expected {len(INTAKE_QUESTIONS)} answers, got {len(agent_answers)}")

        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self.region}.amazonaws.com",
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
        )
        client = BedrockRuntimeClient(config=config)
        stream = await client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
        )

        pname = str(uuid.uuid4())
        sys_cname = str(uuid.uuid4())
        sys_speech_cname = str(uuid.uuid4())
        audio_cname = str(uuid.uuid4())

        async def send(ev: Dict) -> None:
            await stream.input_stream.send(_chunk(ev))

        # ── session setup ──────────────────────────────────────────────────────
        lang_note = f" Respond in {'Hindi' if language == 'hi' else 'English'}." 
        await send(_ev_session_start(max_tokens=512))
        await send(_ev_prompt_start(pname))
        await send(_ev_content_start(pname, sys_cname, "SYSTEM", "TEXT", False))
        await send(_ev_text_input(pname, sys_cname, SYSTEM_PROMPT + lang_note))
        await send(_ev_content_end(pname, sys_cname))

        if "nova-2-sonic" in self.model_id:
            await send(_ev_content_start(pname, sys_speech_cname, "SYSTEM_SPEECH", "TEXT", False))
            await send(_ev_text_input(pname, sys_speech_cname, ""))
            await send(_ev_content_end(pname, sys_speech_cname))

        # send silent audio to keep the session alive
        await send(_ev_content_start(pname, audio_cname, "USER", "AUDIO", True))

        silent_b64 = base64.b64encode(b"\x00" * 2048).decode()
        stop_silence = asyncio.Event()

        async def _silence_loop() -> None:
            while not stop_silence.is_set():
                await send(_ev_audio_input(pname, audio_cname, silent_b64))
                await asyncio.sleep(0.02)

        silence_task = asyncio.create_task(_silence_loop())

        # ── conversation turns ─────────────────────────────────────────────────
        all_text_chunks: List[str] = []
        errors: List[str] = []
        collected_answers: List[Dict[str, str]] = []

        try:
            for i, (question, agent_answer) in enumerate(
                zip(INTAKE_QUESTIONS, agent_answers)
            ):
                # Send question as assistant turn
                q_cname = str(uuid.uuid4())
                await send(_ev_content_start(pname, q_cname, "ASSISTANT", "TEXT", False))
                await send(_ev_text_input(pname, q_cname, question["spoken"]))
                await send(_ev_content_end(pname, q_cname))

                # Send agent answer as user turn
                a_cname = str(uuid.uuid4())
                await send(_ev_content_start(pname, a_cname, "USER", "TEXT", True))
                await send(_ev_text_input(pname, a_cname, agent_answer.strip()))
                await send(_ev_content_end(pname, a_cname))

                # Collect model acknowledgement
                turn_text = await self._collect_turn(stream, self.timeout_per_turn)
                all_text_chunks.append(turn_text)

                collected_answers.append({
                    "key": question["key"],
                    "label": question["label"],
                    "answer": agent_answer.strip(),
                    "nova_ack": turn_text,
                })

        except Exception as exc:
            errors.append(f"interview_error: {exc}")

        finally:
            stop_silence.set()
            with suppress(Exception):
                await asyncio.wait_for(silence_task, timeout=3)
            if not silence_task.done():
                silence_task.cancel()

            for ev in [
                _ev_content_end(pname, audio_cname),
                _ev_prompt_end(pname),
                _ev_session_end(),
            ]:
                with suppress(Exception):
                    await send(ev)

            with suppress(Exception):
                await stream.input_stream.close()

        transcript = self._compile_transcript(collected_answers)
        return {
            "answers": collected_answers,
            "transcript": transcript,
            "full_text": "\n".join(all_text_chunks).strip(),
            "success": len(collected_answers) == len(INTAKE_QUESTIONS) and not errors,
            "errors": errors,
        }

    # ── helpers ────────────────────────────────────────────────────────────────

    async def _collect_turn(self, stream: Any, timeout: int) -> str:
        chunks: List[str] = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                output = await asyncio.wait_for(stream.await_output(), timeout=2)
                message = await asyncio.wait_for(output[1].receive(), timeout=2)
            except asyncio.TimeoutError:
                if chunks:
                    break
                continue
            except Exception:
                break

            payload = getattr(message, "value", None)
            if payload is None:
                continue

            with suppress(Exception):
                raw = getattr(payload, "bytes_", None)
                if not raw:
                    continue
                body = json.loads(raw.decode("utf-8"))
                event = body.get("event", {})

                if "textOutput" in event:
                    text = event["textOutput"].get("content", "")
                    if text and (not chunks or text != chunks[-1]):
                        chunks.append(text)

                if "completionEnd" in event:
                    break

                if "validationException" in event or "modelStreamErrorException" in event:
                    break

        return " ".join(chunks).strip()

    def _compile_transcript(self, answers: List[Dict[str, str]]) -> str:
        by_key = {a["key"]: a["answer"] for a in answers}
        return (
            f"Issue summary: {by_key.get('issue_summary', 'Not provided')}.\n"
            f"Reference details: {by_key.get('reference_details', 'Not provided')}.\n"
            f"Timeline: {by_key.get('timeline', 'Not provided')}.\n"
            f"Customer impact: {by_key.get('customer_impact', 'Not provided')}.\n"
            f"Urgency and risk: {by_key.get('urgency', 'Not provided')}.\n"
            f"Desired resolution: {by_key.get('desired_resolution', 'Not provided')}."
        )

    def _fallback_text_interview(self, agent_answers: List[str]) -> Dict[str, Any]:
        """Used when the Sonic SDK is not installed. Mirrors structure of real result."""
        collected = [
            {"key": q["key"], "label": q["label"], "answer": a.strip(), "nova_ack": ""}
            for q, a in zip(INTAKE_QUESTIONS, agent_answers)
        ]
        return {
            "answers": collected,
            "transcript": self._compile_transcript(collected),
            "full_text": "",
            "success": True,
            "errors": ["sonic_sdk_unavailable – used plain fallback"],
        }
