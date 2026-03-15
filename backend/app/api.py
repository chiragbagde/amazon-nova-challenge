from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .config import load_settings
from .schemas import (
    AnalyzeResponse,
    CreateTicketRequest,
    IntakeNextRequest,
    IntakeNextResponse,
    SonicIntakeRequest,
    SonicIntakeResponse,
    SonicAnswerItem,
    TicketListResponse,
    TicketRecord,
    TranscriptAnalyzeRequest,
)
from .services.bedrock_lite_client import BedrockLiteClient
from .services.intake_service import IntakeService
from .services.sonic_service import INTAKE_QUESTIONS, SonicIntakeService
from .services.ticket_analyzer import TicketAnalyzer
from .storage import TicketStore


def create_app() -> FastAPI:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    settings = load_settings()

    app = FastAPI(
        title="Voice-to-Ticket Auto Resolver API",
        version="0.1.0",
        description="Amazon Nova hackathon backend for converting voice transcripts into support tickets.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    analyzer = TicketAnalyzer(
        llm=BedrockLiteClient(
            api_key=settings.bedrock_api_key,
            region=settings.region,
            model_id=settings.bedrock_model_id,
        )
    )
    intake = IntakeService(
        llm=BedrockLiteClient(
            api_key=settings.bedrock_api_key,
            region=settings.region,
            model_id=settings.bedrock_model_id,
        )
    )
    sonic = SonicIntakeService(
        model_id="amazon.nova-2-sonic-v1:0",
        region=settings.region,
    )
    store = TicketStore(settings.tickets_file)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # ── Nova Sonic routes ──────────────────────────────────────────────────

    @app.get("/api/sonic/questions")
    def sonic_questions() -> dict:
        """Return the ordered list of intake questions (label + spoken prompt)."""
        return {
            "questions": [
                {"key": q["key"], "label": q["label"], "spoken": q["spoken"]}
                for q in INTAKE_QUESTIONS
            ]
        }

    @app.get("/api/sonic/status")
    def sonic_status() -> dict:
        """Check whether the Sonic SDK is available in this environment."""
        from .services.sonic_service import SONIC_SDK_AVAILABLE
        return {"sonic_available": SONIC_SDK_AVAILABLE, "model": "amazon.nova-2-sonic-v1:0"}

    @app.post("/api/sonic/interview", response_model=SonicIntakeResponse)
    async def sonic_interview(payload: SonicIntakeRequest) -> SonicIntakeResponse:
        """
        Run all 6 intake questions through Nova Sonic (text mode).
        The frontend posts the agent’s six answers; Sonic conducts the interview
        and returns a compiled transcript + Nova acknowledgements per question.
        Pass the returned `transcript` directly to POST /api/analyze.
        """
        try:
            result = await sonic.run_text_interview(
                agent_answers=payload.answers,
                language=payload.language,
            )
            return SonicIntakeResponse(
                answers=[SonicAnswerItem(**a) for a in result["answers"]],
                transcript=result["transcript"],
                full_text=result["full_text"],
                success=result["success"],
                errors=result["errors"],
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"sonic interview failed: {exc}") from exc

    # ── Legacy text intake routes ──────────────────────────────────────────────

    @app.post("/api/analyze", response_model=AnalyzeResponse)
    def analyze(payload: TranscriptAnalyzeRequest) -> AnalyzeResponse:
        try:
            return analyzer.analyze(payload)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"analyze failed: {exc}") from exc

    @app.post("/api/intake/next", response_model=IntakeNextResponse)
    def next_intake_step(payload: IntakeNextRequest) -> IntakeNextResponse:
        try:
            return intake.next_step(payload)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"intake failed: {exc}") from exc

    @app.post("/api/tickets", response_model=TicketRecord)
    def create_ticket(payload: CreateTicketRequest) -> TicketRecord:
        try:
            return store.create_ticket(payload)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"create ticket failed: {exc}") from exc

    @app.get("/api/tickets", response_model=TicketListResponse)
    def list_tickets(limit: int = Query(default=20, ge=1, le=100)) -> TicketListResponse:
        try:
            return store.list_tickets(limit)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"list tickets failed: {exc}") from exc

    return app


app = create_app()
