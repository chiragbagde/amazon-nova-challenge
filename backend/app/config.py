import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    region: str
    bedrock_model_id: str
    bedrock_api_key: str
    cors_origins: list[str]
    tickets_file: Path


def load_settings() -> Settings:
    base_dir = Path(__file__).resolve().parents[1]
    tickets_file = Path(os.getenv("TICKETS_FILE", base_dir / "data" / "tickets.json"))

    raw_origins = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:4300,http://127.0.0.1:4300,http://localhost:5173,http://127.0.0.1:5173",
    )
    cors_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]

    return Settings(
        region=os.getenv("AWS_REGION", "us-east-1"),
        bedrock_model_id=os.getenv("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0"),
        bedrock_api_key=os.getenv("AWS_BEDROCK_KEY", ""),
        cors_origins=cors_origins,
        tickets_file=tickets_file,
    )
