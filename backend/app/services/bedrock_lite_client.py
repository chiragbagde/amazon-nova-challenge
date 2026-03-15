from __future__ import annotations

from typing import Any

import requests


class BedrockLiteClient:
    def __init__(self, api_key: str, region: str, model_id: str) -> None:
        self.api_key = api_key
        self.region = region
        self.model_id = model_id

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def converse(self, prompt: str, max_tokens: int = 700, temperature: float = 0.2) -> str:
        if not self.api_key:
            raise ValueError("AWS_BEDROCK_KEY is missing")

        url = f"https://bedrock-runtime.{self.region}.amazonaws.com/model/{self.model_id}/converse"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt}],
                }
            ],
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        }

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(f"Bedrock error ({response.status_code}): {response.text}")

        body = response.json()
        return body["output"]["message"]["content"][0]["text"]