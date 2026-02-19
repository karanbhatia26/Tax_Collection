from __future__ import annotations

from functools import lru_cache
from typing import Dict

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    judge0_base_url: str = "https://ce.judge0.com"
    judge0_timeout_seconds: int = 30

    groq_api_key: str | None = None
    groq_model: str = "llama-3.1-8b-instant"
    ai_max_tokens: int = 220
    ai_temperature: float = 0.7

    default_language_id: int = 71
    language_map: Dict[str, int] = {
        "python": 71,
        "javascript": 63,
        "java": 62,
        "cpp": 54,
        "c": 50,
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()
