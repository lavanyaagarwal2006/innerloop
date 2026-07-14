"""Central config. Reads everything from .env so no secrets live in code."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _req(key: str) -> str:
    val = os.getenv(key, "").strip()
    if not val:
        raise RuntimeError(
            f"Missing required env var: {key}. "
            f"Copy .env.example to .env and fill it in."
        )
    return val


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str


@dataclass(frozen=True)
class RivaConfig:
    host: str
    language_code: str
    use_ssl: bool


@dataclass(frozen=True)
class STTRestConfig:
    url: str
    client_id: str
    client_key: str
    auth_basic: str
    language: str


@dataclass(frozen=True)
class TTSConfig:
    url: str
    auth_basic: str
    client_id: str
    client_key: str
    language: str


def llm() -> LLMConfig:
    return LLMConfig(
        base_url=_req("LLM_BASE_URL"),
        api_key=_req("LLM_API_KEY"),
        model=os.getenv("LLM_MODEL", "gpt-oss-120b"),
    )


def riva() -> RivaConfig:
    return RivaConfig(
        host=_req("RIVA_HOST"),
        language_code=os.getenv("RIVA_LANGUAGE_CODE", "hi-IN"),
        use_ssl=os.getenv("RIVA_USE_SSL", "false").lower() == "true",
    )


def stt_rest() -> STTRestConfig:
    return STTRestConfig(
        url=_req("STT_REST_URL"),
        client_id=_req("STT_CLIENT_ID"),
        client_key=_req("STT_CLIENT_KEY"),
        auth_basic=_req("STT_AUTH_BASIC"),
        language=os.getenv("STT_LANGUAGE", "ENG"),
    )


def tts() -> TTSConfig:
    return TTSConfig(
        url=_req("TTS_URL"),
        auth_basic=_req("TTS_AUTH_BASIC"),
        client_id=_req("TTS_CLIENT_ID"),
        client_key=_req("TTS_CLIENT_KEY"),
        language=os.getenv("TTS_LANGUAGE", "HIN"),
    )
