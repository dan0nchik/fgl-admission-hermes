"""Minimal OpenAI-compatible async client factory (standalone copy for this repo)."""
from typing import Any, Dict
from openai import AsyncOpenAI


def create_client(api_key: str, base_url: str = "", proxy: str = "", **_: Any):
    kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    if proxy:
        import httpx
        kwargs["http_client"] = httpx.AsyncClient(proxy=proxy)
    return AsyncOpenAI(**kwargs)
