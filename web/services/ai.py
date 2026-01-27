"""Lightweight AI helpers for the portal, using vLLM with OpenAI fallback."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Sequence

import httpx
from openai import AsyncOpenAI

from tsn_common.config import get_settings
from tsn_common.logging import get_logger
from web.config import get_web_settings

logger = get_logger(__name__)

_PORTAL_LLM_TIMEOUT_SEC = 6


def _resolve_api_keys() -> tuple[str | None, str | None]:
    """Return (vllm_key, openai_key) using portal overrides and env fallbacks."""

    llm_settings = get_settings().vllm
    web_settings = get_web_settings()

    candidates = [
        (web_settings.vllm_api_key or "").strip(),
        (os.getenv("TSN_WEB_VLLM_API_KEY") or "").strip(),
        (os.getenv("TSN_VLLM_API_KEY") or "").strip(),
        llm_settings.api_key.get_secret_value().strip(),
    ]
    vllm_key = next((val for val in candidates if val), None)

    openai_candidates = [
        (web_settings.openai_api_key or "").strip(),
        (os.getenv("TSN_WEB_OPENAI_API_KEY") or "").strip(),
        (os.getenv("TSN_OPENAI_API_KEY") or "").strip(),
        (llm_settings.openai_api_key.get_secret_value().strip() if llm_settings.openai_api_key else ""),
    ]
    openai_key = next((val for val in openai_candidates if val), None)

    return vllm_key, openai_key


async def _call_vllm(messages: list[dict[str, str]], *, max_tokens: int) -> str:
    """Call the configured vLLM endpoint, falling back to loopback."""

    settings = get_settings().vllm
    api_key, _ = _resolve_api_keys()
    if not api_key:
        raise RuntimeError("TSN_VLLM_API_KEY / TSN_WEB_VLLM_API_KEY is not configured")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    base_urls = [settings.base_url.rstrip("/")]
    loopback = "http://127.0.0.1:8001"
    if loopback not in base_urls:
        base_urls.append(loopback)

    def _endpoint(base: str) -> str:
        base = base.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    payload = {
        "model": settings.model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }

    client_timeout = min(settings.timeout_sec, _PORTAL_LLM_TIMEOUT_SEC)
    async with httpx.AsyncClient(timeout=client_timeout) as client:
        for base in base_urls:
            try:
                response = await client.post(
                    _endpoint(base),
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except Exception as exc:  # pragma: no cover - network variability
                logger.warning("vllm_request_failed", base=base, error=str(exc))
                continue

    raise RuntimeError("All vLLM endpoints failed")


async def _call_openai(messages: list[dict[str, str]], *, max_tokens: int) -> str:
    """Fallback to OpenAI when configured."""

    settings = get_settings().vllm
    _, api_key = _resolve_api_keys()
    if not (settings.fallback_enabled and api_key):
        raise RuntimeError("OpenAI fallback is not configured")

    client = AsyncOpenAI(api_key=api_key)
    # Responses API requires structured content blocks instead of legacy chat payloads.
    formatted_input = [
        {
            "role": message["role"],
            "content": [{"type": "text", "text": message["content"]}],
        }
        for message in messages
    ]

    response = await client.responses.create(
        model=settings.openai_model,
        input=formatted_input,
        temperature=0.2,
        max_output_tokens=max_tokens,
    )

    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", "") in {"text", "output_text"} and getattr(content, "text", None):
                chunks.append(content.text)

    return "".join(chunks)


def _parse_json(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        logger.warning("llm_response_not_json", preview=content[:120])
        return {}


async def invoke_json_prompt(
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 512,
) -> dict[str, Any]:
    """Send a structured prompt and return parsed JSON with fallbacks."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        content = await asyncio.wait_for(
            _call_vllm(messages, max_tokens=max_tokens),
            timeout=_PORTAL_LLM_TIMEOUT_SEC,
        )
        if content:
            return _parse_json(content)
    except asyncio.TimeoutError:
        logger.warning("vllm_unavailable", error="timeout", timeout_sec=_PORTAL_LLM_TIMEOUT_SEC)
    except Exception as exc:  # pragma: no cover - LLM ops
        logger.warning("vllm_unavailable", error=str(exc))

    try:
        content = await asyncio.wait_for(
            _call_openai(messages, max_tokens=max_tokens),
            timeout=_PORTAL_LLM_TIMEOUT_SEC,
        )
        if content:
            return _parse_json(content)
    except asyncio.TimeoutError:
        logger.error("openai_fallback_failed", error="timeout", timeout_sec=_PORTAL_LLM_TIMEOUT_SEC)
    except Exception as exc:  # pragma: no cover - LLM ops
        logger.error("openai_fallback_failed", error=str(exc))

    return {}


async def summarize_dashboard_sections(sections: dict[str, Any]) -> dict[str, str]:
    """Return short natural language blurbs for major dashboard widgets."""

    user_prompt = (
        "Summarize these TSN dashboard sections for a radio operator. "
        "Respond with JSON keys: queue, nets, clubs, callsigns, health. "
        f"Sections: {json.dumps(sections)[:6000]}"
    )
    data = await invoke_json_prompt(
        system_prompt=(
            "You are TSN Copilot. Provide confident, tactical summaries in 2 sentences max per section."
        ),
        user_prompt=user_prompt,
        max_tokens=400,
    )
    return {
        "queue": data.get("queue", "Queue status unavailable."),
        "nets": data.get("nets", "No recent net insight."),
        "clubs": data.get("clubs", "No club commentary."),
        "callsigns": data.get("callsigns", "Callsign summary pending."),
        "health": data.get("health", "Health summary pending."),
    }


async def summarize_callsign(callsign: str, payload: dict[str, Any]) -> str:
    """Generate a profile summary for a callsign."""

    user_prompt = (
        f"Summarize callsign {callsign} for the portal using this JSON payload: "
        f"{json.dumps(payload)[:6000]}"
    )
    data = await invoke_json_prompt(
        system_prompt="You are KK7NQN Net Control analyst.",
        user_prompt=user_prompt,
        max_tokens=350,
    )
    return data.get("summary", "No AI summary available yet.")


async def summarize_club(name: str, payload: dict[str, Any]) -> str:
    """Summarize a club profile."""

    user_prompt = (
        f"Summarize club {name} for TSN operators in under 120 words. Payload: "
        f"{json.dumps(payload)[:6000]}"
    )
    data = await invoke_json_prompt(
        system_prompt="You distill amateur radio club intelligence for dashboard cards.",
        user_prompt=user_prompt,
        max_tokens=320,
    )
    return data.get("summary", "Summary not ready.")


async def merge_entities(entity_type: str, names: Sequence[str]) -> dict[str, str]:
    """Map aliases to canonical entities using the LLM."""

    unique = sorted({n for n in names if n})
    if not unique:
        return {}

    prompt = (
        f"Merge duplicate {entity_type} names (e.g., PSRG vs Puget Sound Repeater Group). "
        "Return JSON of the form {\"aliases\": [{\"canonical\": str, \"alias\": [..]}]}. "
        f"Names: {json.dumps(unique)}"
    )
    data = await invoke_json_prompt(
        system_prompt="You reconcile noisy radio metadata into canonical labels.",
        user_prompt=prompt,
        max_tokens=400,
    )

    mapping: dict[str, str] = {}
    for entry in data.get("aliases", []):
        canonical = entry.get("canonical")
        if not canonical:
            continue
        for alias in entry.get("alias", []):
            mapping[alias] = canonical
        mapping.setdefault(canonical, canonical)
    return mapping
