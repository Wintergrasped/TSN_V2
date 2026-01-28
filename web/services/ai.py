"""Lightweight AI helpers for the portal, using vLLM with OpenAI fallback."""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Sequence

import httpx
from openai import AsyncOpenAI

from tsn_common.config import get_settings
from tsn_common.db import get_session
from tsn_common.logging import get_logger
from tsn_common.models.support import AiRunLog, ProcessingMetric
from web.config import get_web_settings

logger = get_logger(__name__)

_PORTAL_LLM_TIMEOUT_SEC = 6


async def _record_portal_ai_metric(
    *,
    backend: str,
    pass_label: str,
    latency_ms: int | None,
    success: bool,
    error: str | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
    metadata: dict[str, Any] | None,
) -> None:
    payload = {
        "pass_label": pass_label,
        "source": "web_portal",
    }
    if metadata:
        payload.update(metadata)

    async with get_session() as session:
        metric = ProcessingMetric(
            stage=f"ai_pass_{backend}",
            processing_time_ms=max(0, latency_ms or 0),
            success=success,
            error_message=error,
            timestamp=datetime.now(timezone.utc),
            metadata_={
                **payload,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        )
        session.add(metric)
        await session.flush()


async def _log_portal_ai_run(
    *,
    backend: str,
    model: str | None,
    pass_label: str,
    prompt_text: str,
    response_text: str | None,
    success: bool,
    latency_ms: int | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
    metadata: dict[str, Any] | None,
    error_message: str | None,
) -> None:
    payload = {"source": "web_portal"}
    if metadata:
        payload.update(metadata)

    async with get_session() as session:
        entry = AiRunLog(
            backend=backend,
            model=model,
            pass_label=pass_label,
            success=success,
            error_message=error_message,
            prompt_text=prompt_text,
            response_text=response_text,
            prompt_characters=len(prompt_text or ""),
            response_characters=len(response_text) if response_text else None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            metadata_=payload,
        )
        session.add(entry)
        await session.flush()


def _build_prompt_blob(messages: list[dict[str, str]]) -> str:
    return json.dumps(messages, ensure_ascii=False)


def _extract_usage_value(usage: dict[str, Any] | None, keys: Sequence[str]) -> int | None:
    if not usage:
        return None
    for key in keys:
        value = usage.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


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


async def _call_vllm(
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    pass_label: str,
    prompt_text: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Call the configured vLLM endpoint with detailed logging."""

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
        "response_format": {"type": "json_object"},
    }

    client_timeout = min(settings.timeout_sec, _PORTAL_LLM_TIMEOUT_SEC)
    async with httpx.AsyncClient(timeout=client_timeout) as client:
        for attempt, base in enumerate(base_urls, start=1):
            try:
                start_time = time.perf_counter()
                response = await client.post(
                    _endpoint(base),
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                usage = data.get("usage") or {}
                prompt_tokens = _extract_usage_value(usage, ("prompt_tokens", "input_tokens"))
                completion_tokens = _extract_usage_value(usage, ("completion_tokens", "output_tokens"))
                total_tokens = _extract_usage_value(usage, ("total_tokens",))
                content = data["choices"][0]["message"]["content"]

                extra_meta = {"endpoint": base, "attempt": attempt, **(metadata or {})}
                await _record_portal_ai_metric(
                    backend="vllm",
                    pass_label=pass_label,
                    latency_ms=latency_ms,
                    success=True,
                    error=None,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    metadata=extra_meta,
                )
                await _log_portal_ai_run(
                    backend="vllm",
                    model=data.get("model") or settings.model,
                    pass_label=pass_label,
                    prompt_text=prompt_text,
                    response_text=content,
                    success=True,
                    latency_ms=latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    metadata=extra_meta,
                    error_message=None,
                )
                return content
            except Exception as exc:  # pragma: no cover - network variability
                error_msg = str(exc)
                logger.warning("vllm_request_failed", base=base, error=error_msg)
                extra_meta = {"endpoint": base, "attempt": attempt, **(metadata or {})}
                await _record_portal_ai_metric(
                    backend="vllm",
                    pass_label=pass_label,
                    latency_ms=None,
                    success=False,
                    error=error_msg,
                    prompt_tokens=None,
                    completion_tokens=None,
                    total_tokens=None,
                    metadata=extra_meta,
                )
                await _log_portal_ai_run(
                    backend="vllm",
                    model=settings.model,
                    pass_label=pass_label,
                    prompt_text=prompt_text,
                    response_text=None,
                    success=False,
                    latency_ms=None,
                    prompt_tokens=None,
                    completion_tokens=None,
                    total_tokens=None,
                    metadata=extra_meta,
                    error_message=error_msg,
                )
                continue

    raise RuntimeError("All vLLM endpoints failed")


async def _call_openai(
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    pass_label: str,
    prompt_text: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Fallback to OpenAI when configured, with logging."""

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

    start_time = time.perf_counter()
    try:
        response = await client.responses.create(
            model=settings.openai_model,
            input=formatted_input,
            temperature=0.2,
            max_output_tokens=max_tokens,
        )
    except Exception as exc:  # pragma: no cover - remote dependency
        error_msg = str(exc)
        extra_meta = {"endpoint": "openai_responses", **(metadata or {})}
        await _record_portal_ai_metric(
            backend="openai",
            pass_label=pass_label,
            latency_ms=None,
            success=False,
            error=error_msg,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            metadata=extra_meta,
        )
        await _log_portal_ai_run(
            backend="openai",
            model=settings.openai_model,
            pass_label=pass_label,
            prompt_text=prompt_text,
            response_text=None,
            success=False,
            latency_ms=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            metadata=extra_meta,
            error_message=error_msg,
        )
        raise

    latency_ms = int((time.perf_counter() - start_time) * 1000)
    usage = getattr(response, "usage", None) or {}
    prompt_tokens = _extract_usage_value(usage, ("prompt_tokens", "input_tokens"))
    completion_tokens = _extract_usage_value(usage, ("completion_tokens", "output_tokens"))
    total_tokens = _extract_usage_value(usage, ("total_tokens", "output_tokens"))

    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", "") in {"text", "output_text"} and getattr(content, "text", None):
                chunks.append(content.text)

    text_response = "".join(chunks)
    extra_meta = {"endpoint": "openai_responses", **(metadata or {})}
    await _record_portal_ai_metric(
        backend="openai",
        pass_label=pass_label,
        latency_ms=latency_ms,
        success=True,
        error=None,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        metadata=extra_meta,
    )
    await _log_portal_ai_run(
        backend="openai",
        model=settings.openai_model,
        pass_label=pass_label,
        prompt_text=prompt_text,
        response_text=text_response,
        success=True,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        metadata=extra_meta,
        error_message=None,
    )

    return text_response


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
    pass_label: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Send a structured prompt and return parsed JSON with fallbacks."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt_blob = _build_prompt_blob(messages)
    log_meta = dict(metadata or {})

    try:
        content = await asyncio.wait_for(
            _call_vllm(
                messages,
                max_tokens=max_tokens,
                pass_label=pass_label,
                prompt_text=prompt_blob,
                metadata=log_meta,
            ),
            timeout=_PORTAL_LLM_TIMEOUT_SEC,
        )
        if content:
            return _parse_json(content)
    except asyncio.TimeoutError:
        error_msg = "timeout"
        logger.warning("vllm_unavailable", error=error_msg, timeout_sec=_PORTAL_LLM_TIMEOUT_SEC)
        timeout_meta = {"timeout_sec": _PORTAL_LLM_TIMEOUT_SEC, **log_meta}
        await _record_portal_ai_metric(
            backend="vllm",
            pass_label=pass_label,
            latency_ms=None,
            success=False,
            error=error_msg,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            metadata=timeout_meta,
        )
        await _log_portal_ai_run(
            backend="vllm",
            model=get_settings().vllm.model,
            pass_label=pass_label,
            prompt_text=prompt_blob,
            response_text=None,
            success=False,
            latency_ms=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            metadata=timeout_meta,
            error_message=error_msg,
        )
    except Exception as exc:  # pragma: no cover - LLM ops
        logger.warning("vllm_unavailable", error=str(exc))

    try:
        content = await asyncio.wait_for(
            _call_openai(
                messages,
                max_tokens=max_tokens,
                pass_label=pass_label,
                prompt_text=prompt_blob,
                metadata=log_meta,
            ),
            timeout=_PORTAL_LLM_TIMEOUT_SEC,
        )
        if content:
            return _parse_json(content)
    except asyncio.TimeoutError:
        error_msg = "timeout"
        logger.error("openai_fallback_failed", error=error_msg, timeout_sec=_PORTAL_LLM_TIMEOUT_SEC)
        timeout_meta = {"timeout_sec": _PORTAL_LLM_TIMEOUT_SEC, **log_meta}
        await _record_portal_ai_metric(
            backend="openai",
            pass_label=pass_label,
            latency_ms=None,
            success=False,
            error=error_msg,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            metadata=timeout_meta,
        )
        await _log_portal_ai_run(
            backend="openai",
            model=get_settings().vllm.openai_model,
            pass_label=pass_label,
            prompt_text=prompt_blob,
            response_text=None,
            success=False,
            latency_ms=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            metadata=timeout_meta,
            error_message=error_msg,
        )
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
        pass_label="dashboard_summary",
        metadata={"section_keys": list(sections.keys())},
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
        pass_label="callsign_summary",
        metadata={"callsign": callsign},
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
        pass_label="club_summary",
        metadata={"club": name},
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
        pass_label=f"merge_{entity_type}",
        metadata={"entity_type": entity_type, "alias_count": len(unique)},
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
