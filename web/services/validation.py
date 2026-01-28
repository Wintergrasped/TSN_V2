"""Validation service helpers (QRZ scheduling, etc.)."""

from __future__ import annotations

import asyncio
import os
import time

import httpx

from tsn_common.config import get_settings
from tsn_common.logging import get_logger

logger = get_logger(__name__)

_BACKFILL_URL = os.getenv("TSN_WEB_QRZ_BACKFILL_URL", "http://tsn_server:8080/api/v1/qrz/backfill")
_COOLDOWN_SEC = float(os.getenv("TSN_WEB_QRZ_BACKFILL_COOLDOWN_SEC", "300"))
_TASK: asyncio.Task | None = None
_LAST_ATTEMPT = 0.0


def _log_task_result(task: asyncio.Task) -> None:
    try:
        task.result()
    except asyncio.CancelledError:  # pragma: no cover - only during shutdown
        logger.debug("qrz_backfill_task_cancelled")
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("qrz_backfill_task_error", error=str(exc))


def _should_attempt_backfill() -> bool:
    if not _BACKFILL_URL:
        return False
    if not get_settings().qrz.enabled:
        return False

    now = time.time()
    global _LAST_ATTEMPT
    if now - _LAST_ATTEMPT < _COOLDOWN_SEC:
        return False
    _LAST_ATTEMPT = now
    return True


def schedule_qrz_backfill() -> None:
    """Attempt to queue a QRZ validation backfill without blocking the request."""

    global _TASK

    if _TASK and not _TASK.done():
        return

    if not _should_attempt_backfill():
        return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.debug("qrz_backfill_no_event_loop")
        return

    async def _invoke() -> None:
        payload = {"source": "web_dashboard"}
        timeout = httpx.Timeout(5.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(_BACKFILL_URL, json=payload)
                response.raise_for_status()
                logger.info("qrz_backfill_scheduled", status=response.status_code)
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "qrz_backfill_http_error",
                status=exc.response.status_code,
                detail=exc.response.text[:200],
            )
        except Exception as exc:  # pragma: no cover - network best effort
            logger.warning("qrz_backfill_request_failed", error=str(exc))

    _TASK = loop.create_task(_invoke())
    _TASK.add_done_callback(_log_task_result)
