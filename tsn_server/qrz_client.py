"""QRZ XML API helper utilities."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
import xml.etree.ElementTree as ET

import httpx

from tsn_common.config import get_settings
from tsn_common.logging import get_logger
from tsn_common.utils import normalize_callsign

logger = get_logger(__name__)

QRZ_ENDPOINT = "https://xmldata.qrz.com/xml/current/"
AGENT_NAME = "TSN-V2"
SESSION_TTL = timedelta(minutes=15)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _strip(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


@dataclass
class CacheEntry:
    expires_at: datetime
    data: dict[str, Any] | None


class QRZClient:
    """Thin asynchronous client for QRZ XML API lookups."""

    def __init__(self) -> None:
        settings = get_settings().qrz
        self.enabled = bool(
            settings.enabled and settings.username and settings.password
        )
        self.username = settings.username
        self.password = settings.password.get_secret_value() if settings.password else None
        self.cache_ttl = timedelta(seconds=settings.cache_ttl_sec)
        self._session_key: str | None = None
        self._session_expiry: datetime = _now()
        self._session_lock = asyncio.Lock()
        self._cache: dict[str, CacheEntry] = {}
        self._client = httpx.AsyncClient(timeout=30)

        if not self.enabled:
            logger.warning("qrz_client_disabled", reason="missing credentials or disabled flag")

    async def close(self) -> None:
        await self._client.aclose()

    async def lookup_many(self, callsigns: list[str]) -> dict[str, dict[str, Any] | None]:
        results: dict[str, dict[str, Any] | None] = {}
        for cs in callsigns:
            try:
                results[cs] = await self.lookup(cs)
            except Exception as exc:  # pragma: no cover - network variability
                logger.error("qrz_lookup_failed", callsign=cs, error=str(exc))
                results[cs] = None
        return results

    async def lookup(self, callsign: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None

        normalized = normalize_callsign(callsign)
        cached = self._cache.get(normalized)
        now = _now()
        if cached and cached.expires_at > now:
            return cached.data

        for attempt in range(2):
            await self._ensure_session(force=attempt == 1)
            if not self._session_key:
                break

            response = await self._client.get(
                QRZ_ENDPOINT,
                params={
                    "s": self._session_key,
                    "callsign": normalized,
                    "agent": AGENT_NAME,
                },
            )
            response.raise_for_status()
            root = ET.fromstring(response.text)
            self._update_session_from_payload(root)
            error = self._extract_error(root)
            data = self._extract_callsign_data(root)

            if error and "session" in error.lower():
                logger.info("qrz_session_expired", detail=error)
                self._session_key = None
                continue

            expiry = now + self.cache_ttl
            self._cache[normalized] = CacheEntry(expires_at=expiry, data=data)
            if not data and error:
                logger.warning("qrz_lookup_error", callsign=normalized, error=error)
            return data

        logger.error("qrz_lookup_gave_up", callsign=normalized)
        return None

    async def _ensure_session(self, *, force: bool = False) -> None:
        if not self.enabled:
            return
        if not force and self._session_key and _now() < self._session_expiry:
            return

        async with self._session_lock:
            if not force and self._session_key and _now() < self._session_expiry:
                return
            if not (self.username and self.password):
                logger.error("qrz_credentials_missing")
                self._session_key = None
                return

            response = await self._client.get(
                QRZ_ENDPOINT,
                params={
                    "username": self.username,
                    "password": self.password,
                    "agent": AGENT_NAME,
                },
            )
            response.raise_for_status()
            root = ET.fromstring(response.text)
            key = self._extract_session_key(root)
            if not key:
                error = self._extract_error(root)
                logger.error("qrz_login_failed", error=error)
                self._session_key = None
                return
            self._session_key = key
            self._session_expiry = _now() + SESSION_TTL
            logger.info("qrz_session_established", expires_at=self._session_expiry.isoformat())

    def _update_session_from_payload(self, root: ET.Element) -> None:
        key = self._extract_session_key(root)
        if key:
            self._session_key = key
            self._session_expiry = _now() + SESSION_TTL

    def _extract_session_key(self, root: ET.Element) -> str | None:
        session = self._find_first(root, "Session")
        if session is None:
            return None
        for child in session:
            if _strip(child.tag).lower() == "key" and child.text:
                return child.text.strip()
        return None

    def _extract_error(self, root: ET.Element) -> str | None:
        elem = self._find_first(root, "Error")
        return elem.text.strip() if elem is not None and elem.text else None

    def _extract_callsign_data(self, root: ET.Element) -> dict[str, Any] | None:
        callsign_elem = self._find_first(root, "Callsign")
        if callsign_elem is None:
            return None
        data: dict[str, Any] = {}
        for child in callsign_elem:
            key = _strip(child.tag)
            if child.text:
                data[key] = child.text.strip()
        return data if data.get("call") else None

    def _find_first(self, root: ET.Element, tag_name: str) -> ET.Element | None:
        lower = tag_name.lower()
        for elem in root.iter():
            if _strip(elem.tag).lower() == lower:
                return elem
        return None


_qrz_client: QRZClient | None = None


def get_qrz_client() -> QRZClient | None:
    """Return a shared QRZ client if credentials are configured."""

    global _qrz_client
    settings = get_settings().qrz
    if not (
        settings.enabled
        and settings.username
        and settings.password
    ):
        return None

    if _qrz_client is None:
        _qrz_client = QRZClient()
    return _qrz_client
