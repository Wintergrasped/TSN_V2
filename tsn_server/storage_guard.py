"""Storage availability guard for server-side services."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from tsn_common.config import StorageSettings
from tsn_common.logging import get_logger
from tsn_common.resource_lock import get_resource_lock

logger = get_logger(__name__)


class StorageGuard:
    """Monitors the shared storage mount and coordinates backpressure when it fails."""

    def __init__(self, settings: StorageSettings):
        self.settings = settings
        self.base_path: Path = settings.base_path
        self.probe_path: Path = self.base_path / settings.health_probe_filename
        self.check_interval = max(1.0, float(settings.health_check_interval_sec))
        self.pause_duration = max(30, int(settings.health_missing_backoff_sec))
        self.enabled = bool(settings.health_check_enabled)
        self._available_event = asyncio.Event()
        self._available_event.set()
        self._last_wait_log: Optional[float] = None

    def is_available(self) -> bool:
        """Return True when storage is considered available."""
        return not self.enabled or self._available_event.is_set()

    def mark_unavailable(self, source: str, error: Optional[str] = None) -> None:
        """Force storage into unavailable state and pause heavy work."""
        if not self.enabled:
            return

        if self._available_event.is_set():
            self._available_event.clear()
            logger.error(
                "storage_unavailable_detected",
                base_path=str(self.base_path),
                source=source,
                error=error,
                pause_duration_sec=self.pause_duration,
            )
            get_resource_lock().enter_system_pause(
                self.pause_duration,
                reason="storage_unavailable",
            )

    def _mark_recovered(self, source: str) -> None:
        if not self.enabled:
            return

        if not self._available_event.is_set():
            self._available_event.set()
            logger.info(
                "storage_available_recovered",
                base_path=str(self.base_path),
                source=source,
            )

    async def wait_until_available(self, reason: str, poll_interval: float = 5.0) -> None:
        """Block the caller until storage becomes available again."""
        if self.is_available():
            return

        logger.warning(
            "storage_waiting_for_recovery",
            reason=reason,
            base_path=str(self.base_path),
        )

        while not self.is_available():
            await asyncio.sleep(max(1.0, poll_interval))

        logger.info("storage_processing_resumed", reason=reason)

    async def run(self) -> None:
        """Continuously probe the storage mount and update availability state."""
        if not self.enabled:
            logger.info("storage_guard_disabled")
            return

        logger.info(
            "storage_guard_started",
            base_path=str(self.base_path),
            interval_sec=self.check_interval,
            probe_path=str(self.probe_path),
        )

        try:
            while True:
                ok, error = await asyncio.to_thread(self._probe_once)
                if ok:
                    self._mark_recovered("health_probe")
                else:
                    self.mark_unavailable("health_probe", error=error)
                await asyncio.sleep(self.check_interval)
        except asyncio.CancelledError:
            logger.info("storage_guard_stopped")
            raise

    def _probe_once(self) -> tuple[bool, Optional[str]]:
        """Attempt to touch a sentinel file to verify the mount."""
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
            probe_path = self.probe_path
            if not probe_path.exists():
                probe_path.write_text("tsn-storage-probe", encoding="utf-8")
            else:
                probe_path.touch()
            return True, None
        except Exception as exc:  # pragma: no cover - depends on host FS
            return False, str(exc)