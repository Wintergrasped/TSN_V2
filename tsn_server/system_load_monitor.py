"""System load guard that pauses vLLM-heavy work when host is overwhelmed."""

from __future__ import annotations

import asyncio
from typing import Optional

import psutil

from tsn_common.config import SystemLoadSettings
from tsn_common.logging import get_logger
from tsn_common.resource_lock import get_resource_lock

logger = get_logger(__name__)


class SystemLoadMonitor:
    """Monitors CPU/RAM pressure and triggers system pauses via ResourceLock."""

    def __init__(self, settings: SystemLoadSettings):
        self.settings = settings
        self._resource_lock = get_resource_lock()
        self._breach_streak = 0
        self._last_high_log: Optional[str] = None

    async def run(self) -> None:
        if not self.settings.enabled:
            logger.info("system_load_monitor_disabled")
            return

        logger.info(
            "system_load_monitor_started",
            cpu_threshold=self.settings.cpu_percent_threshold,
            memory_threshold=self.settings.memory_percent_threshold,
            check_interval_sec=self.settings.check_interval_sec,
            pause_duration_sec=self.settings.pause_duration_sec,
        )

        # Prime psutil so first cpu_percent call has a baseline
        psutil.cpu_percent(interval=None)

        try:
            while True:
                try:
                    await self._check_once()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.error("system_load_monitor_error", error=str(exc), exc_info=True)

                await asyncio.sleep(max(1.0, self.settings.check_interval_sec))
        except asyncio.CancelledError:
            logger.info("system_load_monitor_cancelled")

    async def _check_once(self) -> None:
        cpu_percent = await asyncio.to_thread(psutil.cpu_percent, None)
        memory_percent = psutil.virtual_memory().percent

        high_cpu = cpu_percent >= self.settings.cpu_percent_threshold
        high_memory = memory_percent >= self.settings.memory_percent_threshold
        high_load = high_cpu or high_memory

        if high_load:
            self._breach_streak += 1
            if self._breach_streak == 1:
                self._last_high_log = (
                    f"cpu={cpu_percent:.1f}% mem={memory_percent:.1f}%"
                )
                logger.warning(
                    "system_load_high_sample",
                    cpu_percent=cpu_percent,
                    memory_percent=memory_percent,
                )

            if self._breach_streak >= self.settings.breach_samples_required:
                reason = f"cpu={cpu_percent:.1f}% mem={memory_percent:.1f}%"
                self._resource_lock.enter_system_pause(
                    duration_sec=self.settings.pause_duration_sec,
                    reason=reason,
                )
                # Keep streak pegged so continuing pressure keeps extending pause
                self._breach_streak = self.settings.breach_samples_required
        else:
            if self._breach_streak > 0:
                logger.info(
                    "system_load_recovered",
                    cpu_percent=cpu_percent,
                    memory_percent=memory_percent,
                    previous_high=self._last_high_log,
                    high_samples=self._breach_streak,
                )
            self._breach_streak = 0
            self._last_high_log = None
