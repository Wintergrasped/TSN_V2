"""Resource locking for coordinating transcription and vLLM usage."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

from tsn_common.logging import get_logger

logger = get_logger(__name__)


class ResourceLock:
    """
    Global resource lock to prevent transcription and vLLM from running simultaneously.
    
    Priority: Transcription > vLLM
    
    When audio files are ingested:
    1. Pause all vLLM operations
    2. Wait a few seconds
    3. Run transcription pipeline
    4. Keep vLLM paused for 3 minutes after ingestion (to catch net starts)
    
    If separate GPUs are configured (TSN_WHISPER_CUDA_DEVICE set), transcription
    lock does NOT block vLLM operations (they can run simultaneously).
    """
    
    def __init__(self, separate_gpus: bool = False):
        self._transcription_lock = asyncio.Lock()
        self._vllm_lock = asyncio.Lock()
        self._transcription_active = False
        self._last_ingestion_time: Optional[datetime] = None
        self._ingestion_cooldown_minutes = 3
        self._system_pause_until: Optional[datetime] = None
        self._system_pause_reason: Optional[str] = None
        self._separate_gpus = separate_gpus
        
        logger.info(
            "resource_lock_initialized",
            separate_gpus=separate_gpus,
            transcription_blocks_vllm=not separate_gpus,
        )
    
    async def acquire_transcription(self) -> None:
        """
        Acquire transcription lock (highest priority).
        
        Blocks vLLM operations while held.
        """
        await self._transcription_lock.acquire()
        self._transcription_active = True
        
        logger.debug("transcription_lock_acquired")
    
    def release_transcription(self) -> None:
        """Release transcription lock."""
        self._transcription_active = False
        self._transcription_lock.release()
        
        logger.debug("transcription_lock_released")
    
    async def acquire_vllm(self) -> None:
        """
        Acquire vLLM lock (lower priority than transcription).
        
        Waits if:
        - Transcription is active (only if using same GPU)
        - Within cooldown period after file ingestion
        """
        while True:
            # Check if transcription is active (only block if sharing GPU)
            if not self._separate_gpus and self._transcription_active:
                logger.debug("vllm_waiting_for_transcription_same_gpu")
                await asyncio.sleep(1.0)
                continue
            
            # Check if within ingestion cooldown
            if self._last_ingestion_time:
                cooldown_end = self._last_ingestion_time + timedelta(
                    minutes=self._ingestion_cooldown_minutes
                )
                now = datetime.now(timezone.utc)
                
                if now < cooldown_end:
                    remaining_seconds = (cooldown_end - now).total_seconds()
                    logger.debug(
                        "vllm_waiting_for_ingestion_cooldown",
                        remaining_seconds=int(remaining_seconds),
                    )
                    await asyncio.sleep(5.0)
                    continue

            if self._system_pause_until:
                now = datetime.now(timezone.utc)
                if now < self._system_pause_until:
                    remaining = int((self._system_pause_until - now).total_seconds())
                    logger.debug(
                        "vllm_waiting_for_system_pause",
                        remaining_seconds=remaining,
                        reason=self._system_pause_reason,
                    )
                    await asyncio.sleep(2.0)
                    continue
                self._system_pause_until = None
                self._system_pause_reason = None
            
            # Safe to acquire
            break
        
        await self._vllm_lock.acquire()
        logger.debug("vllm_lock_acquired")
    
    def release_vllm(self) -> None:
        """Release vLLM lock."""
        self._vllm_lock.release()
        logger.debug("vllm_lock_released")
    
    def notify_ingestion(self) -> None:
        """
        Notify that audio files have been ingested.
        
        Triggers cooldown period to pause vLLM operations.
        """
        self._last_ingestion_time = datetime.now(timezone.utc)
        
        logger.info(
            "audio_ingestion_notified_vllm_paused",
            cooldown_minutes=self._ingestion_cooldown_minutes,
        )
    
    def get_ingestion_cooldown_remaining(self) -> int:
        """Get remaining cooldown seconds, or 0 if no cooldown active."""
        if not self._last_ingestion_time:
            return 0
        
        cooldown_end = self._last_ingestion_time + timedelta(
            minutes=self._ingestion_cooldown_minutes
        )
        now = datetime.now(timezone.utc)
        
        if now >= cooldown_end:
            return 0
        
        return int((cooldown_end - now).total_seconds())
    
    def is_vllm_blocked(self) -> bool:
        """Check if vLLM operations are currently blocked."""
        if not self._separate_gpus and self._transcription_active:
            return True
        
        if self.get_ingestion_cooldown_remaining() > 0:
            return True

        return self.get_system_pause_remaining() > 0

    def enter_system_pause(self, duration_sec: int, reason: str | None = None) -> None:
        """Pause vLLM operations for a duration when system load is high."""
        now = datetime.now(timezone.utc)
        pause_until = now + timedelta(seconds=max(1, duration_sec))
        if not self._system_pause_until or pause_until > self._system_pause_until:
            self._system_pause_until = pause_until
        self._system_pause_reason = reason

        logger.warning(
            "system_pause_engaged",
            duration_sec=duration_sec,
            reason=reason,
            pause_until=self._system_pause_until.isoformat() if self._system_pause_until else None,
        )

    def get_system_pause_remaining(self) -> int:
        if not self._system_pause_until:
            return 0

        now = datetime.now(timezone.utc)
        if now >= self._system_pause_until:
            self._system_pause_until = None
            self._system_pause_reason = None
            return 0

        return int((self._system_pause_until - now).total_seconds())

    def get_system_pause_reason(self) -> Optional[str]:
        if self.get_system_pause_remaining() <= 0:
            return None
        return self._system_pause_reason


# Global singleton instance
_resource_lock: Optional[ResourceLock] = None


def get_resource_lock() -> ResourceLock:
    """Get global resource lock singleton."""
    global _resource_lock
    if _resource_lock is None:
        # Auto-detect if separate GPUs are configured
        import os
        cuda_device = os.environ.get("TSN_WHISPER_CUDA_DEVICE")
        separate_gpus = cuda_device is not None and cuda_device != ""
        _resource_lock = ResourceLock(separate_gpus=separate_gpus)
    return _resource_lock
