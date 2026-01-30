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
    """
    
    def __init__(self):
        self._transcription_lock = asyncio.Lock()
        self._vllm_lock = asyncio.Lock()
        self._transcription_active = False
        self._last_ingestion_time: Optional[datetime] = None
        self._ingestion_cooldown_minutes = 3
        
        logger.info("resource_lock_initialized")
    
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
        - Transcription is active
        - Within cooldown period after file ingestion
        """
        while True:
            # Check if transcription is active
            if self._transcription_active:
                logger.debug("vllm_waiting_for_transcription")
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
        if self._transcription_active:
            return True
        
        return self.get_ingestion_cooldown_remaining() > 0


# Global singleton instance
_resource_lock: Optional[ResourceLock] = None


def get_resource_lock() -> ResourceLock:
    """Get global resource lock singleton."""
    global _resource_lock
    if _resource_lock is None:
        _resource_lock = ResourceLock()
    return _resource_lock
