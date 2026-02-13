"""
Transcription pipeline - processes audio files with Whisper.
"""

import asyncio
import contextlib
import os
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tsn_common.config import TranscriptionSettings, get_settings
from tsn_common.db import get_session
from tsn_common.logging import get_logger
from tsn_common.models import (
    AudioFile,
    AudioFileState,
    Transcription,
    TranscriptionBackend,
)
from tsn_common.resource_lock import get_resource_lock
from tsn_server.storage_guard import StorageGuard

logger = get_logger(__name__)

CPU_SAFE_COMPUTE_TYPES = {"int8", "int8_float16", "int16"}
DEFAULT_CPU_COMPUTE_TYPE = "int8"


class TranscriptionPipeline:
    """
    Transcription pipeline using faster-whisper.
    
    Processes audio files from the queue and stores transcripts.
    """

    def __init__(
        self,
        settings: TranscriptionSettings,
        storage_dir: Path,
        archive_dirs: Iterable[Path] | None = None,
        storage_guard: StorageGuard | None = None,
    ):
        # Note: CUDA_VISIBLE_DEVICES must be set in tsn_orchestrator.py BEFORE imports
        # Setting it here is too late - CUDA context already initialized
        
        self.settings = settings
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dirs: tuple[Path, ...] = tuple(Path(p) for p in (archive_dirs or ()))
        self.storage_guard = storage_guard
        self.model = None
        self.model_device: Optional[str] = None
        self.model_compute_type: Optional[str] = None
        self._missing_file_streak = 0
        self._missing_files_blocked_until: Optional[datetime] = None
        
        logger.info(
            "transcription_pipeline_initialized",
            backend=settings.backend,
            model=settings.model,
            device=settings.device,
            cuda_device=settings.cuda_device,
            archive_dir_count=len(self.archive_dirs),
            storage_guard_enabled=bool(storage_guard),
        )

    def _cuda_available(self) -> bool:
        """Check whether CUDA is available in the current environment."""
        try:
            import torch

            if torch.cuda.is_available():
                return True
        except Exception as exc:  # pragma: no cover - torch not available in tests
            logger.debug("cuda_check_failed", error=str(exc))
        return False

    def _resolve_runtime(self) -> Tuple[str, str]:
        """Determine which device/compute type to use for Whisper."""
        desired_device = self.settings.device
        compute_type = self.settings.compute_type
        fallback_forced = False

        # When cuda_device is explicitly configured, trust the fallback chain
        # to handle GPU detection rather than pre-checking torch.cuda.is_available()
        if desired_device == "cuda":
            device = "cuda"
            # Fallback chain in _load_model() will handle if GPU actually unavailable
        elif desired_device == "auto":
            cuda_ok = self._cuda_available()
            if cuda_ok:
                device = "cuda"
            else:
                device = "cpu"
                fallback_forced = True
        else:
            device = desired_device

        if device == "cpu" and compute_type not in CPU_SAFE_COMPUTE_TYPES:
            logger.info(
                "compute_type_adjusted_for_cpu",
                requested=compute_type,
                fallback=DEFAULT_CPU_COMPUTE_TYPE,
            )
            compute_type = DEFAULT_CPU_COMPUTE_TYPE

        if fallback_forced and device == "cpu" and not self.settings.allow_cpu_fallback:
            raise RuntimeError(
                "cpu_fallback_disabled_no_cuda_available",
            )

        return device, compute_type

    def _load_model(self) -> None:
        """Load Whisper model (lazy initialization)."""
        if self.model is not None:
            return
        
        if self.settings.backend == "faster-whisper":
            from faster_whisper import WhisperModel

            if self.settings.hf_cache_dir:
                cache_dir = self.settings.hf_cache_dir
                cache_dir.mkdir(parents=True, exist_ok=True)
                os.environ.setdefault("HF_HOME", str(cache_dir))
                os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_dir / "hub"))

            # Set HuggingFace token if provided for faster downloads
            if self.settings.hf_token:
                os.environ["HF_TOKEN"] = self.settings.hf_token.get_secret_value()
                logger.debug("huggingface_token_configured")

            device, compute_type = self._resolve_runtime()

            # Fallback chain: Try GPU devices in order, then OpenAI, then CPU
            attempted_devices = []
            
            try:
                self.model = WhisperModel(
                    self.settings.model,
                    device=device,
                    compute_type=compute_type,
                    device_index=0,  # Always 0 since CUDA_VISIBLE_DEVICES remaps
                )
                self.model_device = device
                self.model_compute_type = compute_type
                logger.info(
                    "whisper_model_loaded",
                    device=device,
                    compute_type=compute_type,
                    model=self.settings.model,
                )
            except RuntimeError as exc:
                error_msg = str(exc).lower()
                is_cuda_error = device == "cuda" and ("libcublas" in error_msg or "cuda" in error_msg or "out of memory" in error_msg)
                
                if is_cuda_error:
                    current_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "unknown")
                    attempted_devices.append(f"GPU{current_gpu}")
                    logger.warning(
                        "cuda_device_failed_trying_fallback",
                        failed_device=current_gpu,
                        error=str(exc),
                    )
                    
                    # Try alternate GPU (0 if we tried 1, or 1 if we tried 0)
                    alternate_gpu = "0" if current_gpu == "1" else "1"
                    try:
                        os.environ["CUDA_VISIBLE_DEVICES"] = alternate_gpu
                        logger.info("trying_alternate_gpu", gpu=alternate_gpu)
                        
                        self.model = WhisperModel(
                            self.settings.model,
                            device="cuda",
                            compute_type=compute_type,
                            device_index=0,
                        )
                        self.model_device = "cuda"
                        self.model_compute_type = compute_type
                        logger.info(
                            "whisper_model_loaded_alternate_gpu",
                            device="cuda",
                            gpu=alternate_gpu,
                            compute_type=compute_type,
                            model=self.settings.model,
                        )
                        return  # Success!
                    except RuntimeError as alt_exc:
                        attempted_devices.append(f"GPU{alternate_gpu}")
                        logger.warning(
                            "alternate_gpu_failed",
                            gpu=alternate_gpu,
                            error=str(alt_exc),
                        )
                    
                    # Try OpenAI fallback if enabled
                    if self.settings.openai_fallback_enabled and self.settings.openai_api_key:
                        attempted_devices.append("OpenAI_API")
                        logger.warning(
                            "all_gpus_failed_using_openai_api",
                            attempted=attempted_devices,
                        )
                        # Don't load local model - let transcribe_file use OpenAI
                        self.model = None
                        self.model_device = "openai"
                        self.model_compute_type = "api"
                        return
                    
                    # Last resort: Try CPU fallback if enabled
                    if self.settings.allow_cpu_fallback:
                        attempted_devices.append("CPU")
                        logger.warning(
                            "all_gpus_failed_falling_back_to_cpu",
                            attempted=attempted_devices,
                        )
                        device = "cpu"
                        compute_type = DEFAULT_CPU_COMPUTE_TYPE
                        self.model = WhisperModel(
                            self.settings.model,
                            device=device,
                            compute_type=compute_type,
                        )
                        self.model_device = device
                        self.model_compute_type = compute_type
                        return
                    
                    # No fallback available
                    logger.error(
                        "all_devices_failed_no_fallback_available",
                        attempted=attempted_devices,
                        error=str(exc),
                    )
                
                raise

            logger.info(
                "whisper_model_loaded",
                backend="faster-whisper",
                model=self.settings.model,
                device=self.model_device,
                compute_type=self.model_compute_type,
            )
        else:
            raise NotImplementedError(f"Backend {self.settings.backend} not implemented")

    def _unload_model(self) -> None:
        """Unload Whisper model to free GPU memory for vLLM."""
        if self.model is None:
            return
        
        logger.info(
            "whisper_model_unloading",
            device=self.model_device,
            model=self.settings.model,
        )
        
        # Delete model and force garbage collection
        del self.model
        self.model = None
        
        # Force CUDA memory cleanup if we were using GPU
        if self.model_device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("cuda_memory_freed")
            except Exception as e:
                logger.warning("cuda_cleanup_failed", error=str(e))
        
        logger.info("whisper_model_unloaded")

    async def transcribe_file(
        self,
        audio_file: AudioFile,
    ) -> Optional[Transcription]:
        """
        Transcribe an audio file.
        
        Args:
            audio_file: AudioFile database record
            
        Returns:
            Transcription record if successful, None otherwise
        """
        file_path = self.storage_dir / audio_file.filename
        
        if not file_path.exists():
            restored_file = self._restore_from_archive(audio_file.filename)
            if restored_file and restored_file.exists():
                file_path = restored_file
            else:
                logger.error(
                    "transcribe_file_not_found",
                    audio_file_id=str(audio_file.id),
                    filename=audio_file.filename,
                    tried_archive=bool(self.archive_dirs),
                )
                self._note_missing_file()
                return None

        self._reset_missing_file_streak()
        
        try:
            logger.info(
                "transcription_starting",
                audio_file_id=str(audio_file.id),
                filename=audio_file.filename,
            )
            
            start_time = time.time()
            transcript_text = None
            backend_used = TranscriptionBackend.FASTER_WHISPER
            
            # Check if we're in OpenAI-only mode (GPU failed during init)
            if self.model_device == "openai":
                transcript_text = await self._transcribe_openai(file_path)
                backend_used = TranscriptionBackend.OPENAI
            else:
                # Try local GPU/CPU transcription first
                try:
                    # Load model if needed
                    self._load_model()
                    
                    # Run transcription in executor (blocking I/O)
                    loop = asyncio.get_event_loop()
                    transcript_text = await loop.run_in_executor(
                        None,
                        self._transcribe_sync,
                        str(file_path),
                    )
                except RuntimeError as gpu_error:
                    error_msg = str(gpu_error).lower()
                    is_gpu_error = "cuda" in error_msg or "out of memory" in error_msg or "gpu" in error_msg
                    
                    # Try OpenAI fallback if GPU failed during transcription
                    if is_gpu_error and self.settings.openai_fallback_enabled and self.settings.openai_api_key:
                        logger.warning(
                            "gpu_transcription_failed_trying_openai_fallback",
                            audio_file_id=str(audio_file.id),
                            error=str(gpu_error),
                        )
                        transcript_text = await self._transcribe_openai(file_path)
                        backend_used = TranscriptionBackend.OPENAI
                    # Try CPU fallback if OpenAI not available but CPU fallback enabled
                    elif is_gpu_error and self.settings.allow_cpu_fallback:
                        logger.warning(
                            "gpu_transcription_failed_trying_cpu_fallback",
                            audio_file_id=str(audio_file.id),
                            error=str(gpu_error),
                        )
                        # Force reload model on CPU
                        self._unload_model()
                        self.settings.device = "cpu"
                        self._load_model()
                        loop = asyncio.get_event_loop()
                        transcript_text = await loop.run_in_executor(
                            None,
                            self._transcribe_sync,
                            str(file_path),
                        )
                    else:
                        raise
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Create transcription record
            transcription = Transcription(
                audio_file_id=audio_file.id,
                transcript_text=transcript_text,
                language=self.settings.language,
                backend=backend_used,
                processing_time_ms=processing_time_ms,
                word_count=len(transcript_text.split()) if transcript_text else 0,
                char_count=len(transcript_text) if transcript_text else 0,
            )
            
            logger.info(
                "transcription_completed",
                audio_file_id=str(audio_file.id),
                filename=audio_file.filename,
                processing_time_ms=processing_time_ms,
                word_count=transcription.word_count,
                char_count=transcription.char_count,
            )
            
            # Unload model to free GPU memory for vLLM
            self._unload_model()
            
            return transcription
            
        except Exception as e:
            logger.error(
                "transcription_failed",
                audio_file_id=str(audio_file.id),
                filename=audio_file.filename,
                error=str(e),
                exc_info=True,
            )
            # Unload model even on failure
            self._unload_model()
            return None

    def _restore_from_archive(self, filename: str) -> Optional[Path]:
        if not self.archive_dirs:
            return None

        target_path = self.storage_dir / filename
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error(
                "archive_restore_target_unavailable",
                filename=filename,
                target_dir=str(target_path.parent),
                error=str(exc),
            )
            if self.storage_guard:
                self.storage_guard.mark_unavailable(
                    source="transcriber_restore_target",
                    error=str(exc),
                )
            return None

        for archive_dir in self.archive_dirs:
            try:
                archive_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                # Directory may be read-only; skip creation attempt failures silently
                pass

            candidate = archive_dir / filename
            if not candidate.exists():
                continue

            if target_path.exists():
                return target_path

            try:
                shutil.copy2(candidate, target_path)
            except Exception as exc:
                logger.error(
                    "archive_restore_copy_failed",
                    filename=filename,
                    source=str(candidate),
                    destination=str(target_path),
                    error=str(exc),
                )
                if self.storage_guard:
                    self.storage_guard.mark_unavailable(
                        source="transcriber_restore_copy",
                        error=str(exc),
                    )
                continue

            logger.warning(
                "transcription_file_restored_from_archive",
                filename=filename,
                source=str(candidate),
                destination=str(target_path),
            )
            return target_path

        return None

    def _note_missing_file(self) -> None:
        self._missing_file_streak += 1
        threshold = max(1, self.settings.missing_file_error_threshold)
        if self._missing_file_streak >= threshold:
            backoff = max(10, self.settings.missing_file_backoff_sec)
            self._missing_files_blocked_until = datetime.now(timezone.utc) + timedelta(seconds=backoff)
            logger.error(
                "transcription_missing_files_backoff",
                streak=self._missing_file_streak,
                backoff_seconds=backoff,
                storage_dir=str(self.storage_dir),
            )

    def _reset_missing_file_streak(self) -> None:
        if self._missing_file_streak:
            logger.info(
                "transcription_missing_files_recovered",
                streak=self._missing_file_streak,
            )
        self._missing_file_streak = 0
        self._missing_files_blocked_until = None

    def _missing_file_backoff_remaining(self) -> int:
        if not self._missing_files_blocked_until:
            return 0
        now = datetime.now(timezone.utc)
        if now >= self._missing_files_blocked_until:
            self._missing_files_blocked_until = None
            return 0
        return int((self._missing_files_blocked_until - now).total_seconds())

    def _transcribe_sync(self, file_path: str) -> str:
        """
        Synchronous transcription (runs in executor).
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Transcript text
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        segments, _ = self.model.transcribe(
            file_path,
            beam_size=self.settings.beam_size,
            vad_filter=self.settings.vad_filter,
            language=self.settings.language,
            temperature=self.settings.temperature,
            condition_on_previous_text=False,
        )
        
        # Collect all segments
        text = " ".join(seg.text.strip() for seg in segments).strip()
        
        return text

    async def _transcribe_openai(self, file_path: Path) -> str:
        """
        Transcribe using OpenAI API as fallback.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Transcript text
        """
        if not self.settings.openai_api_key:
            raise RuntimeError("OpenAI API key not configured")
        
        try:
            import httpx
            
            logger.info(
                "transcription_using_openai_fallback",
                filename=file_path.name,
                model=self.settings.openai_model,
            )
            
            # OpenAI Whisper API expects the file as multipart/form-data
            with open(file_path, "rb") as audio_file:
                files = {
                    "file": (file_path.name, audio_file, "audio/wav"),
                }
                data = {
                    "model": self.settings.openai_model,
                    "language": self.settings.language,
                }
                headers = {
                    "Authorization": f"Bearer {self.settings.openai_api_key.get_secret_value()}",
                }
                
                async with httpx.AsyncClient(timeout=self.settings.timeout_sec) as client:
                    response = await client.post(
                        "https://api.openai.com/v1/audio/transcriptions",
                        files=files,
                        data=data,
                        headers=headers,
                    )
                    response.raise_for_status()
                    result = response.json()
                    
                    transcript_text = result.get("text", "").strip()
                    
                    logger.info(
                        "openai_transcription_completed",
                        filename=file_path.name,
                        word_count=len(transcript_text.split()) if transcript_text else 0,
                    )
                    
                    return transcript_text
                    
        except Exception as e:
            logger.error(
                "openai_transcription_failed",
                filename=file_path.name,
                error=str(e),
                exc_info=True,
            )
            raise

    async def get_next_file(
        self,
        session: AsyncSession,
    ) -> Optional[AudioFile]:
        """
        Get next file to transcribe from queue.
        
        Uses SELECT FOR UPDATE SKIP LOCKED for concurrent workers.
        
        Args:
            session: Database session
            
        Returns:
            AudioFile to process, or None if queue empty
        """
        result = await session.execute(
            select(AudioFile)
            .where(AudioFile.state == AudioFileState.QUEUED_TRANSCRIPTION)
            .order_by(AudioFile.created_at.desc())  # Newest first - prioritize live files
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        
        return result.scalar_one_or_none()

    async def process_one(self) -> bool:
        """
        Process one file from the queue.
        
        Returns:
            True if file was processed, False if queue empty
        """
        # Acquire transcription lock (blocks vLLM)
        resource_lock = get_resource_lock()
        await resource_lock.acquire_transcription()
        
        try:
            # Get next file and extract needed data within transaction
            async with get_session() as session:
                audio_file = await self.get_next_file(session)
                
                if audio_file is None:
                    return False
                
                # Update state and extract data before commit
                audio_file.state = AudioFileState.TRANSCRIBING
                await session.flush()
                
                # Extract data while still in session
                audio_file_id = audio_file.id
                audio_filename = audio_file.filename
                # Session commits here, releasing locks
            
            # Transcribe (outside transaction for long-running operation)
            # Create temporary object with just the data we need
            from types import SimpleNamespace
            audio_file_data = SimpleNamespace(id=audio_file_id, filename=audio_filename)
            transcription = await self.transcribe_file(audio_file_data)
            
            # Update database with retry logic for lock timeouts
            max_retries = 3
            retry_delay = 1.0
            
            for attempt in range(max_retries):
                try:
                    async with get_session() as session:
                        # Re-fetch audio file
                        audio_file = await session.get(AudioFile, audio_file_id)
                        
                        if audio_file is None:
                            logger.error("audio_file_disappeared", audio_file_id=str(audio_file_id))
                            return True
                        
                        if transcription:
                            # Save transcription
                            session.add(transcription)
                            
                            # Update state
                            audio_file.state = AudioFileState.TRANSCRIBED
                            audio_file.state = AudioFileState.QUEUED_EXTRACTION
                            
                            logger.info(
                                "file_transcribed_queued_extraction",
                                audio_file_id=str(audio_file.id),
                                transcription_id=str(transcription.id),
                            )
                        else:
                            # Mark as failed
                            audio_file.state = AudioFileState.FAILED_TRANSCRIPTION
                            audio_file.retry_count += 1
                            
                            logger.error(
                                "file_transcription_failed",
                                audio_file_id=str(audio_file.id),
                                retry_count=audio_file.retry_count,
                            )
                    
                    # Success - break retry loop
                    break
                    
                except Exception as e:
                    # Check if it's a database lock timeout
                    error_msg = str(e)
                    is_lock_timeout = "Lock wait timeout" in error_msg or "1205" in error_msg
                    
                    if is_lock_timeout and attempt < max_retries - 1:
                        # Retry with exponential backoff
                        wait_time = retry_delay * (2 ** attempt)
                        logger.warning(
                            "database_lock_timeout_retry",
                            audio_file_id=str(audio_file_id),
                            attempt=attempt + 1,
                            wait_time=wait_time,
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        # Give up or different error
                        logger.error(
                            "database_update_failed",
                            audio_file_id=str(audio_file_id),
                            error=error_msg,
                            is_lock_timeout=is_lock_timeout,
                            exc_info=True,
                        )
                        # Don't return False - we did process it, just couldn't update DB
                        break
            
            return True
            
        finally:
            # Always release lock
            resource_lock.release_transcription()

    async def run_worker(self, worker_id: int = 0) -> None:
        """
        Run transcription worker loop.
        
        Args:
            worker_id: Worker identifier for logging
        """
        logger.info("transcription_worker_started", worker_id=worker_id)
        
        while True:
            try:
                if self.storage_guard and not self.storage_guard.is_available():
                    await self.storage_guard.wait_until_available(f"transcription_worker_{worker_id}")
                    continue

                backoff_remaining = self._missing_file_backoff_remaining()
                if backoff_remaining > 0:
                    logger.warning(
                        "transcription_worker_backing_off_missing_files",
                        worker_id=worker_id,
                        remaining_seconds=backoff_remaining,
                        storage_dir=str(self.storage_dir),
                    )
                    await asyncio.sleep(min(backoff_remaining, 5))
                    continue

                processed = await self.process_one()
                
                if not processed:
                    # Queue empty, wait before polling again
                    await asyncio.sleep(1.0)
                    
            except asyncio.CancelledError:
                logger.info("transcription_worker_cancelled", worker_id=worker_id)
                break
            except Exception as e:
                error_msg = str(e)
                is_lock_timeout = "Lock wait timeout" in error_msg or "1205" in error_msg
                
                logger.error(
                    "transcription_worker_error",
                    worker_id=worker_id,
                    error=error_msg,
                    is_lock_timeout=is_lock_timeout,
                    exc_info=True,
                )
                
                # Longer backoff for lock timeouts to reduce contention
                await asyncio.sleep(10.0 if is_lock_timeout else 5.0)


async def main() -> None:
    """Main entry point for transcription pipeline."""
    import sqlalchemy.exc
    from tsn_common import setup_logging
    
    settings = get_settings()
    setup_logging(settings.logging)
    
    storage_dir = settings.storage.base_path
    storage_guard = StorageGuard(settings.storage)
    guard_task: Optional[asyncio.Task] = None
    if storage_guard.enabled:
        guard_task = asyncio.create_task(storage_guard.run())
    
    # Create pipeline
    pipeline = TranscriptionPipeline(
        settings.transcription,
        storage_dir,
        archive_dirs=settings.storage.archive_dirs,
        storage_guard=storage_guard,
    )
    
    # Run multiple workers
    workers = [
        asyncio.create_task(pipeline.run_worker(i))
        for i in range(settings.transcription.max_concurrent)
    ]
    
    # Wait for all workers
    await asyncio.gather(*workers)

    if guard_task:
        guard_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await guard_task


if __name__ == "__main__":
    asyncio.run(main())
