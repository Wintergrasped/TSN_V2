"""Archive compaction task for TSN nodes."""

import asyncio
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from tsn_common.config import NodeSettings
from tsn_common.logging import get_logger

logger = get_logger(__name__)


class ArchiveManager:
    """Periodically converts archived WAV files to MP3 and bundles them."""

    def __init__(self, settings: NodeSettings):
        self.settings = settings
        self.archive_dir = settings.audio_archive_dir
        self.interval_hours = max(1, settings.archive_cleanup_interval_hours)
        self.min_age_seconds = max(0, settings.archive_cleanup_min_age_minutes * 60)

    async def run(self) -> None:
        if not self.settings.archive_cleanup_enabled:
            logger.info("archive_cleanup_disabled")
            return

        logger.info(
            "archive_manager_started",
            interval_hours=self.interval_hours,
            min_age_minutes=self.settings.archive_cleanup_min_age_minutes,
        )

        try:
            while True:
                start_ts = time.perf_counter()
                try:
                    await self.process_once()
                except asyncio.CancelledError:
                    logger.info("archive_manager_cancelled")
                    raise
                except Exception as exc:
                    logger.error(
                        "archive_manager_iteration_failed",
                        error=str(exc),
                        exc_info=True,
                    )

                elapsed = time.perf_counter() - start_ts
                sleep_seconds = max(60.0, self.interval_hours * 3600 - elapsed)
                await asyncio.sleep(sleep_seconds)
        except asyncio.CancelledError:
            logger.info("archive_manager_stopped")

    async def process_once(self) -> None:
        wav_files = self._eligible_wav_files()
        if not wav_files:
            logger.debug("archive_manager_no_eligible_files")
            return

        zip_path = self._zip_path_for_today()
        processed = 0

        with zipfile.ZipFile(zip_path, mode="a", compression=zipfile.ZIP_DEFLATED) as zip_file:
            for wav_path in wav_files:
                success = await self._convert_and_package(wav_path, zip_file)
                if success:
                    processed += 1

        if processed:
            logger.info(
                "archive_manager_compacted_files",
                processed=processed,
                zip_path=str(zip_path),
            )

    def _eligible_wav_files(self) -> list[Path]:
        now = time.time()
        eligibles: list[Path] = []
        for wav_path in self.archive_dir.glob("*.wav"):
            try:
                age = now - wav_path.stat().st_mtime
            except FileNotFoundError:
                continue
            if age >= self.min_age_seconds:
                eligibles.append(wav_path)
        return eligibles

    def _zip_path_for_today(self) -> Path:
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        return self.archive_dir / f"{date_str}_archive.zip"

    async def _convert_and_package(self, wav_path: Path, zip_file: zipfile.ZipFile) -> bool:
        with tempfile.TemporaryDirectory(prefix="tsn-archive-") as tmpdir:
            mp3_path = Path(tmpdir) / f"{wav_path.stem}.mp3"
            converted = await self._convert_to_mp3(wav_path, mp3_path)
            if not converted:
                return False

            arcname = f"{wav_path.stem}_{datetime.now(timezone.utc).strftime('%H%M%S')}.mp3"
            try:
                zip_file.write(mp3_path, arcname=arcname)
            except Exception as exc:
                logger.error(
                    "archive_zip_failed",
                    filename=wav_path.name,
                    archive=str(zip_file.filename),
                    error=str(exc),
                    exc_info=True,
                )
                return False

        try:
            wav_path.unlink()
        except OSError as exc:
            logger.warning(
                "archive_wav_delete_failed",
                filename=wav_path.name,
                error=str(exc),
            )
        else:
            logger.info(
                "archive_file_compacted",
                filename=wav_path.name,
                archive=str(zip_file.filename),
                mp3_entry=arcname,
            )

        return True

    async def _convert_to_mp3(self, wav_path: Path, mp3_path: Path) -> bool:
        cmd = [
            self.settings.archive_ffmpeg_path,
            "-y",
            "-i",
            str(wav_path),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            f"{self.settings.archive_mp3_bitrate_kbps}k",
            str(mp3_path),
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            logger.error(
                "archive_ffmpeg_failed",
                filename=wav_path.name,
                error_output=stderr.decode(errors="ignore"),
            )
            return False

        return True
