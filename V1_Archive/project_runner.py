#!/usr/bin/env python3
"""Unified runner/orchestrator for the repeater audio intelligence stack.

This script centralizes configuration (DB, AI, filesystem paths, scheduler cadence)
and can either:
  * start long-running services (watchers) plus periodic batch jobs, or
  * run any individual stage once for ad-hoc maintenance.

It is informed by the pipeline documented in PROJECT_SUMMARY.md and is meant to be
the single entry point for day-to-day operations.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shlex
import signal
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# ----------------------------
# Configuration dataclasses
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


@dataclass(frozen=True)
class DatabaseSettings:
    host: str = _env("DB_HOST", "127.0.0.1")
    port: int = int(_env("DB_PORT", "3306"))
    user: str = _env("DB_USER", "repeateruser")
    password: str = _env("DB_PASS", "changeme123")
    name: str = _env("DB_NAME", "repeater")


@dataclass(frozen=True)
class AISettings:
    base_url: str = _env("OPENAI_BASE_URL", "http://127.0.0.1:8001/v1")
    backup_url: str = _env("AI_BACKUP_BASE_URL", "https://api.openai.com/v1")
    model: str = _env("AI_MODEL", "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4")
    backup_model: str = _env("AI_BACKUP_MODEL", "gpt-4o-mini")
    api_key: str = _env("AI_API_KEY", "sk-no-auth-needed-for-local")


@dataclass(frozen=True)
class PathSettings:
    repo_root: Path = PROJECT_ROOT
    audio_incoming: Path = PROJECT_ROOT / "repeater_audio" / "incoming"
    audio_processing: Path = PROJECT_ROOT / "repeater_audio" / "processing"
    audio_processed: Path = PROJECT_ROOT / "repeater_audio" / "processed"
    audio_failed: Path = PROJECT_ROOT / "repeater_audio" / "failed"
    audio_archive: Path = PROJECT_ROOT / "repeater_audio" / "archive"
    audio_logs: Path = PROJECT_ROOT / "repeater" / "audio_logs"


@dataclass(frozen=True)
class SchedulerSettings:
    smoother_interval_sec: int = 45
    callsign_interval_sec: int = 90
    topic_interval_sec: int = 75
    analyzer_interval_sec: int = 150
    profiles_interval_sec: int = 3600
    archive_interval_sec: int = 900
    system_stats_interval_sec: int = 300


@dataclass(frozen=True)
class TranscribeSettings:
    backend: str = os.getenv("TRANSCRIBE_BACKEND", "faster-whisper")
    model: str = os.getenv("FASTER_WHISPER_MODEL", os.getenv("WHISPER_MODEL", "medium.en"))
    device: str = os.getenv("FASTER_WHISPER_DEVICE", "cuda")
    compute_type: str = os.getenv("FASTER_WHISPER_COMPUTE_TYPE", "float16")
    beam_size: int = int(os.getenv("FASTER_WHISPER_BEAM_SIZE", "5"))
    vad_filter: bool = os.getenv("FASTER_WHISPER_VAD", "1") == "1"
    temperature: float = float(os.getenv("FASTER_WHISPER_TEMPERATURE", "0.0"))
    whisper_cpp_bin: str = os.getenv("WHISPER_CPP_BIN", "")
    whisper_cpp_model: str = os.getenv("WHISPER_CPP_MODEL", "ggml-medium.en.bin")
    whisper_cpp_extra: str = os.getenv("WHISPER_CPP_EXTRA_ARGS", "--threads 8 --gpu 1 --no-timestamps")
    legacy_model: str = os.getenv("OPENAI_WHISPER_MODEL", os.getenv("WHISPER_MODEL", "medium.en"))
    legacy_threads: int = int(os.getenv("WHISPER_THREADS", "8"))
    language: str = os.getenv("WHISPER_LANGUAGE", "en")


@dataclass(frozen=True)
class ProjectConfig:
    database: DatabaseSettings = DatabaseSettings()
    ai: AISettings = AISettings()
    paths: PathSettings = PathSettings()
    scheduler: SchedulerSettings = SchedulerSettings()
    transcribe: TranscribeSettings = TranscribeSettings()

    def as_env(self) -> Dict[str, str]:
        env = {
            "DB_HOST": self.database.host,
            "DB_PORT": str(self.database.port),
            "DB_USER": self.database.user,
            "DB_PASS": self.database.password,
            "DB_NAME": self.database.name,
            "AI_BASE_URL": self.ai.base_url,
            "AI_BACKUP_BASE_URL": self.ai.backup_url,
            "AI_MODEL": self.ai.model,
            "AI_BACKUP_MODEL": self.ai.backup_model,
            "AI_API_KEY": self.ai.api_key,
            "AUDIO_INCOMING_DIR": str(self.paths.audio_incoming),
            "AUDIO_PROCESSING_DIR": str(self.paths.audio_processing),
            "AUDIO_PROCESSED_DIR": str(self.paths.audio_processed),
            "AUDIO_FAILED_DIR": str(self.paths.audio_failed),
            "AUDIO_ARCHIVE_DIR": str(self.paths.audio_archive),
            "AUDIO_LOG_ROOT": str(self.paths.audio_logs),
            "TRANSCRIBE_BACKEND": self.transcribe.backend,
            "FASTER_WHISPER_MODEL": self.transcribe.model,
            "FASTER_WHISPER_DEVICE": self.transcribe.device,
            "FASTER_WHISPER_COMPUTE_TYPE": self.transcribe.compute_type,
            "FASTER_WHISPER_BEAM_SIZE": str(self.transcribe.beam_size),
            "FASTER_WHISPER_VAD": "1" if self.transcribe.vad_filter else "0",
            "FASTER_WHISPER_TEMPERATURE": str(self.transcribe.temperature),
            "WHISPER_CPP_BIN": self.transcribe.whisper_cpp_bin,
            "WHISPER_CPP_MODEL": self.transcribe.whisper_cpp_model,
            "WHISPER_CPP_EXTRA_ARGS": self.transcribe.whisper_cpp_extra,
            "OPENAI_WHISPER_MODEL": self.transcribe.legacy_model,
            "WHISPER_THREADS": str(self.transcribe.legacy_threads),
            "WHISPER_LANGUAGE": self.transcribe.language,
        }
        return env


# ----------------------------
# Stage definitions
# ----------------------------
@dataclass
class StageDefinition:
    key: str
    script: Path
    description: str
    long_running: bool = False
    interval_sec: Optional[int] = None
    args: List[str] = field(default_factory=list)
    env_overrides: Dict[str, str] = field(default_factory=dict)
    cwd: Optional[Path] = None
    skip_when_audio_pending: bool = False

    @property
    def exists(self) -> bool:
        return self.script.exists()


def build_stage_catalog(cfg: ProjectConfig) -> Dict[str, StageDefinition]:
    return {
        "transcribe_watcher": StageDefinition(
            key="transcribe_watcher",
            script=PROJECT_ROOT / "transcribe_watcher.py",
            description="Watch incoming WAV files and dispatch Whisper",
            long_running=True,
        ),
        "smooth_transcripts": StageDefinition(
            key="smooth_transcripts",
            script=PROJECT_ROOT / "ai_smoother.py",
            description="LLM smoothing + structured callsigns",
            interval_sec=cfg.scheduler.smoother_interval_sec,
            skip_when_audio_pending=True,
        ),
        "callsign_extractor": StageDefinition(
            key="callsign_extractor",
            script=PROJECT_ROOT / "callsign_extractor.py",
            description="Regex + QRZ validation for callsigns",
            interval_sec=cfg.scheduler.callsign_interval_sec,
            skip_when_audio_pending=True,
        ),
        "topic_extractor": StageDefinition(
            key="topic_extractor",
            script=PROJECT_ROOT / "topic_extractor.py",
            description="LLM topic tagging per callsign",
            interval_sec=cfg.scheduler.topic_interval_sec,
            skip_when_audio_pending=True,
        ),
        "net_analyzer": StageDefinition(
            key="net_analyzer",
            script=PROJECT_ROOT / "Transcript_Analyzer.py",
            description="Sessionize transcripts and persist nets",
            interval_sec=cfg.scheduler.analyzer_interval_sec,
            skip_when_audio_pending=True,
        ),
        "profile_loader": StageDefinition(
            key="profile_loader",
            script=PROJECT_ROOT / "extended_profiles_loader.py",
            description="Extended callsign/net/NCS profiling",
            interval_sec=cfg.scheduler.profiles_interval_sec,
        ),
        "audio_archiver": StageDefinition(
            key="audio_archiver",
            script=PROJECT_ROOT / "convert_and_archive.py",
            description="Convert WAV→MP3 and compress archives",
            interval_sec=cfg.scheduler.archive_interval_sec,
        ),
        "system_stats": StageDefinition(
            key="system_stats",
            script=PROJECT_ROOT / "system_status.py",
            description="Push local CPU/memory/temp metrics",
            interval_sec=cfg.scheduler.system_stats_interval_sec,
        ),
    }


# ----------------------------
# Process helpers
# ----------------------------
async def run_stage_once(stage: StageDefinition, config: ProjectConfig, dry_run: bool = False) -> int:
    if not stage.exists:
        logging.error("Stage %s missing script at %s", stage.key, stage.script)
        return 127

    cmd = [sys.executable, str(stage.script)] + stage.args
    env = os.environ.copy()
    env.update(config.as_env())
    env.update(stage.env_overrides)
    cwd = str(stage.cwd or stage.script.parent)

    logging.info("Running %s -> %s", stage.key, shlex.join(cmd))
    if dry_run:
        return 0

    proc = await asyncio.create_subprocess_exec(*cmd, cwd=cwd, env=env)
    return await proc.wait()


def has_pending_audio(config: ProjectConfig) -> bool:
    incoming = config.paths.audio_incoming
    if not incoming.exists():
        return False
    try:
        return any(p.suffix.lower() == ".wav" for p in incoming.iterdir())
    except Exception:
        return False


async def supervise_long_running(stage: StageDefinition, config: ProjectConfig, dry_run: bool = False):
    while True:
        code = await run_stage_once(stage, config, dry_run=dry_run)
        if dry_run:
            return
        if code == 0:
            logging.info("Stage %s exited cleanly; restarting in 3s", stage.key)
        else:
            logging.warning("Stage %s exited with code %s; restarting in 5s", stage.key, code)
        await asyncio.sleep(5 if code else 3)


async def schedule_periodic(stage: StageDefinition, config: ProjectConfig, dry_run: bool = False):
    interval = stage.interval_sec or 60
    while True:
        if stage.skip_when_audio_pending and has_pending_audio(config):
            logging.info("Skipping %s because incoming WAV queue is not empty", stage.key)
            await asyncio.sleep(min(interval, 10))
            continue
        code = await run_stage_once(stage, config, dry_run=dry_run)
        if dry_run:
            return
        if code != 0:
            logging.warning("Stage %s returned code %s", stage.key, code)
        await asyncio.sleep(interval)


async def run_daemon(config: ProjectConfig, selected: Iterable[str], dry_run: bool = False):
    catalog = build_stage_catalog(config)
    tasks = []
    for key in selected:
        stage = catalog.get(key)
        if not stage:
            logging.error("Unknown stage %s", key)
            continue
        if stage.long_running:
            tasks.append(asyncio.create_task(supervise_long_running(stage, config, dry_run)))
        else:
            tasks.append(asyncio.create_task(schedule_periodic(stage, config, dry_run)))
    if not tasks:
        logging.error("No stages selected for daemon mode")
        return
    await wait_for_tasks(tasks)


async def run_once(config: ProjectConfig, selected: Iterable[str], dry_run: bool = False):
    catalog = build_stage_catalog(config)
    for key in selected:
        stage = catalog.get(key)
        if not stage:
            logging.error("Unknown stage %s", key)
            continue
        code = await run_stage_once(stage, config, dry_run)
        if code != 0:
            logging.warning("Stage %s exited with code %s", key, code)


async def wait_for_tasks(tasks: List[asyncio.Task]):
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        raise


# ----------------------------
# CLI helpers
# ----------------------------
def print_stage_table(config: ProjectConfig):
    catalog = build_stage_catalog(config)
    rows = []
    for key, stage in catalog.items():
        mode = "watch" if stage.long_running else f"every {stage.interval_sec or 'n/a'}s"
        rows.append((key, mode, stage.script.relative_to(PROJECT_ROOT)))
    table = [(k, m, str(p)) for k, m, p in rows]
    header = ("Stage", "Mode", "Script")
    widths = [
        max(len(header[i]), *(len(str(row[i])) for row in table))
        for i in range(3)
    ]
    print(f"{header[0]:<{widths[0]}}  {header[1]:<{widths[1]}}  {header[2]}")
    print("-" * (sum(widths) + 4))
    for (stage, mode, script) in table:
        print(f"{stage:<{widths[0]}}  {mode:<{widths[1]}}  {script}")


def print_config(config: ProjectConfig):
    serializable = json.dumps({
        "database": asdict(config.database),
        "ai": asdict(config.ai),
        "paths": {k: str(v) for k, v in asdict(config.paths).items()},
        "scheduler": asdict(config.scheduler),
        "transcribe": asdict(config.transcribe),
    }, indent=2)
    print(serializable)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified project runner")
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=["run", "once", "list", "config"],
        help="Action to perform (default: run)",
    )
    parser.add_argument("stages", nargs="*", help="Subset of stages to target (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--log-level", default="INFO", help="Python logging level (default: INFO)")
    return parser.parse_args()


def resolve_stage_selection(config: ProjectConfig, requested: Iterable[str]) -> List[str]:
    catalog = build_stage_catalog(config)
    if not requested:
        return list(catalog.keys())
    normalized = []
    for name in requested:
        if name not in catalog:
            raise ValueError(f"Unknown stage '{name}'")
        normalized.append(name)
    return normalized


def configure_logging(level: str):
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    args = parse_args()
    configure_logging(args.log_level)
    config = ProjectConfig()

    if args.command == "list":
        print_stage_table(config)
        return
    if args.command == "config":
        print_config(config)
        return

    try:
        stages = resolve_stage_selection(config, args.stages)
    except ValueError as exc:
        logging.error(exc)
        sys.exit(2)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        if args.command == "run":
            loop.run_until_complete(run_daemon(config, stages, dry_run=args.dry_run))
        elif args.command == "once":
            loop.run_until_complete(run_once(config, stages, dry_run=args.dry_run))
    except KeyboardInterrupt:
        logging.info("Shutdown requested by user")
    finally:
        loop.stop()
        loop.close()


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]
    # allow Ctrl+C to propagate
    signal.signal(signal.SIGINT, signal.default_int_handler)
    main()
