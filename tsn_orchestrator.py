"""
Main orchestrator - launches all TSN services.
"""

from datetime import datetime, timezone
import asyncio
import signal
from pathlib import Path
from typing import List

from tsn_common.config import get_settings
from tsn_common.logging import get_logger, setup_logging

logger = get_logger(__name__)


class ServiceOrchestrator:
    """Manages lifecycle of all TSN services."""

    def __init__(self):
        self.settings = get_settings()
        self.tasks: List[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()
        self._purge_sentinel = self.settings.storage.base_path / ".net_history_purge"

    async def maybe_purge_net_history(self) -> None:
        """Optionally purge net history once per boot when configured."""

        if not self.settings.analysis.purge_net_history_on_boot:
            return

        sentinel: Path = self._purge_sentinel
        if sentinel.exists():
            logger.info(
                "net_history_purge_skipped",
                reason="sentinel_present",
                sentinel=str(sentinel),
            )
            return

        logger.warning("net_history_purge_requested")

        from scripts.reset_net_history import reset_net_history

        await reset_net_history()

        sentinel.write_text(
            f"purged at {datetime.now(timezone.utc).isoformat()}"
        )

        logger.info(
            "net_history_purge_complete",
            sentinel=str(sentinel),
        )

    async def start_node_services(self) -> None:
        """Start node-side services (watcher + transfer)."""
        if not self.settings.node.enabled:
            logger.info("node_services_disabled")
            return

        logger.info("starting_node_services")

        from tsn_node.transfer import TransferAgent, transfer_worker
        from tsn_node.watcher import FileWatcher

        # Create components
        watcher = FileWatcher(self.settings.node)

        # Start watcher
        watcher_task = asyncio.create_task(watcher.watch_loop())
        self.tasks.append(watcher_task)

        # Start transfer workers (pass watcher for file cleanup)
        for i in range(self.settings.node.transfer_workers):
            worker_task = asyncio.create_task(
                transfer_worker(watcher.transfer_queue, self.settings.node, watcher)
            )
            self.tasks.append(worker_task)

        logger.info(
            "node_services_started",
            transfer_workers=self.settings.node.transfer_workers,
        )

    async def start_server_services(self) -> None:
        """Start server-side services (ingestion, transcription, extraction, analysis)."""
        if not self.settings.server.enabled:
            logger.info("server_services_disabled")
            return

        logger.info("starting_server_services")

        from tsn_server.analyzer import TranscriptAnalyzer
        from tsn_server.extractor import CallsignExtractor
        from tsn_server.ingestion import IngestionService
        from tsn_server.transcriber import TranscriptionPipeline

        # Ingestion service
        ingestion = IngestionService(
            self.settings.server,
            self.settings.storage.base_path,
        )
        ingestion_task = asyncio.create_task(ingestion.run_loop())
        self.tasks.append(ingestion_task)

        # Transcription workers
        transcriber = TranscriptionPipeline(
            self.settings.transcription,
            self.settings.storage.base_path,
        )
        for i in range(self.settings.transcription.max_concurrent):
            worker_task = asyncio.create_task(transcriber.run_worker(i))
            self.tasks.append(worker_task)

        # Extraction workers
        extractor = CallsignExtractor(self.settings.vllm)
        for i in range(self.settings.vllm.max_concurrent):
            worker_task = asyncio.create_task(extractor.run_worker(i))
            self.tasks.append(worker_task)

        # Analysis workers - create separate instance per worker to avoid shared state
        for i in range(self.settings.analysis.worker_count):
            analyzer = TranscriptAnalyzer(self.settings.vllm, self.settings.analysis)
            worker_task = asyncio.create_task(analyzer.run_worker(i))
            self.tasks.append(worker_task)

        # Net auto-detection orchestrator (NEW)
        if self.settings.net_autodetect.enabled:
            from tsn_server.services.net_autodetect import NetAutoDetectOrchestrator
            
            net_autodetect = NetAutoDetectOrchestrator()
            autodetect_task = asyncio.create_task(net_autodetect.run())
            self.tasks.append(autodetect_task)
            
            logger.info("net_autodetect_orchestrator_started")

        # Duplicate detection service (NEW)
        from tsn_server.duplicate_detection import DuplicateDetectionService
        
        duplicate_detector = DuplicateDetectionService()
        duplicate_task = asyncio.create_task(duplicate_detector.run())
        self.tasks.append(duplicate_task)
        
        logger.info("duplicate_detection_service_started")

        # Stuck file recovery service (NEW)
        from tsn_server.stuck_file_recovery import StuckFileRecovery
        
        stuck_recovery = StuckFileRecovery()
        stuck_recovery_task = asyncio.create_task(stuck_recovery.run())
        self.tasks.append(stuck_recovery_task)
        
        logger.info("stuck_file_recovery_service_started")

        logger.info(
            "server_services_started",
            transcription_workers=self.settings.transcription.max_concurrent,
            extraction_workers=self.settings.vllm.max_concurrent,
            analysis_workers=self.settings.analysis.worker_count,
            net_autodetect_enabled=self.settings.net_autodetect.enabled,
            duplicate_detection_enabled=True,
            stuck_file_recovery_enabled=True,
        )

    async def start_health_server(self) -> None:
        """Start health check and metrics server."""
        if not self.settings.monitoring.enabled:
            logger.info("monitoring_disabled")
            return

        if not self.settings.server.enabled:
            logger.info("health_server_skipped", reason="server_disabled")
            return

        logger.info("starting_health_server")

        try:
            from tsn_server.health import run_server
        except ModuleNotFoundError as exc:
            logger.warning(
                "health_server_module_missing",
                error=str(exc),
            )
            return

        health_task = asyncio.create_task(
            run_server(
                host=self.settings.monitoring.host,
                port=self.settings.monitoring.port,
            )
        )
        self.tasks.append(health_task)

        logger.info(
            "health_server_started",
            url=f"http://{self.settings.monitoring.host}:{self.settings.monitoring.port}",
        )

    async def shutdown(self) -> None:
        """Gracefully shutdown all services."""
        logger.info("shutting_down_services")

        # Cancel all tasks
        for task in self.tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)

        logger.info("all_services_stopped")

    async def run(self) -> None:
        """Run all services."""
        logger.info("tsn_orchestrator_starting")

        try:
            await self.maybe_purge_net_history()
            # Start all services
            await self.start_node_services()
            await self.start_server_services()
            await self.start_health_server()

            logger.info(
                "tsn_orchestrator_running",
                total_tasks=len(self.tasks),
            )

            # Wait for shutdown signal
            await self.shutdown_event.wait()

        except asyncio.CancelledError:
            logger.info("orchestrator_cancelled")
        except Exception as e:
            logger.error("orchestrator_error", error=str(e), exc_info=True)
        finally:
            await self.shutdown()


def handle_signal(orchestrator: ServiceOrchestrator) -> None:
    """Handle shutdown signals."""
    logger.info("shutdown_signal_received")
    orchestrator.shutdown_event.set()


async def main() -> None:
    """Main entry point."""
    settings = get_settings()
    setup_logging(settings.logging)

    orchestrator = ServiceOrchestrator()

    # Setup signal handlers
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: handle_signal(orchestrator))

    # Run orchestrator
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
