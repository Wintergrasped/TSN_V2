"""Orchestrator for net auto-detection - coordinates all components."""

import asyncio
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from tsn_common.config import get_settings
from tsn_common.db import get_session
from tsn_common.logging import get_logger
from tsn_common.models import NetCandidate, CandidateStatus
from tsn_server.analyzer import TranscriptAnalyzer
from tsn_server.services.net_autodetect.window_builder import (
    build_micro_window,
    get_active_nodes,
)
from tsn_server.services.net_autodetect.excerpt_selector import select_excerpts
from tsn_server.services.net_autodetect.vllm_pass import evaluate_micro_window
from tsn_server.services.net_autodetect.candidate_state import CandidateStateMachine
from tsn_server.services.net_autodetect.openai_verify import verify_candidate_with_openai

logger = get_logger(__name__)


class NetAutoDetectOrchestrator:
    """
    Main orchestrator for streaming net auto-detection.
    
    Runs continuously, evaluating micro-windows and managing candidates.
    """
    
    def __init__(self):
        settings = get_settings()
        self.settings = settings.net_autodetect
        self.vllm_settings = settings.vllm
        self.analysis_settings = settings.analysis
        
        # Create TranscriptAnalyzer for vLLM client
        self.analyzer = TranscriptAnalyzer(
            vllm_settings=self.vllm_settings,
            analysis_settings=self.analysis_settings,
        )
        
        # Create state machine
        self.state_machine = CandidateStateMachine(self.settings)
        
        # Per-node concurrency semaphores
        self.node_semaphores: dict[str, asyncio.Semaphore] = {}
        
        logger.info(
            "net_autodetect_orchestrator_initialized",
            window_size_minutes=self.settings.window_size_minutes,
            window_step_seconds=self.settings.window_step_seconds,
            vllm_call_interval_sec=self.settings.vllm_call_interval_sec,
        )
    
    def _get_node_semaphore(self, node_id: str) -> asyncio.Semaphore:
        """Get or create semaphore for node to limit concurrent vLLM calls."""
        if node_id not in self.node_semaphores:
            self.node_semaphores[node_id] = asyncio.Semaphore(
                self.settings.vllm_max_concurrent_per_node
            )
        return self.node_semaphores[node_id]
    
    async def process_node_window(
        self,
        node_id: str,
        window_start: datetime,
        window_end: datetime,
    ) -> None:
        """Process a single micro-window for a node."""
        semaphore = self._get_node_semaphore(node_id)
        
        async with semaphore:
            try:
                async with get_session() as session:
                    # Build micro-window
                    window_data = await build_micro_window(
                        session,
                        node_id,
                        window_start,
                        window_end,
                    )
                    
                    # Skip if no transmissions
                    if window_data["transmissions"] == 0:
                        return
                    
                    # Select excerpts
                    excerpts = select_excerpts(
                        window_data["excerpts"],
                        max_excerpts=self.settings.max_excerpts_per_window,
                    )
                    
                    # Call vLLM
                    vllm_output = await evaluate_micro_window(
                        self.analyzer,
                        window_data,
                        excerpts,
                    )
                    
                    # Get or create candidate
                    candidate = await self.state_machine.get_or_create_active_candidate(
                        session,
                        node_id,
                        window_start,
                    )
                    
                    if candidate:
                        # Update candidate with vLLM results
                        await self.state_machine.update_candidate(
                            session,
                            candidate,
                            window_data,
                            vllm_output,
                        )
                    
            except Exception as e:
                logger.error(
                    "net_autodetect_process_window_failed",
                    node_id=node_id,
                    window_start=window_start.isoformat(),
                    error=str(e),
                    exc_info=True,
                )
    
    async def verify_ended_candidates(self) -> None:
        """Check for ENDED candidates and trigger OpenAI verification."""
        try:
            async with get_session() as session:
                stmt = (
                    select(NetCandidate)
                    .where(NetCandidate.status == CandidateStatus.ENDED)
                    .order_by(NetCandidate.end_ts.desc())
                    .limit(5)  # Process max 5 per cycle
                )
                result = await session.execute(stmt)
                ended_candidates = list(result.scalars().all())
                
                for candidate in ended_candidates:
                    if not self.settings.openai_verify_enabled:
                        # Auto-promote without OpenAI
                        candidate.status = CandidateStatus.VERIFIED
                        logger.info(
                            "net_autodetect_candidate_auto_verified",
                            candidate_id=str(candidate.id),
                        )
                        await session.flush()
                        continue
                    
                    # Call OpenAI for verification
                    await verify_candidate_with_openai(
                        candidate,
                        session,
                        self.settings,
                        self.vllm_settings,
                    )
                
        except Exception as e:
            logger.error(
                "net_autodetect_verify_ended_failed",
                error=str(e),
                exc_info=True,
            )
    
    async def run_cycle(self) -> None:
        """Run one orchestration cycle."""
        now = datetime.now(timezone.utc)
        
        # Get active nodes
        async with get_session() as session:
            inactivity_cutoff = now - timedelta(
                minutes=self.settings.node_inactivity_minutes
            )
            active_nodes = await get_active_nodes(session, inactivity_cutoff)
        
        if not active_nodes:
            return
        
        # Process micro-windows for each node
        window_size = timedelta(minutes=self.settings.window_size_minutes)
        window_end = now
        window_start = window_end - window_size
        
        tasks = []
        for node_id in active_nodes:
            task = asyncio.create_task(
                self.process_node_window(node_id, window_start, window_end)
            )
            tasks.append(task)
        
        # Wait for all windows to process
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify ended candidates
        await self.verify_ended_candidates()
    
    async def run(self) -> None:
        """Main orchestrator loop - runs continuously."""
        logger.info("net_autodetect_orchestrator_started")
        
        try:
            while True:
                try:
                    await self.run_cycle()
                except Exception as e:
                    logger.error(
                        "net_autodetect_cycle_failed",
                        error=str(e),
                        exc_info=True,
                    )
                
                # Wait before next cycle
                await asyncio.sleep(self.settings.orchestrator_poll_interval_sec)
                
        except asyncio.CancelledError:
            logger.info("net_autodetect_orchestrator_cancelled")
            raise
        except Exception as e:
            logger.error(
                "net_autodetect_orchestrator_fatal",
                error=str(e),
                exc_info=True,
            )
            raise
