"""Candidate state machine for net auto-detection."""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tsn_common.models import NetCandidate, NetCandidateWindow, CandidateStatus
from tsn_common.config import NetAutoDetectSettings
from tsn_common.logging import get_logger

logger = get_logger(__name__)


class CandidateStateMachine:
    """Manages candidate lifecycle and state transitions."""
    
    def __init__(self, settings: NetAutoDetectSettings):
        self.settings = settings
    
    async def update_candidate(
        self,
        session: AsyncSession,
        candidate: NetCandidate,
        window_data: dict[str, Any],
        vllm_output: dict[str, Any],
    ) -> None:
        """
        Update candidate based on new vLLM window evaluation.
        
        State transitions:
        - WARMUP: net_likelihood trending up → start candidate
        - ACTIVE: sustained likelihood → keep extending
        - COOLING: likelihood dropping → prepare to end
        - ENDED: likelihood stayed low → await OpenAI verification
        """
        # Record window
        window = NetCandidateWindow(
            candidate_id=candidate.id,
            window_start=datetime.fromisoformat(window_data["window_start"]),
            window_end=datetime.fromisoformat(window_data["window_end"]),
            vllm_output_json=vllm_output,
            vllm_latency_ms=vllm_output.get("vllm_latency_ms"),
        )
        session.add(window)
        
        # Update candidate metrics
        likelihood = vllm_output["net_likelihood"]
        candidate.vllm_evaluation_count += 1
        
        # Update running average
        if candidate.vllm_confidence_avg is None:
            candidate.vllm_confidence_avg = float(likelihood)
        else:
            # Exponential moving average (weight recent windows more)
            alpha = 0.3
            candidate.vllm_confidence_avg = (
                alpha * likelihood + (1 - alpha) * candidate.vllm_confidence_avg
            )
        
        # Update peak
        if candidate.vllm_confidence_peak is None or likelihood > candidate.vllm_confidence_peak:
            candidate.vllm_confidence_peak = float(likelihood)
        
        # Update features
        features = dict(candidate.features_json or {})
        signals = vllm_output.get("signals", {})
        
        # Merge signals (keep track of NCS, callsigns, etc.)
        if "control_station_callsign" in signals and signals["control_station_callsign"]:
            features["control_station_callsign"] = signals["control_station_callsign"]
        
        # Track unique callsigns
        callsigns_set = set(features.get("unique_callsigns", []))
        callsigns_set.update(window_data.get("unique_callsigns", []))
        features["unique_callsigns"] = list(callsigns_set)
        
        # Average signal strengths
        for signal_key in ["checkin_activity", "directed_net_style", "roll_call_style", "social_net_style"]:
            if signal_key in signals:
                old_val = features.get(f"{signal_key}_avg", 0)
                new_val = signals[signal_key]
                features[f"{signal_key}_avg"] = (old_val + new_val) / 2.0
        
        candidate.features_json = features
        
        # Update evidence
        evidence = dict(candidate.evidence_json or {})
        excerpts = list(evidence.get("excerpts", []))
        excerpts.extend(vllm_output.get("evidence", []))
        evidence["excerpts"] = excerpts[-50:]  # Keep last 50
        
        transcript_ids = set(evidence.get("transcript_ids", []))
        transcript_ids.update(window_data.get("transcript_ids", []))
        evidence["transcript_ids"] = list(transcript_ids)
        
        audio_ids = set(evidence.get("audio_file_ids", []))
        audio_ids.update(window_data.get("audio_file_ids", []))
        evidence["audio_file_ids"] = list(audio_ids)
        
        candidate.evidence_json = evidence
        
        # State machine logic
        await self._evaluate_state_transition(session, candidate, likelihood)
        
        await session.flush()
    
    async def _evaluate_state_transition(
        self,
        session: AsyncSession,
        candidate: NetCandidate,
        current_likelihood: int,
    ) -> None:
        """Evaluate and perform state transitions."""
        
        # Get recent window likelihoods
        recent_stmt = (
            select(NetCandidateWindow)
            .where(NetCandidateWindow.candidate_id == candidate.id)
            .order_by(NetCandidateWindow.window_start.desc())
            .limit(max(self.settings.candidate_start_consecutive_windows,
                      self.settings.candidate_end_consecutive_windows))
        )
        recent_result = await session.execute(recent_stmt)
        recent_windows = list(recent_result.scalars().all())
        
        recent_likelihoods = [
            w.vllm_output_json.get("net_likelihood", 0)
            for w in recent_windows
        ]
        
        # WARMUP → ACTIVE
        # RELAXED CRITERIA: Use peak likelihood + average, not all windows
        if candidate.status == CandidateStatus.WARMUP:
            if len(recent_likelihoods) >= self.settings.candidate_start_consecutive_windows:
                # Check if ANY window hit high likelihood AND average is reasonable
                peak_in_recent = max(recent_likelihoods) if recent_likelihoods else 0
                avg_in_recent = sum(recent_likelihoods) / len(recent_likelihoods) if recent_likelihoods else 0
                
                # Activate if: (peak >= 65 AND avg >= 25) OR (avg >= 40)
                meets_likelihood = (
                    (peak_in_recent >= self.settings.candidate_start_likelihood and avg_in_recent >= 25) or
                    (avg_in_recent >= 40)
                )
                
                if meets_likelihood:
                    unique_callsigns = len(candidate.features_json.get("unique_callsigns", []))
                    if unique_callsigns >= self.settings.candidate_min_unique_callsigns:
                        candidate.status = CandidateStatus.ACTIVE
                        logger.info(
                            "net_autodetect_candidate_activated",
                            candidate_id=str(candidate.id),
                            likelihood_avg=candidate.vllm_confidence_avg,
                            likelihood_peak=peak_in_recent,
                            likelihood_recent_avg=avg_in_recent,
                            callsigns=unique_callsigns,
                        )
        
        # ACTIVE → COOLING
        elif candidate.status == CandidateStatus.ACTIVE:
            if current_likelihood < self.settings.candidate_extend_likelihood:
                # Check if sustained drop
                if len(recent_likelihoods) >= 2:
                    if all(l < self.settings.candidate_extend_likelihood for l in recent_likelihoods[:2]):
                        candidate.status = CandidateStatus.COOLING
                        logger.info(
                            "net_autodetect_candidate_cooling",
                            candidate_id=str(candidate.id),
                            likelihood=current_likelihood,
                        )
        
        # COOLING → ENDED or back to ACTIVE
        elif candidate.status == CandidateStatus.COOLING:
            if current_likelihood >= self.settings.candidate_extend_likelihood:
                # Reactivate
                candidate.status = CandidateStatus.ACTIVE
                logger.info(
                    "net_autodetect_candidate_reactivated",
                    candidate_id=str(candidate.id),
                )
            elif len(recent_likelihoods) >= self.settings.candidate_end_consecutive_windows:
                if all(l < self.settings.candidate_end_likelihood for l in recent_likelihoods):
                    candidate.status = CandidateStatus.ENDED
                    candidate.end_ts = datetime.now(timezone.utc)
                    logger.info(
                        "net_autodetect_candidate_ended",
                        candidate_id=str(candidate.id),
                        duration_minutes=int((candidate.end_ts - candidate.start_ts).total_seconds() / 60),
                        evaluations=candidate.vllm_evaluation_count,
                    )
    
    async def get_or_create_active_candidate(
        self,
        session: AsyncSession,
        node_id: str,
        window_start: datetime,
    ) -> NetCandidate | None:
        """Get active candidate for node or create new WARMUP candidate."""
        
        # Look for ACTIVE/WARMUP/COOLING candidates for this node
        stmt = (
            select(NetCandidate)
            .where(
                NetCandidate.node_id == node_id,
                NetCandidate.status.in_([
                    CandidateStatus.WARMUP,
                    CandidateStatus.ACTIVE,
                    CandidateStatus.COOLING,
                ]),
            )
            .order_by(NetCandidate.start_ts.desc())
            .limit(1)
        )
        result = await session.execute(stmt)
        existing = result.scalar_one_or_none()
        
        if existing:
            return existing
        
        # Create new WARMUP candidate
        candidate = NetCandidate(
            status=CandidateStatus.WARMUP,
            start_ts=window_start,
            node_id=node_id,
            vllm_evaluation_count=0,
        )
        session.add(candidate)
        await session.flush()
        
        logger.info(
            "net_autodetect_candidate_created",
            candidate_id=str(candidate.id),
            node_id=node_id,
            start_ts=window_start.isoformat(),
        )
        
        return candidate
