# TSN v2 Net AutoDetect Rebuild - Implementation Guide

## STATUS: Database Schema Complete, Core Modules In Progress

### Completed (Commit Ready)

#### 1. Database Models (`tsn_common/models/net_candidate.py`)
✅ **NetCandidate** table with streaming detection support:
- Status enum: WARMUP → ACTIVE → COOLING → ENDED → VERIFIED/REJECTED/DISMISSED/PROMOTED
- Time windows (start_ts, end_ts)
- Node scope tracking
- vLLM confidence metrics (avg, peak, evaluation_count)
- OpenAI verdict storage (JSON + timestamp)
- Features JSON (extracted signals from vLLM)
- Evidence JSON (excerpts + transcript/audio IDs)
- Link to promoted NetSession

✅ **NetCandidateWindow** table for micro-window evaluations:
- Links to parent candidate
- Window time range
- Full vLLM output JSON (likelihood, signals, evidence, suggested_action)
- Performance tracking (vllm_latency_ms)

✅ **Indexes** for performance:
- `(status, updated_at)` - for API queries
- `(node_id, start_ts)` - for per-node lookups
- `(status, node_id, start_ts)` - for active candidate queries
- `(candidate_id, window_start)` - for timeline queries

#### 2. Configuration (`tsn_common/config.py`)
✅ **NetAutoDetectSettings** class with all required parameters:
- **Micro-window**: 4-minute windows, 45-second steps, 20 excerpts/window
- **vLLM frequency**: Target 1 call/node/minute, 3 concurrent/node, backpressure at 10
- **Thresholds**: Start at 65% (3 windows), extend at 55%, end at <40% (4 windows)
- **Multi-pass intervals**: Boundary refinement every 7 min, roster assist every 10 min
- **OpenAI**: Enabled, 80% min confidence, gpt-4o-mini, max 60 excerpts
- **Performance**: 5s orchestrator poll, 15 min node inactivity timeout

#### 3. Model Exports (`tsn_common/models/__init__.py`)
✅ Added NetCandidate, NetCandidateWindow, CandidateStatus to exports

---

## Next Steps: Core Module Implementation

### Module 1: window_builder.py
**Purpose**: Build micro-windows of recent activity per node

```python
"""Build micro-windows of transcripts for vLLM evaluation."""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from tsn_common.models import AudioFile, Transcription, CallsignLog
from tsn_common.logging import get_logger

logger = get_logger(__name__)


async def build_micro_window(
    session: AsyncSession,
    node_id: str | None,
    window_start: datetime,
    window_end: datetime,
) -> dict[str, Any]:
    """
    Build a micro-window summary for vLLM evaluation.
    
    Returns:
        {
            "window_start": ISO timestamp,
            "window_end": ISO timestamp,
            "node_id": str,
            "transmissions": 5,
            "unique_callsigns": ["K7XYZ", "W1ABC"],
            "top_callsigns": [{"callsign": "K7XYZ", "count": 3}],
            "duration_sec": 240,
            "excerpts": ["transcript snippet 1", ...],
            "transcript_ids": [uuid, ...],
            "audio_file_ids": [uuid, ...]
        }
    """
    # Query transcripts in window
    stmt = (
        select(Transcription, AudioFile)
        .join(AudioFile, Transcription.audio_file_id == AudioFile.id)
        .where(
            AudioFile.created_at >= window_start,
            AudioFile.created_at < window_end,
        )
    )
    if node_id:
        stmt = stmt.where(AudioFile.node_id == node_id)
    
    stmt = stmt.order_by(AudioFile.created_at)
    
    result = await session.execute(stmt)
    rows = result.all()
    
    if not rows:
        return {
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "node_id": node_id,
            "transmissions": 0,
            "unique_callsigns": [],
            "top_callsigns": [],
            "duration_sec": 0,
            "excerpts": [],
            "transcript_ids": [],
            "audio_file_ids": [],
        }
    
    # Extract data
    transcript_ids = []
    audio_file_ids = []
    all_text = []
    total_duration = 0
    
    for trans, audio in rows:
        transcript_ids.append(trans.id)
        audio_file_ids.append(audio.id)
        all_text.append(trans.transcript_text or "")
        total_duration += audio.duration_sec or 0
    
    # Get callsigns in window
    callsign_stmt = (
        select(
            CallsignLog.callsign,
            func.count(CallsignLog.id).label("count")
        )
        .join(AudioFile, CallsignLog.audio_file_id == AudioFile.id)
        .where(
            AudioFile.created_at >= window_start,
            AudioFile.created_at < window_end,
        )
        .group_by(CallsignLog.callsign)
        .order_by(func.count(CallsignLog.id).desc())
    )
    if node_id:
        callsign_stmt = callsign_stmt.where(AudioFile.node_id == node_id)
    
    callsign_result = await session.execute(callsign_stmt)
    callsign_rows = callsign_result.all()
    
    unique_callsigns = [row.callsign for row in callsign_rows]
    top_callsigns = [
        {"callsign": row.callsign, "count": row.count}
        for row in callsign_rows[:10]
    ]
    
    return {
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "node_id": node_id,
        "transmissions": len(rows),
        "unique_callsigns": unique_callsigns,
        "top_callsigns": top_callsigns,
        "duration_sec": int(total_duration),
        "excerpts": all_text,  # Will be curated by excerpt_selector
        "transcript_ids": [str(tid) for tid in transcript_ids],
        "audio_file_ids": [str(aid) for aid in audio_file_ids],
    }


async def get_active_nodes(
    session: AsyncSession,
    since: datetime,
) -> list[str]:
    """Get list of nodes with activity since given time."""
    stmt = (
        select(AudioFile.node_id)
        .where(
            AudioFile.created_at >= since,
            AudioFile.node_id.isnot(None),
        )
        .group_by(AudioFile.node_id)
    )
    result = await session.execute(stmt)
    return [row[0] for row in result.all() if row[0]]
```

### Module 2: excerpt_selector.py
**Purpose**: Curate transcript excerpts using heuristics (NOT skipping vLLM)

```python
"""Select relevant excerpts from transcripts for vLLM micro-window evaluation."""

import re
from typing import Any

# Net phrase patterns (used to PRIORITIZE excerpts, not GATE vLLM calls)
NET_PHRASES = [
    r"\bthis is the\b.*\bnet\b",
    r"\bcheck.?in\b",
    r"\bany check.?ins\b",
    r"\bgo ahead with your call\b",
    r"\bthis concludes\b",
    r"\b73s?\b",
    r"\bnet control\b",
    r"\broll call\b",
    r"\btraffic\b.*\bnet\b",
]

CALLSIGN_PATTERN = re.compile(r"\b[A-Z]{1,2}\d[A-Z]{1,4}\b")


def select_excerpts(
    transcripts: list[str],
    max_excerpts: int = 20,
) -> list[str]:
    """
    Curate transcript excerpts for vLLM prompt.
    
    Priority:
    1. Excerpts with net phrases
    2. Excerpts with multiple callsigns
    3. Longer excerpts (more context)
    4. Time-distributed samples
    
    Args:
        transcripts: List of transcript texts
        max_excerpts: Max number of excerpts to return
        
    Returns:
        List of curated excerpt strings
    """
    if not transcripts:
        return []
    
    # Score each transcript
    scored = []
    for i, text in enumerate(transcripts):
        score = 0.0
        
        # Net phrase hits
        for pattern in NET_PHRASES:
            if re.search(pattern, text, re.IGNORECASE):
                score += 2.0
        
        # Callsign count
        callsigns = CALLSIGN_PATTERN.findall(text)
        score += len(set(callsigns)) * 0.5
        
        # Length bonus (prefer substantial excerpts)
        score += min(len(text) / 500, 1.0)
        
        scored.append((score, i, text))
    
    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Take top N, but ensure time distribution
    selected = []
    selected_indices = set()
    
    # First pass: high-scoring excerpts
    for score, idx, text in scored[:max_excerpts]:
        if score > 1.0:  # Has net phrase or multiple callsigns
            selected.append(f"[T{idx+1}] {text[:400]}")
            selected_indices.add(idx)
    
    # Second pass: fill remainder with time-distributed samples
    if len(selected) < max_excerpts:
        stride = max(1, len(transcripts) // (max_excerpts - len(selected)))
        for i in range(0, len(transcripts), stride):
            if i not in selected_indices and len(selected) < max_excerpts:
                text = transcripts[i]
                selected.append(f"[T{i+1}] {text[:400]}")
                selected_indices.add(i)
    
    return selected[:max_excerpts]
```

### Module 3: vllm_pass.py
**Purpose**: Call vLLM with micro-window data (FREQUENTLY)

```python
"""vLLM micro-window evaluation pass."""

import json
import time
from typing import Any

from pydantic import BaseModel, Field

from tsn_common.logging import get_logger
from tsn_server.analyzer import TranscriptAnalyzer  # Reuse existing vLLM client

logger = get_logger(__name__)


class VLLMWindowOutput(BaseModel):
    """Strict schema for vLLM micro-window output."""
    
    net_likelihood: int = Field(ge=0, le=100, description="Likelihood this is a net (0-100)")
    signals: dict[str, Any] = Field(description="Extracted signals")
    evidence: list[str] = Field(default_factory=list, description="Evidence excerpts")
    suggested_action: str = Field(
        default="continue",
        description="continue|start_candidate|extend_candidate|end_candidate|ignore"
    )


async def evaluate_micro_window(
    analyzer: TranscriptAnalyzer,
    window_data: dict[str, Any],
    excerpts: list[str],
) -> dict[str, Any]:
    """
    Call vLLM to evaluate a micro-window for net likelihood.
    
    Args:
        analyzer: TranscriptAnalyzer instance (for vLLM client)
        window_data: Window metadata from window_builder
        excerpts: Curated excerpts from excerpt_selector
        
    Returns:
        {
            "net_likelihood": 0-100,
            "signals": {
                "control_station_callsign": "K7XYZ",
                "checkin_activity": 0-100,
                "directed_net_style": 0-100,
                "roll_call_style": 0-100,
                "social_net_style": 0-100
            },
            "evidence": ["excerpt1", "excerpt2"],
            "suggested_action": "extend_candidate",
            "vllm_latency_ms": 1234
        }
    """
    prompt = f"""You are analyzing a 3-5 minute micro-window of amateur radio activity.

**WINDOW SUMMARY**:
- Time: {window_data['window_start']} to {window_data['window_end']}
- Node: {window_data['node_id']}
- Transmissions: {window_data['transmissions']}
- Unique Callsigns: {len(window_data['unique_callsigns'])}
- Top Callsigns: {', '.join([f"{c['callsign']} ({c['count']})" for c in window_data['top_callsigns'][:5]])}
- Duration: {window_data['duration_sec']}s

**TRANSCRIPT EXCERPTS**:
{chr(10).join(excerpts)}

**TASK**: Evaluate net likelihood and extract signals.

Respond STRICTLY with JSON:
{{
  "net_likelihood": 0-100,
  "signals": {{
    "control_station_callsign": "K7XYZ or null",
    "checkin_activity": 0-100,
    "directed_net_style": 0-100,
    "roll_call_style": 0-100,
    "social_net_style": 0-100
  }},
  "evidence": ["Short excerpts showing net behavior"],
  "suggested_action": "continue|start_candidate|extend_candidate|end_candidate|ignore"
}}

FORMAL NETS have:
- NCS managing traffic ("Any check-ins?", "Go ahead", "We have...")
- Formal check-ins with callsign/name/location
- Opening ("This is the [name] net") or closing statements
- Directed conversation flow

RANDOM QSOs have:
- Unstructured back-and-forth
- No formal check-ins
- No NCS coordination
"""
    
    start = time.perf_counter()
    try:
        response_text, _ = await analyzer.call_vllm(
            prompt,
            pass_label="net_autodetect_microwindow",
            audio_file_ids=None,
            extra_metadata={
                "window_start": window_data["window_start"],
                "node_id": window_data["node_id"],
                "transmissions": window_data["transmissions"],
            },
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        
        # Parse and validate
        try:
            output = json.loads(response_text)
            validated = VLLMWindowOutput(**output)
            result = validated.model_dump()
            result["vllm_latency_ms"] = latency_ms
            
            logger.info(
                "net_autodetect_vllm_pass_microwindow",
                node_id=window_data["node_id"],
                window_start=window_data["window_start"],
                net_likelihood=result["net_likelihood"],
                suggested_action=result["suggested_action"],
                latency_ms=latency_ms,
            )
            
            return result
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(
                "net_autodetect_vllm_invalid_json",
                error=str(e),
                response_preview=response_text[:200],
            )
            # Return low-confidence default
            return {
                "net_likelihood": 0,
                "signals": {},
                "evidence": [],
                "suggested_action": "ignore",
                "vllm_latency_ms": latency_ms,
            }
            
    except Exception as e:
        logger.error(
            "net_autodetect_vllm_call_failed",
            error=str(e),
            node_id=window_data["node_id"],
        )
        return {
            "net_likelihood": 0,
            "signals": {},
            "evidence": [],
            "suggested_action": "ignore",
            "vllm_latency_ms": 0,
        }
```

### Module 4: candidate_state.py  
**Purpose**: State machine for managing candidates

```python
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
        if candidate.status == CandidateStatus.WARMUP:
            if len(recent_likelihoods) >= self.settings.candidate_start_consecutive_windows:
                if all(l >= self.settings.candidate_start_likelihood for l in recent_likelihoods):
                    unique_callsigns = len(candidate.features_json.get("unique_callsigns", []))
                    if unique_callsigns >= self.settings.candidate_min_unique_callsigns:
                        candidate.status = CandidateStatus.ACTIVE
                        logger.info(
                            "net_autodetect_candidate_activated",
                            candidate_id=str(candidate.id),
                            likelihood_avg=candidate.vllm_confidence_avg,
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
```

### REMAINING MODULES (Summary)

#### Module 5: openai_verify.py
- Reuse existing OpenAI client from analyzer.py
- Build candidate summary package (30-80 excerpts, timeline, features)
- Call OpenAI API with strict JSON schema
- Update candidate status to VERIFIED/REJECTED based on response

#### Module 6: orchestrator.py
- Main loop: poll every 5 seconds
- Get active nodes from DB
- For each node: build micro-windows, call vLLM, update candidates
- Check ENDED candidates → trigger OpenAI verification
- Semaphore-based concurrency control (3 concurrent vLLM/node)
- Restart resilience: reload ACTIVE candidates from DB

#### Module 7: API Endpoints (`tsn_server/routes/`)
- GET `/api/net/candidates?status=ACTIVE` - list candidates
- GET `/api/net/candidates/{id}` - get candidate details
- POST `/api/net/candidates/{id}/promote` - create NetSession
- POST `/api/net/candidates/{id}/dismiss` - mark DISMISSED
- GET `/api/net/candidates/{id}/timeline` - vLLM likelihood over time

#### Module 8: Web UI (`web/templates/net_suggestions.html` + routes)
- Live candidates table with status badges
- vLLM confidence sparkline charts
- Evidence excerpts collapsible sections
- OpenAI verdict display
- Promote/Dismiss action buttons

---

## Migration Strategy (Auto-Migration Style)

The system will automatically create tables on first run via existing auto-migration infrastructure in `tsn_common/db_init.py`.

**Verification**:
```bash
# After deployment, verify tables created:
docker exec tsn_server_db mysql -u tsn_user -p tsn -e "SHOW TABLES LIKE 'net_candidate%';"
```

---

## Testing Plan

1. **vLLM Call Frequency**: Check logs for "NET_AUTODETECT vllm pass micro-window" every ~60s per node
2. **False Positive Elimination**: Upload 3-transmission test case → should NOT create VERIFIED candidate
3. **Real Net Detection**: Record real net → ACTIVE candidate within 5-10 minutes
4. **OpenAI Verification**: ENDED candidate → VERIFIED status with verdict JSON
5. **Promotion**: Manually promote candidate → NetSession created with correct data

---

## Deployment Instructions

### 1. Commit Changes
```bash
git add tsn_common/models/net_candidate.py
git add tsn_common/models/__init__.py
git add tsn_common/config.py
git add tsn_server/services/net_autodetect/
git commit -m "Net AutoDetect: Add streaming vLLM-heavy detection with OpenAI verification"
git push
```

### 2. Server Deployment
```bash
# On server
cd /opt/tsn/TSN_V2
git pull
docker compose down
docker compose up -d --build
```

### 3. Environment Variables (Optional Tuning)
Add to `.env`:
```bash
TSN_NET_AUTODETECT_ENABLED=true
TSN_NET_AUTODETECT_WINDOW_SIZE_MINUTES=4
TSN_NET_AUTODETECT_WINDOW_STEP_SECONDS=45
TSN_NET_AUTODETECT_CANDIDATE_START_LIKELIHOOD=65
TSN_NET_AUTODETECT_OPENAI_MODEL=gpt-4o-mini
```

### 4. Monitor Logs
```bash
docker logs tsn_server -f | grep -E "net_autodetect|NET_AUTODETECT"
```

Expected log patterns:
```
{"event": "net_autodetect_vllm_pass_microwindow", "node_id": "66296", "net_likelihood": 72, ...}
{"event": "net_autodetect_candidate_activated", "candidate_id": "...", "callsigns": 8}
{"event": "net_autodetect_candidate_ended", "duration_minutes": 45}
{"event": "net_autodetect_openai_verified", "is_net": true, "confidence": 92}
```

---

## Key Differences from Old System

| Aspect | OLD System | NEW System |
|--------|-----------|------------|
| **vLLM Calls** | 1 per batch (avoid calling) | 1/min/node (aggressive) |
| **Detection** | Single-pass classification | Streaming state machine |
| **Evidence** | Batch context | 50+ micro-windows |
| **Verification** | vLLM validation pass | OpenAI final adjudication |
| **False Positives** | Common (3 transmissions) | Eliminated (sustained trend) |
| **Latency** | Minutes to hours | 5-10 minutes |
| **Confidence** | Single score | Timeline with 50+ datapoints |

---

## Next Actions

1. **IMMEDIATE**: Implement remaining modules 5-8 (see code templates above)
2. **TEST**: Deploy to staging, test with recorded net samples
3. **TUNE**: Adjust thresholds based on real-world performance
4. **MONITOR**: Watch logs for vLLM call frequency and detection accuracy

