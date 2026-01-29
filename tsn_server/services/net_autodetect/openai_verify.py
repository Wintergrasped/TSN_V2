"""OpenAI verification for final net candidate adjudication."""

import json
import os
from datetime import datetime, timezone
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from tsn_common.models import NetCandidate, CandidateStatus
from tsn_common.config import NetAutoDetectSettings, VLLMSettings
from tsn_common.logging import get_logger

logger = get_logger(__name__)


class OpenAIVerdict(BaseModel):
    """Strict schema for OpenAI verification response."""
    
    is_net: bool = Field(description="True if this is a legitimate net")
    confidence: int = Field(ge=0, le=100, description="Confidence 0-100")
    net_type: str = Field(
        default="unknown",
        description="open|directed|roll_call|traffic|emergency|social|unknown"
    )
    ncs_callsign: str | None = Field(default=None, description="Net Control Station callsign")
    start_time_adjust: str | None = Field(default=None, description="ISO timestamp adjustment")
    end_time_adjust: str | None = Field(default=None, description="ISO timestamp adjustment")
    why: list[str] = Field(default_factory=list, description="Reasoning bullets (max 6)")
    false_positive_reason: str | None = Field(default=None, description="If is_net=false, why")


async def verify_candidate_with_openai(
    candidate: NetCandidate,
    session: AsyncSession,
    settings: NetAutoDetectSettings,
    vllm_settings: VLLMSettings,
) -> dict[str, Any] | None:
    """
    Call OpenAI API to verify a candidate as final adjudication.
    
    Args:
        candidate: NetCandidate to verify
        session: Database session
        settings: NetAutoDetectSettings
        vllm_settings: VLLMSettings (for OpenAI key)
        
    Returns:
        Verdict dict or None if verification fails
    """
    # Get OpenAI API key
    api_key = None
    for key_source in [
        os.getenv("TSN_WEB_OPENAI_API_KEY"),
        os.getenv("TSN_VLLM_OPENAI_API_KEY"),
        os.getenv("TSN_OPENAI_API_KEY"),
    ]:
        if key_source and key_source.strip():
            api_key = key_source.strip()
            break
    
    if vllm_settings.openai_api_key:
        api_key = vllm_settings.openai_api_key.get_secret_value().strip()
    
    if not api_key:
        logger.warning("net_autodetect_openai_no_api_key")
        return None
    
    # Build summary package
    features = candidate.features_json or {}
    evidence = candidate.evidence_json or {}
    excerpts = evidence.get("excerpts", [])[:settings.openai_max_evidence_excerpts]
    
    duration_minutes = 0
    if candidate.end_ts:
        duration_minutes = int((candidate.end_ts - candidate.start_ts).total_seconds() / 60)
    
    summary_package = {
        "time_window": {
            "start": candidate.start_ts.isoformat(),
            "end": candidate.end_ts.isoformat() if candidate.end_ts else None,
            "duration_minutes": duration_minutes,
        },
        "metrics": {
            "vllm_evaluations": candidate.vllm_evaluation_count,
            "vllm_confidence_avg": candidate.vllm_confidence_avg,
            "vllm_confidence_peak": candidate.vllm_confidence_peak,
            "unique_callsigns": len(features.get("unique_callsigns", [])),
        },
        "features": features,
        "evidence_excerpts": excerpts,
        "vllm_likelihood_timeline": "(sparkline data not yet implemented)",
    }
    
    prompt = f"""You are the final adjudicator for amateur radio net detection.

**CANDIDATE SUMMARY**:
{json.dumps(summary_package, indent=2)}

**TASK**: Determine if this is a legitimate amateur radio net.

FORMAL NETS have:
- Opening statement with net name/purpose
- Formal check-ins with callsigns, names, locations
- Net Control Station (NCS) managing traffic
- Closing statement

RESPOND STRICTLY WITH JSON:
{{
  "is_net": true|false,
  "confidence": 0-100,
  "net_type": "open|directed|roll_call|traffic|emergency|social|unknown",
  "ncs_callsign": "K7XYZ or null",
  "start_time_adjust": "ISO or null",
  "end_time_adjust": "ISO or null",
  "why": ["short bullet 1", "short bullet 2", ...],
  "false_positive_reason": "short explanation if is_net=false"
}}

Be conservative. Confidence >=80 required for VERIFIED status.
"""
    
    try:
        client = AsyncOpenAI(api_key=api_key)
        
        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert amateur radio analyst performing final net verification.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        
        response_text = response.choices[0].message.content
        
        # Parse and validate
        try:
            verdict_data = json.loads(response_text)
            verdict = OpenAIVerdict(**verdict_data)
            
            # Update candidate
            candidate.openai_verdict_json = verdict.model_dump()
            candidate.openai_verified_at = datetime.now(timezone.utc)
            
            # Update status based on verdict
            if verdict.is_net and verdict.confidence >= settings.openai_min_confidence:
                candidate.status = CandidateStatus.VERIFIED
                logger.info(
                    "net_autodetect_openai_verified",
                    candidate_id=str(candidate.id),
                    confidence=verdict.confidence,
                    net_type=verdict.net_type,
                )
            else:
                candidate.status = CandidateStatus.REJECTED
                logger.info(
                    "net_autodetect_openai_rejected",
                    candidate_id=str(candidate.id),
                    confidence=verdict.confidence,
                    reason=verdict.false_positive_reason,
                )
            
            await session.flush()
            
            return verdict.model_dump()
            
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(
                "net_autodetect_openai_invalid_json",
                error=str(e),
                response_preview=response_text[:200],
            )
            return None
            
    except Exception as e:
        logger.error(
            "net_autodetect_openai_call_failed",
            error=str(e),
            candidate_id=str(candidate.id),
        )
        return None
