"""vLLM micro-window evaluation pass."""

import json
import time
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from tsn_common.logging import get_logger

logger = get_logger(__name__)


class VLLMWindowOutput(BaseModel):
    """Strict schema for vLLM micro-window output."""
    
    net_likelihood: int = Field(ge=0, le=100, description="Likelihood this is a net (0-100)")
    signals: dict[str, Any] = Field(default_factory=dict, description="Extracted signals")
    evidence: list[str] = Field(default_factory=list, description="Evidence excerpts")
    suggested_action: str = Field(
        default="continue",
        description="continue|start_candidate|extend_candidate|end_candidate|ignore"
    )


async def evaluate_micro_window(
    analyzer,  # TranscriptAnalyzer instance
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
    if not excerpts:
        return {
            "net_likelihood": 0,
            "signals": {},
            "evidence": [],
            "suggested_action": "ignore",
            "vllm_latency_ms": 0,
        }
    
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

**TASK**: Evaluate net likelihood (0-100) and extract signals.

**NET INDICATORS** (score HIGH if present):
- ANY mention of "net" (e.g., "net control", "this is the [name] net", "net is now", "happy hour net")
- Check-in activity (calling for check-ins, operators checking in with callsigns)
- NCS managing traffic ("any check-ins?", "go ahead", "thanks for checking in")
- Opening/closing statements ("this is", "net will close")
- Roll call or directed conversation flow
- Multiple operators communicating in organized manner

**NOT A NET** (score LOW):
- Single operator transmission with no response
- Unstructured ragchew (back-and-forth conversation with no net structure)
- Testing or technical discussions only
- Completely random, unrelated transmissions

**SCORING GUIDE**:
- 80-100: Clear net activity with NCS and structure
- 60-79: Likely a net (multiple operators, some organization)
- 40-59: Possible net (check-ins mentioned, organized feel)
- 20-39: Weak signals (maybe informal net or ragchew)
- 0-19: No net activity (random QSOs or single operators)

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
            
        except (json.JSONDecodeError, ValidationError) as e:
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
