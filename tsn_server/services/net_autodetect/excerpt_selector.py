"""Select relevant excerpts from transcripts for vLLM micro-window evaluation."""

import re

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
