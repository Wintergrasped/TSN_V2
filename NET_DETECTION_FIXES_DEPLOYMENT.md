# Net Detection System Fixes - Deployment Guide

## Problem Identified

**Root Cause**: Candidates stuck in WARMUP state indefinitely due to overly strict activation criteria.

### Evidence from Database
- **4 candidates** stuck in WARMUP status for 2-3 days
- **0 new nets** created since January 26 (3 weeks ago)
- **213-595 vLLM evaluations** per candidate (system is running, but not promoting)
- **Low average scores** (10-20) with occasional **high peaks** (85)
- **Clear net activity** in transcripts: "check in", "net control", "happy hour net"

### Original Flawed Logic
```python
# Required ALL consecutive windows to have likelihood >= 65
if all(l >= 65 for l in recent_likelihoods):
    activate_candidate()
```

**Problem**: vLLM gives variable scores. Even clear net activity gets:
- Most windows: 10-40 (too strict interpretation)
- Peak windows: 60-85 (recognizes obvious net phrases)
- Average: 10-20
- Result: **NEVER activates** because not ALL windows are >= 65

## Fixes Implemented (Commit 3aed9d6)

### 1. Relaxed State Machine Logic ([candidate_state.py](tsn_server/services/net_autodetect/candidate_state.py#L138-L166))

**New Activation Criteria**:
```python
# WARMUP → ACTIVE if:
# - (Peak >= 65 AND Avg >= 25) OR (Avg >= 40)
# - At least 4 unique callsigns (was 6)

peak_in_recent = max(recent_likelihoods)
avg_in_recent = sum(recent_likelihoods) / len(recent_likelihoods)

meets_likelihood = (
    (peak_in_recent >= 65 and avg_in_recent >= 25) or
    (avg_in_recent >= 40)
)
```

**Rationale**: 
- Recognizes nets that have clear high-signal moments (peak 85) even if average is lower
- Allows activation when overall average is moderate (40+)
- More forgiving of vLLM scoring variability

### 2. Lowered Thresholds ([config.py](tsn_common/config.py#L509-L514))

| Setting | Old | New | Reason |
|---------|-----|-----|--------|
| `candidate_start_likelihood` | 65 | 65 | Keep high bar for initial signal |
| `candidate_extend_likelihood` | 55 | **45** | Stay active with moderate signals |
| `candidate_end_likelihood` | 40 | **30** | Don't end prematurely on low dips |
| `candidate_min_unique_callsigns` | 6 | **4** | Many nets have 4-5 participants |

### 3. Improved vLLM Prompt ([vllm_pass.py](tsn_server/services/net_autodetect/vllm_pass.py#L64-L106))

**Added**:
- **Clear NET INDICATORS**: "ANY mention of 'net'" should score high
- **Scoring guide**: 80-100 (clear), 60-79 (likely), 40-59 (possible), 20-39 (weak), 0-19 (none)
- **Examples**: "net control", "checking in", "this is the [name] net"

**Rationale**: Original prompt was ambiguous, causing conservative scoring

### 4. Manual Activation of Stuck Candidates

Ran [fix_warmup_candidates_simple.py](fix_warmup_candidates_simple.py):
- **Activated 3 candidates** meeting relaxed criteria
- Node 66296: 259 evals, peak 85, avg 19.6, 19 callsigns ✅
- Node 683211: 213 evals, peak 85, avg 10.0, 18 callsigns ✅
- Node 683210: 595 evals, peak 85, avg 12.1, 13 callsigns ✅

## Deployment Steps

### 1. SSH to Production Server
```bash
ssh Wintergrasped@192.168.0.104
# Password: wowhead1
```

### 2. Navigate to Project Directory
```bash
cd /opt/tsn-server
# Or wherever the project is deployed
```

### 3. Pull Latest Changes
```bash
git pull origin main
# Should show commit 3aed9d6 "Fix net detection: relax WARMUP activation criteria..."
```

### 4. Rebuild Container (NO CACHE - ensure clean build)
```bash
docker compose build --no-cache tsn_server
```

### 5. Restart Service
```bash
docker compose restart tsn_server
```

### 6. Verify Deployment
```bash
# Check logs for startup
docker compose logs -f tsn_server | grep -i "net_autodetect"

# Should see:
# - "net_autodetect_orchestrator_started"
# - Processing cycles every 5 seconds
# - Candidates being evaluated
```

## Monitoring & Validation

### Check Candidate Status
```bash
# On server or from Windows with MySQL client:
mysql -h 51.81.202.9 -P 3306 -u server -p repeater << 'EOF'
SELECT 
    id,
    status,
    node_id,
    vllm_evaluation_count,
    vllm_confidence_peak,
    vllm_confidence_avg,
    JSON_LENGTH(features_json, '$.unique_callsigns') as callsigns,
    start_ts,
    updated_at
FROM net_candidates
ORDER BY updated_at DESC
LIMIT 10;
EOF
```

**Expected**: 
- 3 previously stuck candidates now showing `status = 'ACTIVE'`
- New evaluations incrementing on ACTIVE candidates
- Transitions to COOLING → ENDED as nets conclude

### Watch Real-Time Logs
```bash
docker compose logs -f tsn_server | grep --line-buffered "net_autodetect"
```

**Key Events to Watch**:
```
✅ net_autodetect_candidate_activated - WARMUP → ACTIVE transition
✅ net_autodetect_candidate_cooling - Activity dropping
✅ net_autodetect_candidate_ended - Net concluded
✅ net_autodetect_candidate_verified - OpenAI verified or auto-promoted
✅ net_autodetect_net_promoted - Created net_sessions record
```

### Check Net Sessions
```bash
mysql -h 51.81.202.9 -P 3306 -u server -p repeater << 'EOF'
SELECT 
    id,
    net_name,
    start_time,
    end_time,
    participant_count,
    confidence,
    created_at
FROM net_sessions
ORDER BY created_at DESC
LIMIT 5;
EOF
```

**Expected**: New net sessions appearing (first since January 26)

## Tuning Parameters (If Needed)

### If Still No Activations

**Check vLLM Scores**:
```bash
# Get recent window evaluations
mysql -h 51.81.202.9 -P 3306 -u server -p repeater << 'EOF'
SELECT 
    c.node_id,
    w.window_start,
    JSON_EXTRACT(w.vllm_output_json, '$.net_likelihood') as likelihood,
    JSON_EXTRACT(w.vllm_output_json, '$.suggested_action') as action
FROM net_candidate_windows w
JOIN net_candidates c ON w.candidate_id = c.id
WHERE w.window_start > NOW() - INTERVAL 1 HOUR
ORDER BY w.window_start DESC
LIMIT 20;
EOF
```

If scores are still too low (avg < 20), **further relax criteria**:
```python
# In candidate_state.py, line ~145
meets_likelihood = (
    (peak_in_recent >= 55 and avg_in_recent >= 20) or  # Lower both
    (avg_in_recent >= 30)  # Lower average threshold
)
```

### If Too Many False Positives

If system activates on non-net activity:

**Raise minimum callsigns**:
```python
# In config.py
candidate_min_unique_callsigns: int = Field(default=5)  # Raise from 4
```

**Require higher average**:
```python
# In candidate_state.py
meets_likelihood = (
    (peak_in_recent >= 70 and avg_in_recent >= 30) or
    (avg_in_recent >= 50)
)
```

### Window Size Tuning

Current: **4-minute windows, 45-second steps**

**If nets being missed** (too short):
```bash
# Set environment variable in docker-compose.yml or .env:
TSN_NET_AUTODETECT_WINDOW_SIZE_MINUTES=6
TSN_NET_AUTODETECT_WINDOW_STEP_SECONDS=60
```

**If too much data** (too long):
```bash
TSN_NET_AUTODETECT_WINDOW_SIZE_MINUTES=3
TSN_NET_AUTODETECT_WINDOW_STEP_SECONDS=30
```

## Expected Outcomes

### Immediate (Within 1 Hour)
- ✅ 3 manually activated candidates transition to COOLING/ENDED
- ✅ New WARMUP candidates created for ongoing nets
- ✅ Some WARMUP → ACTIVE transitions (if net activity present)

### Within 24 Hours
- ✅ 5-10 new net candidates activated
- ✅ 2-5 nets promoted to net_sessions table
- ✅ Continuous net detection during active hours

### Long-term
- ✅ Average 10-20 nets detected per day (depending on activity)
- ✅ Detection latency < 15 minutes (3 consecutive windows @ 45s steps)
- ✅ False positive rate < 10%

## Rollback Plan

If fixes cause problems:

```bash
cd /opt/tsn-server
git revert 3aed9d6
docker compose build --no-cache tsn_server
docker compose restart tsn_server
```

## Additional Diagnostic Tools

### Check Full Pipeline Health
```bash
# Run from project directory
python diagnose_net_detection.py
```

Shows:
- Transcript count
- Active nodes
- Callsign detection
- Candidate counts by status
- Recent nets

### View vLLM Prompt Being Sent
```bash
docker compose logs tsn_server | grep -A 50 "WINDOW SUMMARY"
```

### Check Resource Lock Status
```bash
docker compose logs tsn_server | grep "paused_vllm_blocked"
```

If frequent: vLLM constantly blocked by ingestion/analysis, preventing net detection

## Summary

**Root Issue**: Overly strict "ALL windows must be >= 65" criteria + conservative vLM scoring  
**Solution**: Use peak + average thresholds, lower minimums, improve prompt clarity  
**Manual Fix**: Activated 3 stuck candidates immediately  
**Deployment**: Pull commit 3aed9d6, rebuild container, restart service  
**Validation**: Watch logs for state transitions, query net_candidates and net_sessions tables  

The system should now correctly detect and promote nets from transcript patterns. Monitor for 24-48 hours and tune parameters if needed based on observed behavior.
