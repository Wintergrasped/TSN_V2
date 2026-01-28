# vLLM Keep-Hot System Documentation

## Overview

The TSN V2 system has been enhanced with an **aggressive keep-hot strategy** to eliminate idle time on the vLLM GPU server. This ensures the GPU stays loaded and productive 24/7, maximizing the value of the AI infrastructure.

## Problem Statement

**Before**: The vLLM server would sit idle between incoming audio transcription batches, wasting expensive GPU capacity and creating gaps in analysis throughput.

**After**: The system now continuously feeds vLLM with productive background work, ensuring near-zero idle time while maintaining priority for incoming audio.

---

## Architecture Changes

### 1. Reduced Idle Polling (0.1s vs 2.0s)

**Configuration**: `TSN_ANALYSIS_IDLE_POLL_INTERVAL_SEC=0.1`

- **Before**: 2.0 second sleep between work checks = up to 2s of idle time per cycle
- **After**: 0.1 second sleep = maximum 100ms idle before checking for new work
- **Impact**: 20x faster response to new work, minimal CPU overhead

### 2. Aggressive Backfill Mode

**Configuration**: `TSN_ANALYSIS_AGGRESSIVE_BACKFILL_ENABLED=true`

When enabled, the system **chains multiple background tasks** instead of doing one task then sleeping:

#### Priority Queue (Executed in Order):

1. **Rescue Failed Analysis** (Priority 1)
   - Retries failed analysis files after configurable cooldown
   - Ensures temporary failures don't result in lost data

2. **Transcript Smoothing** (Priority 2)
   - Cleans ASR artifacts from raw transcripts
   - Quick wins that improve downstream analysis quality
   - Processes up to 3 batches in sequence

3. **Refinement Passes** (Priority 3)
   - Re-analyzes completed audio for deeper insights
   - Focuses on audio that never spawned a net session
   - Configurable window (default: 38 hours)

4. **Profile Refresh** (Priority 4)
   - Updates callsign profiles with recent activity
   - Summarizes favorite topics, NCS history, engagement
   - Processes up to 2 profiles per cycle

5. **Overdrive Re-analysis** (Priority 5)
   - Forces completed audio through analysis again
   - Uses full 32k context window (31,500 char budget)
   - Configurable cooldown (default: 12 hours)

6. **Preemptive Profile Generation** (Priority 6)
   - Creates profiles for callsigns with 3+ segments but no profile
   - Proactive analysis for newly active operators

7. **Deep Club Analysis** (Priority 7)
   - Analyzes clubs with recent activity
   - Identifies patterns, schedules, membership trends

**Configuration**: `TSN_ANALYSIS_IDLE_WORK_CHAIN_LIMIT=10`
- Limits consecutive background tasks per cycle to prevent infinite loops
- Default: 10 tasks before yielding to primary queue

---

## Database Logging & Monitoring

### 1. Idle Time Tracking

Every idle period is logged to `processing_metrics` table:

```sql
SELECT 
    SUM(processing_time_ms) / 1000.0 as total_idle_seconds,
    COUNT(*) as idle_events,
    AVG(processing_time_ms) as avg_idle_ms
FROM processing_metrics
WHERE stage = 'analysis_idle'
    AND timestamp >= NOW() - INTERVAL 1 HOUR;
```

**Metadata Captured**:
- `idle_streak`: Number of consecutive idle cycles
- `total_idle_ms`: Cumulative idle time for worker
- `gpu_utilization`: GPU % at time of idle event

### 2. Worker Statistics

Every 5 minutes, each worker logs comprehensive stats to `processing_metrics`:

```sql
SELECT 
    stage,
    metadata_->>'$.worker_id' as worker_id,
    metadata_->>'$.work_percent' as work_pct,
    metadata_->>'$.idle_percent' as idle_pct,
    metadata_->>'$.gpu_utilization' as gpu_pct,
    timestamp
FROM processing_metrics
WHERE stage LIKE 'analysis_worker_%'
ORDER BY timestamp DESC
LIMIT 20;
```

**Metrics Tracked**:
- Total iterations
- Work cycles vs idle cycles
- Work percentage (target: >90%)
- Idle percentage (target: <10%)
- GPU utilization
- Aggressive backfill status

### 3. GPU Utilization Samples

Enhanced `gpu_utilization_samples` table now includes:
- Idle streak counter
- Total accumulated idle time
- Warnings logged when GPU < threshold

```sql
SELECT 
    AVG(utilization_pct) as avg_util,
    MIN(utilization_pct) as min_util,
    MAX(utilization_pct) as max_util,
    COUNT(*) as samples
FROM gpu_utilization_samples
WHERE created_at >= NOW() - INTERVAL 1 HOUR;
```

### 4. AI Run Logs

Every vLLM call is logged to `ai_run_logs` table:

```sql
SELECT 
    pass_label,
    backend,
    COUNT(*) as calls,
    AVG(latency_ms) as avg_latency,
    AVG(gpu_utilization_pct) as avg_gpu,
    SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes,
    SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as failures
FROM ai_run_logs
WHERE created_at >= NOW() - INTERVAL 1 HOUR
GROUP BY pass_label, backend
ORDER BY calls DESC;
```

**Fields Logged**:
- Backend (vllm, openai)
- Model name
- Pass label (analysis, smoothing, profile_refresh, overdrive, etc.)
- Success/failure status
- Full prompt text
- Full response text
- Prompt/completion/total tokens
- Latency in milliseconds
- GPU utilization at time of call
- Associated audio file IDs
- Custom metadata per pass type

---

## Web Dashboard Integration

The web portal now displays real-time GPU and idle metrics:

### System Health Widget

**New Components**:

1. **vllm_gpu**
   - Status: healthy (≥75%), degraded (50-75%), down (<50%)
   - Average utilization over last hour
   - Min/max utilization
   - Sample count
   - Last sample timestamp

2. **analysis_workers**
   - Status: healthy (≤10% idle), degraded (10-30% idle), down (>30% idle)
   - Total idle time (seconds)
   - Average work percentage
   - Average idle percentage
   - Idle event count

### Query Endpoints

Access via `/api/health` or dashboard widgets:

```python
GET /api/health
{
  "components": [
    {
      "component": "vllm_gpu",
      "status": "healthy",
      "metrics": {
        "avg_utilization_pct": 87.3,
        "min_utilization_pct": 45.2,
        "max_utilization_pct": 98.7,
        "sample_count": 240,
        "window_hours": 1
      }
    },
    {
      "component": "analysis_workers",
      "status": "healthy",
      "metrics": {
        "total_idle_seconds": 124.5,
        "avg_work_percent": 94.2,
        "avg_idle_percent": 5.8,
        "idle_event_count": 42,
        "window_hours": 1
      }
    }
  ]
}
```

---

## Configuration Reference

### Environment Variables (.env)

```bash
# Idle polling (reduced to 0.1s for aggressive keep-hot)
TSN_ANALYSIS_IDLE_POLL_INTERVAL_SEC=0.1

# Enable aggressive multi-task chaining
TSN_ANALYSIS_AGGRESSIVE_BACKFILL_ENABLED=true

# Maximum consecutive background tasks
TSN_ANALYSIS_IDLE_WORK_CHAIN_LIMIT=10

# GPU monitoring
TSN_ANALYSIS_GPU_WATCH_ENABLED=true
TSN_ANALYSIS_GPU_LOW_UTILIZATION_PCT=65.0
TSN_ANALYSIS_GPU_CHECK_INTERVAL_SEC=15.0
TSN_ANALYSIS_GPU_SATURATION_THRESHOLD_PCT=95.0

# Overdrive settings (heavy re-analysis)
TSN_ANALYSIS_GPU_OVERDRIVE_BUDGET=31500
TSN_ANALYSIS_OVERDRIVE_WINDOW_HOURS=168
TSN_ANALYSIS_OVERDRIVE_BATCH_SIZE=6
TSN_ANALYSIS_OVERDRIVE_COOLDOWN_HOURS=12

# Refinement settings (medium re-analysis)
TSN_ANALYSIS_REFINEMENT_WINDOW_HOURS=38
TSN_ANALYSIS_REFINEMENT_BATCH_SIZE=8
TSN_ANALYSIS_MAX_REFINEMENT_PASSES=10

# Profile refresh settings
TSN_ANALYSIS_PROFILE_REFRESH_HOURS=12
TSN_ANALYSIS_PROFILE_CONTEXT_HOURS=640
TSN_ANALYSIS_PROFILE_BATCH_SIZE=3
TSN_ANALYSIS_PROFILE_MIN_SEEN_COUNT=5

# Transcript smoothing settings
TSN_ANALYSIS_TRANSCRIPT_SMOOTHING_ENABLED=true
TSN_ANALYSIS_TRANSCRIPT_SMOOTHING_BATCH_SIZE=4

# Failed analysis rescue
TSN_ANALYSIS_FAILED_ANALYSIS_RESCUE_MINUTES=10
TSN_ANALYSIS_FAILED_ANALYSIS_RESCUE_BATCH=25
TSN_ANALYSIS_FAILED_ANALYSIS_RETRY_LIMIT=6
```

### Key Tuning Parameters

| Parameter | Recommended | Purpose |
|-----------|------------|---------|
| `IDLE_POLL_INTERVAL_SEC` | 0.1 | Minimize idle gaps between work |
| `AGGRESSIVE_BACKFILL_ENABLED` | true | Enable multi-task chaining |
| `IDLE_WORK_CHAIN_LIMIT` | 10 | Prevent infinite background loops |
| `GPU_LOW_UTILIZATION_PCT` | 65.0 | Trigger backfill below this % |
| `OVERDRIVE_BATCH_SIZE` | 6 | Balance load vs queue churn |
| `REFINEMENT_BATCH_SIZE` | 8 | Medium batch for quality passes |

---

## Performance Monitoring

### Key Metrics to Watch

1. **GPU Utilization** (Target: >85% sustained)
   ```sql
   SELECT AVG(utilization_pct) 
   FROM gpu_utilization_samples 
   WHERE created_at >= NOW() - INTERVAL 1 DAY;
   ```

2. **Worker Idle Time** (Target: <10%)
   ```sql
   SELECT 
       stage,
       AVG(JSON_EXTRACT(metadata_, '$.idle_percent')) as avg_idle_pct
   FROM processing_metrics
   WHERE stage LIKE 'analysis_worker_%'
       AND timestamp >= NOW() - INTERVAL 1 DAY
   GROUP BY stage;
   ```

3. **Backfill Effectiveness**
   ```sql
   SELECT 
       pass_label,
       COUNT(*) as calls,
       AVG(latency_ms) as avg_latency
   FROM ai_run_logs
   WHERE pass_label IN ('smoothing', 'refinement', 'overdrive', 'profile_refresh')
       AND created_at >= NOW() - INTERVAL 1 DAY
   GROUP BY pass_label
   ORDER BY calls DESC;
   ```

4. **Primary vs Background Work Balance**
   ```sql
   SELECT 
       CASE 
           WHEN pass_label = 'analysis' THEN 'primary'
           ELSE 'background'
       END as work_type,
       COUNT(*) as calls,
       SUM(total_tokens) as total_tokens
   FROM ai_run_logs
   WHERE created_at >= NOW() - INTERVAL 1 DAY
   GROUP BY work_type;
   ```

### Expected Behavior

**Healthy System**:
- GPU utilization: 85-95% sustained
- Worker idle percentage: <10%
- Backfill work: 30-50% of total vLLM calls
- Primary analysis: Always takes priority when available
- Idle periods: <5 seconds between work

**Warning Signs**:
- GPU utilization: <70% for extended periods
- Worker idle percentage: >20%
- Long idle streaks (>10 seconds)
- Backfill work: >70% of total (may indicate not enough incoming audio)

---

## Troubleshooting

### GPU Still Shows Low Utilization

1. Check if `nvidia-smi` is available:
   ```bash
   nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits
   ```

2. Verify aggressive backfill is enabled:
   ```bash
   grep AGGRESSIVE_BACKFILL .env
   ```

3. Check worker logs for backfill activity:
   ```bash
   docker logs tsn_analyzer | grep aggressive_backfill
   ```

4. Verify work is being queued:
   ```sql
   SELECT state, COUNT(*) 
   FROM audio_files 
   GROUP BY state;
   ```

### Workers Spending Too Much Time Idle

1. Reduce idle poll interval further (try 0.05s):
   ```bash
   TSN_ANALYSIS_IDLE_POLL_INTERVAL_SEC=0.05
   ```

2. Increase chain limit:
   ```bash
   TSN_ANALYSIS_IDLE_WORK_CHAIN_LIMIT=15
   ```

3. Increase batch sizes:
   ```bash
   TSN_ANALYSIS_OVERDRIVE_BATCH_SIZE=10
   TSN_ANALYSIS_REFINEMENT_BATCH_SIZE=12
   ```

4. Extend overdrive window:
   ```bash
   TSN_ANALYSIS_OVERDRIVE_WINDOW_HOURS=336  # 14 days
   ```

### Too Much Database Write Load

If logging creates DB pressure:

1. Reduce GPU check frequency:
   ```bash
   TSN_ANALYSIS_GPU_CHECK_INTERVAL_SEC=30
   ```

2. Worker stats log every 10 minutes instead of 5 (code change needed)

3. Implement log aggregation/batching (future enhancement)

---

## Future Enhancements

1. **Predictive Work Scheduling**
   - ML model to predict incoming audio patterns
   - Pre-load context windows before audio arrives

2. **Multi-GPU Support**
   - Distribute work across multiple vLLM instances
   - Load balance based on queue depth

3. **Priority-Based Backfill**
   - Assign scores to background tasks
   - Always pick highest-value idle work

4. **Real-Time Dashboard Alerts**
   - WebSocket updates for GPU utilization
   - Alert when idle time exceeds thresholds

5. **Automatic Tuning**
   - Self-adjust batch sizes based on observed latency
   - Dynamic chain limits based on queue pressure

---

## Summary

The vLLM keep-hot system transforms idle GPU time into productive analysis work through:

- **20x faster** work detection (0.1s vs 2.0s polling)
- **7 types** of background work to keep GPU loaded
- **Comprehensive logging** to MySQL for monitoring
- **Real-time dashboard** metrics for GPU and idle time
- **Priority queue** ensures incoming audio always processed first
- **Configurable tuning** for different workload patterns

**Result**: Near-zero idle time, maximum ROI on GPU infrastructure, deeper analysis quality across all data.
