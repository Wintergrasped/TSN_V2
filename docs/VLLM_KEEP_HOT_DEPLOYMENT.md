# vLLM Keep-Hot System - Deployment Guide

## Overview

This guide covers deploying the vLLM keep-hot system enhancements to your TSN V2 installation.

---

## Pre-Deployment Checklist

### 1. Database Compatibility

The system uses existing tables - no schema migrations required:
- ✅ `processing_metrics` - Already exists
- ✅ `gpu_utilization_samples` - Already exists
- ✅ `ai_run_logs` - Already exists
- ✅ `audio_files` - Already exists

### 2. Backup Current Configuration

```bash
# Backup your current .env file
cp .env .env.backup.$(date +%Y%m%d_%H%M%S)

# Backup current docker containers (if using Docker)
docker-compose ps > deployment_backup_containers.txt
docker images > deployment_backup_images.txt
```

### 3. Check vLLM Server Status

```bash
# Verify vLLM is accessible
curl -X POST http://192.168.0.104:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4","messages":[{"role":"user","content":"test"}],"max_tokens":10}'

# Check GPU availability
nvidia-smi
```

---

## Deployment Steps

### Step 1: Update Configuration Files

#### Update `.env` file:

```bash
# Add new configuration parameters to your .env file
cat >> .env << 'EOF'

# ============================================================================
# VLLM KEEP-HOT SYSTEM (NEW)
# ============================================================================

# Aggressive idle elimination (reduced from 2.0s to 0.1s)
TSN_ANALYSIS_IDLE_POLL_INTERVAL_SEC=0.1

# Enable multi-task chaining for continuous GPU load
TSN_ANALYSIS_AGGRESSIVE_BACKFILL_ENABLED=true

# Maximum background tasks per cycle (prevents infinite loops)
TSN_ANALYSIS_IDLE_WORK_CHAIN_LIMIT=10

EOF
```

Or manually edit `.env` and update these lines:

```diff
- TSN_ANALYSIS_IDLE_POLL_INTERVAL_SEC=2.0
+ TSN_ANALYSIS_IDLE_POLL_INTERVAL_SEC=0.1

# Add these new lines:
+ TSN_ANALYSIS_AGGRESSIVE_BACKFILL_ENABLED=true
+ TSN_ANALYSIS_IDLE_WORK_CHAIN_LIMIT=10
```

### Step 2: Update Python Code

The following files have been modified:

1. **`tsn_common/config.py`**
   - Added `aggressive_backfill_enabled` setting
   - Added `idle_work_chain_limit` setting
   - Changed default `idle_poll_interval_sec` from 2.0 to 0.1

2. **`tsn_server/analyzer.py`**
   - Added `_aggressive_backfill_work()` method
   - Added `_queue_preemptive_profiles()` method
   - Added `_queue_club_deep_analysis()` method
   - Added `_record_idle_period()` method
   - Added `_track_idle_time()` and `_reset_idle_tracking()` methods
   - Enhanced `_record_gpu_sample()` with idle tracking
   - Enhanced `run_worker()` with comprehensive statistics
   - Added `_record_worker_statistics()` method
   - Modified `process_one()` to use aggressive backfill

3. **`web/services/dashboard.py`**
   - Added `get_gpu_statistics()` function
   - Added `get_idle_statistics()` function
   - Enhanced `get_system_health()` to include GPU and idle metrics

If you've made local modifications to these files, you'll need to merge carefully. Otherwise, simply pull the updated code.

### Step 3: Verify Code Update

```bash
# Check that new configuration options are present
grep -r "aggressive_backfill_enabled" tsn_common/config.py
grep -r "_aggressive_backfill_work" tsn_server/analyzer.py
grep -r "get_gpu_statistics" web/services/dashboard.py

# Expected: All three should return matches
```

### Step 4: Restart Services

#### Option A: Docker Deployment

```bash
# Pull latest changes (if using git)
git pull origin main

# Rebuild containers
docker-compose build tsn_analyzer tsn_web

# Restart services with new configuration
docker-compose down
docker-compose up -d

# Verify services started
docker-compose ps
docker-compose logs -f tsn_analyzer | head -50
```

#### Option B: Systemd Services

```bash
# Restart analyzer service
sudo systemctl restart tsn-server

# Restart web portal
sudo systemctl restart tsn-web

# Check status
sudo systemctl status tsn-server
sudo systemctl status tsn-web

# View logs
sudo journalctl -u tsn-server -f --since "5 minutes ago"
```

#### Option C: Manual Python

```bash
# Stop existing processes
pkill -f tsn_orchestrator
pkill -f "uvicorn.*web.main"

# Start orchestrator (analyzer + server)
cd /path/to/TSN_V2
python -m tsn_orchestrator &

# Start web portal
cd /path/to/TSN_V2
uvicorn web.main:app --host 0.0.0.0 --port 8081 &
```

---

## Verification

### Step 1: Check Configuration Loaded

```bash
# Docker
docker-compose logs tsn_analyzer | grep "analyzer_initialized"

# Should show:
# analyzer_initialized ... aggressive_backfill=true
```

Look for these log entries:
```json
{
  "event": "analyzer_initialized",
  "vllm_url": "http://192.168.0.104:8001/v1",
  "model": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
  "analysis_workers": 2,
  "batch_size": 4
}
```

### Step 2: Monitor Idle Time Reduction

#### Before (baseline):
```bash
# Check idle cycles over 10 minutes
docker-compose logs tsn_analyzer --since 10m | grep "analysis_worker_statistics" | tail -1
```

Expected before: `idle_percent: 40-60%` (with 2.0s polling)

#### After (aggressive mode):
Expected after: `idle_percent: 5-15%` (with 0.1s polling + backfill)

### Step 3: Verify Background Work Execution

```bash
# Look for aggressive backfill logs
docker-compose logs tsn_analyzer --since 10m | grep "aggressive_backfill"

# Expected output:
# aggressive_backfill_rescued_failed: count=5
# aggressive_backfill_refinement: count=8
# aggressive_backfill_overdrive: count=6
# aggressive_backfill_completed: total_work_items=25
```

### Step 4: Check GPU Utilization

```bash
# Monitor GPU utilization
watch -n 2 nvidia-smi

# Check logged samples in database
mysql -u server -p repeater << 'EOF'
SELECT 
    AVG(utilization_pct) as avg_util,
    MIN(utilization_pct) as min_util,
    MAX(utilization_pct) as max_util,
    COUNT(*) as samples
FROM gpu_utilization_samples
WHERE created_at >= NOW() - INTERVAL 1 HOUR;
EOF
```

Expected: Average utilization should increase from ~50-60% to 85-95%

### Step 5: Verify Database Logging

```bash
# Check that idle periods are being logged
mysql -u server -p repeater << 'EOF'
SELECT 
    COUNT(*) as idle_events,
    SUM(processing_time_ms) / 1000.0 as total_idle_seconds,
    AVG(processing_time_ms) as avg_idle_ms
FROM processing_metrics
WHERE stage = 'analysis_idle'
    AND timestamp >= NOW() - INTERVAL 1 HOUR;
EOF
```

Expected: Events logged, but total_idle_seconds should be low (<300s per hour target)

```bash
# Check worker statistics
mysql -u server -p repeater << 'EOF'
SELECT 
    stage,
    JSON_EXTRACT(metadata_, '$.work_percent') as work_pct,
    JSON_EXTRACT(metadata_, '$.idle_percent') as idle_pct,
    JSON_EXTRACT(metadata_, '$.gpu_utilization') as gpu_pct,
    timestamp
FROM processing_metrics
WHERE stage LIKE 'analysis_worker_%'
    AND timestamp >= NOW() - INTERVAL 1 HOUR
ORDER BY timestamp DESC
LIMIT 10;
EOF
```

### Step 6: Check Web Dashboard

```bash
# Access health endpoint
curl http://localhost:8081/api/health | jq '.components[] | select(.component=="vllm_gpu")'
curl http://localhost:8081/api/health | jq '.components[] | select(.component=="analysis_workers")'
```

Expected JSON response:
```json
{
  "component": "vllm_gpu",
  "status": "healthy",
  "metrics": {
    "avg_utilization_pct": 87.5,
    "min_utilization_pct": 45.0,
    "max_utilization_pct": 98.2,
    "sample_count": 240
  }
}

{
  "component": "analysis_workers",
  "status": "healthy",
  "metrics": {
    "total_idle_seconds": 124.5,
    "avg_work_percent": 94.2,
    "avg_idle_percent": 5.8,
    "idle_event_count": 42
  }
}
```

---

## Performance Tuning

### If GPU Still Shows Low Utilization (<80%)

1. **Increase work chain limit**:
   ```bash
   TSN_ANALYSIS_IDLE_WORK_CHAIN_LIMIT=15
   ```

2. **Increase batch sizes**:
   ```bash
   TSN_ANALYSIS_OVERDRIVE_BATCH_SIZE=10
   TSN_ANALYSIS_REFINEMENT_BATCH_SIZE=12
   TSN_ANALYSIS_PROFILE_BATCH_SIZE=5
   ```

3. **Extend overdrive window**:
   ```bash
   TSN_ANALYSIS_OVERDRIVE_WINDOW_HOURS=336  # 14 days
   ```

4. **Reduce overdrive cooldown**:
   ```bash
   TSN_ANALYSIS_OVERDRIVE_COOLDOWN_HOURS=6
   ```

### If Workers Show >20% Idle Time

1. **Further reduce polling interval** (careful - increases CPU):
   ```bash
   TSN_ANALYSIS_IDLE_POLL_INTERVAL_SEC=0.05
   ```

2. **Check for transcription bottleneck**:
   ```sql
   SELECT state, COUNT(*) 
   FROM audio_files 
   GROUP BY state;
   ```
   
   If many files stuck in `QUEUED_TRANSCRIPTION`, increase transcription workers.

3. **Lower GPU threshold for backfill trigger**:
   ```bash
   TSN_ANALYSIS_GPU_LOW_UTILIZATION_PCT=75.0
   ```

### If Database Writes Too High

1. **Reduce GPU sampling frequency**:
   ```bash
   TSN_ANALYSIS_GPU_CHECK_INTERVAL_SEC=30
   ```

2. **Increase worker stats interval** (requires code change):
   In `analyzer.py`, change `if now - last_stats_log >= 300:` to 600 (10 minutes)

---

## Rollback Procedure

If you need to revert to the old behavior:

### Quick Rollback (Configuration Only)

```bash
# Edit .env file
TSN_ANALYSIS_IDLE_POLL_INTERVAL_SEC=2.0
TSN_ANALYSIS_AGGRESSIVE_BACKFILL_ENABLED=false

# Restart services
docker-compose restart tsn_analyzer tsn_web
```

### Full Rollback (Code + Config)

```bash
# Restore backup
cp .env.backup.YYYYMMDD_HHMMSS .env

# Checkout previous code version
git log --oneline | head -10  # Find commit before changes
git checkout <previous_commit_hash>

# Rebuild and restart
docker-compose down
docker-compose build
docker-compose up -d
```

---

## Monitoring and Alerts

### Set Up Daily Reports

Create a cron job to email daily GPU utilization:

```bash
# Add to crontab
0 8 * * * /path/to/TSN_V2/scripts/daily_gpu_report.sh | mail -s "TSN GPU Daily Report" admin@example.com
```

Create `scripts/daily_gpu_report.sh`:
```bash
#!/bin/bash
mysql -u server -pPASSWORD repeater << 'EOF'
SELECT 
    DATE(created_at) as date,
    AVG(utilization_pct) as avg_util,
    MIN(utilization_pct) as min_util,
    MAX(utilization_pct) as max_util,
    COUNT(*) as samples
FROM gpu_utilization_samples
WHERE created_at >= NOW() - INTERVAL 7 DAY
GROUP BY DATE(created_at)
ORDER BY date DESC;

SELECT 
    stage,
    AVG(JSON_EXTRACT(metadata_, '$.work_percent')) as avg_work_pct,
    AVG(JSON_EXTRACT(metadata_, '$.idle_percent')) as avg_idle_pct
FROM processing_metrics
WHERE stage LIKE 'analysis_worker_%'
    AND timestamp >= NOW() - INTERVAL 7 DAY
GROUP BY stage;
EOF
```

### Grafana Dashboard (Optional)

If you have Grafana + Prometheus:

1. Add Prometheus MySQL exporter
2. Import dashboard template (create custom)
3. Monitor these queries:
   - `rate(gpu_utilization_samples[5m])`
   - `rate(processing_metrics{stage="analysis_idle"}[5m])`
   - `rate(ai_run_logs[5m]) by (pass_label)`

---

## Support and Troubleshooting

### Common Issues

#### 1. "aggressive_backfill_work not found" error

**Cause**: Old code still running
**Solution**: 
```bash
git pull origin main
docker-compose build tsn_analyzer
docker-compose restart tsn_analyzer
```

#### 2. Database errors about missing columns

**Cause**: Shouldn't happen - we use existing tables
**Solution**: Check MySQL server version and table schemas:
```sql
SHOW CREATE TABLE processing_metrics;
SHOW CREATE TABLE gpu_utilization_samples;
SHOW CREATE TABLE ai_run_logs;
```

#### 3. GPU utilization not showing in dashboard

**Cause**: `nvidia-smi` not available or GPU not accessible
**Solution**:
```bash
# Test from analyzer container
docker exec -it tsn_analyzer nvidia-smi

# Verify GPU_WATCH_ENABLED
docker exec -it tsn_analyzer env | grep GPU_WATCH
```

#### 4. Workers not starting

**Cause**: Configuration syntax error
**Solution**:
```bash
# Check logs for Python exceptions
docker-compose logs tsn_analyzer | grep -i error | tail -20

# Verify .env syntax
python << 'EOF'
from tsn_common.config import get_settings
settings = get_settings()
print(f"Aggressive backfill: {settings.analysis.aggressive_backfill_enabled}")
print(f"Idle poll: {settings.analysis.idle_poll_interval_sec}")
EOF
```

### Getting Help

1. Check documentation: `docs/VLLM_KEEP_HOT_SYSTEM.md`
2. Review logs: `docker-compose logs tsn_analyzer --tail=100`
3. Query metrics: Run SQL queries from verification section
4. Open GitHub issue with:
   - Full error logs
   - Configuration (.env with secrets redacted)
   - Database schema versions
   - GPU hardware info

---

## Success Criteria

After 24 hours of operation, you should observe:

✅ **GPU Utilization**: Average >85% (up from ~50-60%)
✅ **Worker Idle Time**: <10% per hour (down from 40-60%)
✅ **Background Work**: 30-50% of total vLLM calls
✅ **Response Time**: Primary analysis still <5 seconds to start
✅ **Database Health**: No excessive write load (monitor slow query log)
✅ **Dashboard**: Real-time metrics visible and updating

**Congratulations!** Your vLLM GPU is now running hot 24/7, maximizing ROI on AI infrastructure while maintaining priority for real-time audio analysis.
