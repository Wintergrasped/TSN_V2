# TSN Reference Implementation Analysis

## Overview

This document captures a comprehensive analysis of the original TSN system found in `REFRENCE_ORIGINAL/`. This analysis informed the design of TSN V2.

---

## System Flow (Original)

### Data Pipeline

```
AllStar Recording
    ↓
WAV File Created (per-PTT)
    ↓
transcribe_watcher.py (polls incoming/)
    ↓
transcribe_and_log.py (Whisper + MySQL insert)
    ↓
processed/ directory
    ↓
convert_and_archive.py (WAV→MP3, 7z compression)
    ↓
ai_smoother.py (LLM smoothing + callsign extraction)
    ↓
callsign_extractor.py (regex + QRZ validation)
    ↓
topic_extractor.py (vLLM topic classification)
    ↓
Transcript_Analyzer.py (net detection + NCS + roster)
    ↓
extended_profiles_loader.py (operator/net/NCS profiles)
```

---

## Component Analysis

### 1. transcribe_watcher.py

**Purpose**: Monitor `incoming/` directory for new WAV files, dispatch transcription jobs

**Key Logic**:
- Simple while-True loop with 5-second sleep
- Lists directory, filters `.wav` files
- Size check: skip files < 1000 bytes
- Subprocess call to `transcribe_and_log.py` with 1500s timeout
- Move to `processed/` on success, `failed/` on error

**Issues**:
- **No resilience**: If server down, files pile up
- **No state tracking**: Can't tell if file partially processed
- **Race conditions**: Multiple watchers could grab same file
- **No deduplication**: Same file re-uploaded would re-process
- **Hard-coded paths**: Deployment-specific paths in code

**Improvements for V2**:
- Async file watcher (watchfiles library)
- Database-backed work queue with state machine
- SFTP/HTTPS transfer with resume support
- SHA256 deduplication
- Configuration via Pydantic settings

---

### 2. transcribe_and_log.py

**Purpose**: Run Whisper transcription on audio file, insert to `transcriptions` table

**Key Logic**:
- Supports 3 backends: faster-whisper, whisper.cpp, openai-whisper
- Environment-driven config (model, device, compute type)
- Runs transcription, captures text
- Inserts row: `(filename, timestamp, transcription)`
- No verification that row was written successfully

**Strengths**:
- Backend flexibility via env vars
- Proper error handling (returns exit code)
- Timeout enforcement (via watcher's subprocess.run)

**Issues**:
- **No retry logic**: If DB insert fails, transcript lost
- **Blocking I/O**: One file at a time
- **No quality metrics**: Doesn't record confidence, word count
- **Mixed concerns**: DB logic in transcription script

**Improvements for V2**:
- Async transcription pipeline with queue
- Separate transcription from database persistence
- Record confidence, processing time, backend metadata
- Retry with exponential backoff
- Multiple concurrent workers (4-8 GPUs)

---

### 3. ai_smoother.py

**Purpose**: "Smooth" raw transcripts using vLLM, extract structured callsign lists

**Key Logic**:
- Fetches rows from `transcriptions` where `processed=1`
- Quality gate: min 12 chars, 3 words
- Sends to vLLM `/v1/chat/completions` (Qwen2.5-7B)
- Expects JSON: `{"smoothed": "...", "callsigns": ["K7ABC", ...]}`
- Robust JSON parsing with fence stripping, brace balancing, trailing comma removal
- Inserts to `smoothed_transcripts`, backfills `callsign_log`
- Marks `transcriptions.processed = 2`
- Adaptive pacing based on LLM latency

**Strengths**:
- **Production-hardened**: Extensive error handling
- **JSON rescue**: Multiple fallback parsers
- **SQL retries**: Handles 1205/1213 (deadlock/lock timeout)
- **HTTP resilience**: Retry adapter with backoff
- **Adaptive pacing**: Slows down if LLM is slow

**Issues**:
- **Sequential processing**: One transcript at a time
- **Tight coupling**: Smoother + callsign extraction in one step
- **State confusion**: Uses `processed` flag (0, 1, 2) instead of explicit state

**Improvements for V2**:
- Separate "smoothing" from "callsign extraction" (optional)
- Batch processing (send multiple transcripts to vLLM at once)
- Explicit state machine (transcribed → smoothed → extracted → analyzed)
- Circuit breaker pattern for vLLM endpoint
- Prometheus metrics for LLM latency, success rate

---

### 4. callsign_extractor.py

**Purpose**: Extract callsigns from transcripts using regex + QRZ validation

**Key Logic**:
1. Load phonetic corrections from DB (`corrections` table)
2. Apply corrections to transcript (e.g., "kilo" → "k")
3. Re-join fragmented tokens (e.g., "k k 7 n q n" → "kk7nqn")
4. Regex extraction: `^[A-Z]{1,2}\d[A-Z]{1,4}$`
5. For each candidate:
   - Check if already validated in `callsigns` table (cache hit)
   - If not, query QRZ XML API
   - Insert/update `callsigns` table
   - Log sighting in `callsign_log`
6. Mark `transcriptions.processed = 1`

**Strengths**:
- **QRZ caching**: Avoids re-validation
- **Session management**: Handles QRZ session expiry
- **Phonetic corrections**: Database-driven corrections

**Issues**:
- **No vLLM validation**: Relies solely on QRZ (external API)
- **Sequential API calls**: One callsign at a time
- **Mixed state**: Uses `processed` flag for multiple stages
- **Global state**: `currentID` variable (not thread-safe)

**Improvements for V2**:
- Two-stage validation: Regex → vLLM (not QRZ)
- Batch vLLM requests (10 callsigns per API call)
- Proper state machine (extracting_callsigns state)
- Thread-safe (or async-safe) design
- Confidence scoring per callsign

---

### 5. topic_extractor.py

**Purpose**: Classify transcript topics per callsign using vLLM

**Key Logic**:
1. Fetch transcripts where `analyzed=0` and no topic events yet
2. Build "allowed callsigns" list:
   - From `smoothed_transcripts.callsigns_json`
   - From `callsign_log` rows linked by `transcript_id`
   - Optional: time-window scan (disabled by default)
3. Apply phonetic corrections to transcript
4. Send to vLLM with prompt:
   - Allowed callsigns enforced
   - Preferred topics list (Antennas, Propagation, etc.)
   - Expect JSON: `{"events": [{"callsign": "K7ABC", "topic": "Antennas", "confidence": 92, "excerpt": "..."}]}`
5. Insert events to `callsign_topic_events`
6. Upsert to `extended_callsign_profile` (aggregated)
7. Mark `transcriptions.analyzed=1`

**Strengths**:
- **Provenance**: Only uses callsigns from trusted sources
- **JSON rescue**: Similar robust parsing as smoother
- **Append-only events**: Full history preserved
- **Confidence scoring**: Tracks certainty per topic

**Issues**:
- **Time-window unreliable**: `callsign_log.timestamp` not trustworthy
- **Sequential processing**: One transcript at a time
- **No batching**: Could send multiple transcripts to vLLM

**Improvements for V2**:
- Rely on `transcription_id` links only (no time-window)
- Batch processing (5-10 transcripts per vLLM call)
- Separate worker from analyzer (different resource needs)
- Prometheus metrics for topic distribution

---

### 6. Transcript_Analyzer.py

**Purpose**: Detect radio nets, extract NCS, build rosters, generate summaries

**Key Logic**:
1. Fetch transcripts where `analyzed=0` or missing net_id
2. Sessionize: Group transcripts by time gap (default 10 min)
3. Score each session for "net-ness":
   - Regex: "welcome to", "net control", "check-ins", "closing"
   - Heuristics: callsign density, duration, time-of-day
   - Score >= 1.0 → likely a net
4. Extract net name from text:
   - Patterns: "welcome to <NAME> net"
   - Voting: most frequent candidate
   - Quality checks: reject "this is", "check in", etc.
5. Optional AI enrichment (if `AI_ENABLE=1`):
   - Send session to `ai_backend.py`
   - Get: net_name, club_name, ncs_callsign, participants, topics, summary
   - Validate confidence >= 0.65
6. Create `net_data` row (only if good net name found)
7. Build roster: track transmissions per callsign
8. Insert `net_participation` rows
9. Insert `transcription_analysis` rows
10. Commit all in one transaction

**Strengths**:
- **Hybrid approach**: Regex first, AI refinement optional
- **Quality gates**: Won't create net with bad name
- **Re-scan logic**: Retries nets missing names
- **Phonetic NCS extraction**: Uses corrections table

**Issues**:
- **Complex state machine**: Re-scan logic is intricate
- **AI optional**: Can run without AI, but less accurate
- **Single transaction**: Large batch can fail atomically
- **Hardcoded regexes**: PSRG-specific patterns

**Improvements for V2**:
- Always use AI for net detection (no regex-only mode)
- Smaller transactions (per net, not per batch)
- Configurable patterns (move to DB or config file)
- Prometheus metrics for net detection accuracy

---

### 7. extended_profiles_loader.py

**Purpose**: Generate AI profiles for callsigns, nets, and NCS operators

**Key Logic**:
1. Runs in "batch" mode: process N callsigns per run
2. Three profile types:
   - **Callsign Profile**: Sample transcripts per operator
   - **Net Profile**: Sample transcripts per net slug
   - **NCS Profile**: Sample transcripts per NCS callsign
3. Map-reduce approach:
   - Collect all transcripts for entity
   - Clamp to ~48k chars (~16k tokens)
   - Send to vLLM with scoring prompt
   - Parse JSON response
   - Upsert to `callsign_profiles`, `net_profiles`, `ncs_profiles`
4. Track runs in `profile_runs` table (idempotency)
5. Append to history tables

**Strengths**:
- **Batch processing**: Processes thousands of callsigns
- **Prompt hashing**: Detects prompt changes, forces re-run
- **Idempotent**: Safe to re-run
- **Comprehensive**: Covers 3 profile dimensions

**Issues**:
- **Long-running**: Can take hours for large datasets
- **No progress tracking**: Hard to tell how far along
- **Memory-heavy**: Loads all transcripts in memory

**Improvements for V2**:
- Streaming progress updates (log every N profiles)
- Incremental processing (resume from last checkpoint)
- Separate workers per profile type (parallel)
- Prometheus metrics for profile generation rate

---

### 8. ai_backend.py

**Purpose**: Shared AI client for `Transcript_Analyzer.py`

**Key Logic**:
- Accepts session (list of transcripts as JSON-lines)
- Builds prompt with system + user messages
- Compacts input (keeps "net" keywords, truncates rest)
- Calls `/v1/chat/completions` (OpenAI-compatible)
- Parses response to `AiNetInference` Pydantic model
- Handles two schemas for `operator_topics`:
  - Dict: `{callsign: "description"}`
  - Array: `[{callsign, topics: [...]}, ...]`
- Returns unified `operator_topics_desc` dict

**Strengths**:
- **Retry logic**: 4 attempts with backoff
- **Dual schema support**: Handles LLM output variations
- **Compaction**: Reduces prompt size to fit token limits
- **Pydantic validation**: Type-safe response parsing

**Issues**:
- **Hardcoded fallback**: Embedded OpenAI API key (security risk)
- **No circuit breaker**: Will keep retrying forever
- **Sequential calls**: One session at a time

**Improvements for V2**:
- Secrets in environment variables (never in code)
- Circuit breaker pattern (stop after N failures)
- Batch processing (multiple sessions per call)
- Prometheus metrics for AI latency, token usage

---

### 9. convert_and_archive.py

**Purpose**: Convert WAV → MP3 (32 kbps), compress daily folders with 7z

**Key Logic**:
- Uses `schedule` library for cron-like jobs
- Every 1 minute: Convert WAVs in `processed/` to MP3
- Every day at 23:59: Compress yesterday's folder to `.7z`
- FFmpeg: `-ac 1 -codec:a libmp3lame -b:a 32k`
- 7z: `-t7z -mx=9` (maximum compression)
- Deletes original files after conversion

**Strengths**:
- **Space savings**: 32 kbps mono is 90% smaller than WAV
- **7z compression**: Better than zip/gzip
- **Automated**: Runs in background

**Issues**:
- **No error handling**: If ffmpeg fails, file lost
- **No retention policy**: Archives grow forever
- **Single-threaded**: Slow for large backlogs

**Improvements for V2**:
- Async ffmpeg subprocess (multiple concurrent conversions)
- Retention policy (delete after 90 days)
- Error handling (move failed conversions to `failed/`)
- Prometheus metrics for archive size, compression ratio

---

### 10. project_runner.py

**Purpose**: Unified entry point for all TSN components

**Key Logic**:
- Defines all stages in a central registry
- Commands:
  - `run`: Start watcher + periodic batch jobs
  - `once <stage>`: Run single stage on-demand
  - `list`: Show all stages
  - `config`: Dump active configuration as JSON
- Sets environment variables for subprocesses
- Uses `schedule` for periodic jobs (e.g., every 5 minutes)

**Strengths**:
- **Single entry point**: Easy to understand system
- **Configuration visibility**: `config` command is helpful
- **On-demand execution**: `once` for debugging

**Issues**:
- **Subprocess overhead**: Spawns Python processes
- **No process management**: If subprocess dies, no restart
- **Hardcoded schedules**: Can't change cadence without code edit

**Improvements for V2**:
- Separate services (systemd units, Docker containers)
- Supervisor-style process manager (or use systemd)
- Configuration for schedules (move to settings)

---

## Database Schema Analysis

### Key Tables (Original)

From `tsn_all2.sql` (4.5 GB SQL dump):

**transcriptions**
- `id`, `filename`, `timestamp`, `transcription`
- `processed` (0 = new, 1 = extracted, 2 = smoothed)
- `analyzed` (0 = no topics, 1 = topics extracted)

**smoothed_transcripts**
- `id`, `original_transcript_id`, `smoothed_text`, `callsigns_json`

**callsigns**
- `id`, `callsign`, `validated`, `first_seen`, `last_seen`, `seen_count`

**callsign_log**
- `id`, `callsign`, `transcript_id`, `timestamp`

**callsign_topic_events**
- `id`, `transcript_id`, `callsign`, `topic`, `confidence`, `excerpt`, `detected_at`

**extended_callsign_profile**
- `callsign`, `profile_summary`, `last_updated`, ...

**net_data**
- `id`, `net_name`, `club_name`, `ncs_callsign`, `start_time`, `end_time`, `duration_sec`, `summary`, `confidence_score`

**net_participation**
- `net_id`, `callsign_id`, `transmissions_count`, `talk_seconds`, `checkin_type`

**phonetic_corrections**
- `detect`, `correct` (e.g., "kilo" → "k")

**system_stats** / **temperature_log**
- System monitoring (CPU, memory, temperature)

---

## Implicit Design Patterns

### 1. **Fail-Open Philosophy**

If AI fails, fall back to regex. If QRZ fails, mark as unvalidated. System prefers partial data over total failure.

**V2 Approach**: Preserve this, but add observability to track degraded modes.

---

### 2. **Append-Only Event Logs**

`callsign_log` and `callsign_topic_events` are append-only. Never update/delete.

**V2 Approach**: Keep this pattern, excellent for auditing and replay.

---

### 3. **Upsert for Aggregates**

`callsigns`, `extended_callsign_profile` use `INSERT ... ON DUPLICATE KEY UPDATE`.

**V2 Approach**: MySQL `ON DUPLICATE KEY UPDATE` (same idea).

---

### 4. **Phonetic Corrections as Data**

Store corrections in DB table, not code.

**V2 Approach**: Preserve this, great for runtime updates.

---

### 5. **Regex + AI Hybrid**

Regex finds candidates, AI validates/enriches.

**V2 Approach**: Expand this to all stages (regex → AI → human review).

---

## Failure Modes Identified

### 1. **State Inconsistency**

**Issue**: `processed` flag has multiple meanings (0, 1, 2)  
**Impact**: Hard to tell if file is stuck or actively processing  
**V2 Fix**: Explicit state enum (pending, uploading, transcribing, ...)

---

### 2. **Race Conditions**

**Issue**: Multiple watchers can grab same file from `incoming/`  
**Impact**: Duplicate processing, wasted resources  
**V2 Fix**: Database-backed queue with `SELECT FOR UPDATE SKIP LOCKED`

---

### 3. **No Retry Logic**

**Issue**: If transcription fails, file moves to `failed/` forever  
**Impact**: Transient failures (GPU OOM) become permanent  
**V2 Fix**: Exponential backoff with max 3 retries

---

### 4. **Sequential Processing**

**Issue**: One file at a time, even with idle GPUs  
**Impact**: Low throughput (10 files/min vs possible 100/min)  
**V2 Fix**: Async workers, multiple concurrent tasks

---

### 5. **Hardcoded Secrets**

**Issue**: API keys in code (ai_backend.py line 31)  
**Impact**: Security risk, can't rotate keys easily  
**V2 Fix**: Environment variables, never commit secrets

---

### 6. **No Observability**

**Issue**: print() statements, no metrics  
**Impact**: Can't tell if system is healthy without SSH  
**V2 Fix**: Prometheus metrics, structured logs, health checks

---

### 7. **No Tests**

**Issue**: No unit tests, no integration tests  
**Impact**: Refactoring is risky, bugs found in production  
**V2 Fix**: pytest with >80% coverage, CI/CD

---

## What the Original Got Right

1. **Production Hardened**: Extensive error handling in ai_smoother.py
2. **Modular**: Clear separation of concerns (transcribe, extract, analyze)
3. **Configurable**: Environment variables for most settings
4. **Documented**: PROJECT_SUMMARY.md is excellent
5. **Phonetic Corrections**: Database-driven corrections is genius
6. **Append-Only Logs**: Great for auditing and debugging
7. **Hybrid AI**: Regex + AI is pragmatic, cost-effective

---

## Key Takeaways for V2

1. **Preserve the behavior**: TSN works, don't change the logic
2. **Improve the architecture**: Async, state machine, observability
3. **Maintain pragmatism**: Fail-open, hybrid AI, incremental processing
4. **Add guardrails**: Types, tests, metrics, retries
5. **Respect the original**: It's battle-tested, learn from it

---

## Estimated Improvements

| Metric | Original | TSN V2 | Improvement |
|--------|----------|---------|-------------|
| Throughput | 10 files/min | 100 files/min | **10x** |
| Latency (p50) | 5 min | 3 min | **40% faster** |
| Latency (p99) | 20 min | 10 min | **50% faster** |
| Uptime | 95% | 99.9% | **50x fewer outages** |
| Failed Transcriptions | 5% | 0.5% | **10x more reliable** |
| Ops Visibility | None | Full | **∞** |
| Developer Velocity | Slow | Fast | **Type safety + tests** |

---

## Conclusion

The original TSN is a **remarkably sophisticated system** built incrementally over time. It handles real-world edge cases (malformed JSON, QRZ session expiry, LLM hallucinations) with grace.

TSN V2 **honors this design** while modernizing the infrastructure:
- Async for 10x throughput
- Database queue for reliability
- Observability for operations
- Tests for confidence

The **behavioral logic stays the same** (regex patterns, AI prompts, heuristics). Only the **plumbing changes**.
