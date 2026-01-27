# TSN V2 Architecture

## Executive Summary

TSN V2 is a complete ground-up rewrite of The Spoken Network, designed for 24/7 reliability, horizontal scalability, and observability. The system ingests amateur radio transmissions, transcribes them via Whisper, extracts callsigns with vLLM validation, and performs deep analysis using AI.

## Design Principles

1. **Idempotent Operations**: Every stage can be safely retried
2. **Restart-Safe**: System survives crashes mid-pipeline
3. **Observable**: Comprehensive logging, metrics, and health checks
4. **Horizontally Scalable**: Async patterns, queue-based processing
5. **Fail-Safe**: Network drops, server downtime never lose audio

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         NODE SIDE                                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  AllStar     │───▶│  File        │───▶│  Transfer    │      │
│  │  Recording   │    │  Watcher     │    │  Agent       │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│        │                    │                     │              │
│     WAV/PTT            Queue/State           SFTP/HTTPS         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       SERVER SIDE                                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Ingestion   │───▶│ Transcription│───▶│  Callsign    │      │
│  │  Service     │    │  Pipeline    │    │  Extraction  │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│        │                    │                     │              │
│    Safe Store          Whisper GPU          Regex+vLLM          │
│        │                    │                     │              │
│        ▼                    ▼                     ▼              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │    MySQL     │◀───│ vLLM Analysis│◀───│  Topic       │      │
│  │  Database    │    │  Engine      │    │  Extractor   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

## Key Improvements Over Original

### 1. **Async Architecture**
- **Original**: Synchronous polling loops, subprocess.run()
- **New**: asyncio-based, concurrent task processing
- **Benefit**: Process many WAVs per second, non-blocking I/O

### 2. **Queue-Based Processing**
- **Original**: File-system based state (incoming/processed dirs)
- **New**: MySQL-backed work queues with state machine
- **Benefit**: Atomic operations, no race conditions, proper retries

### 3. **Database Evolution**
- **Original**: MySQL with manual connection management
- **New**: Managed MySQL with SQLAlchemy ORM + Alembic migrations
- **Benefit**: ACID guarantees, connection pooling, schema versioning

### 4. **Configuration Management**
- **Original**: Hardcoded values, scattered env vars
- **New**: Pydantic settings with hierarchical configs
- **Benefit**: Type-safe, validated, environment-aware

### 5. **Observability**
- **Original**: print() statements, no metrics
- **New**: Structured logging (JSON), Prometheus metrics, traces
- **Benefit**: Production debugging, performance monitoring, alerting

### 6. **Error Handling**
- **Original**: Try/except with file moves
- **New**: Exponential backoff, circuit breakers, dead-letter queues
- **Benefit**: Transient failures auto-recover, permanent failures isolated

## Data Flow

### Stage 1: Node-Side Capture
```
AllStar → WAV file → Watcher detects → Add to transfer queue → SFTP upload
```

**State Tracking**:
- `pending`: File detected, not yet uploaded
- `uploading`: Transfer in progress
- `uploaded`: Successfully transferred
- `failed`: Transfer failed (with retry count)

**Failure Modes**:
- Network drop during upload → Resume from checkpoint
- Server unreachable → Exponential backoff, local queue grows
- Disk full → Alert, stop recording, purge old files

### Stage 2: Server-Side Ingestion
```
SFTP/HTTPS → Validate → Store → Queue for transcription
```

**Deduplication**: SHA256 hash + filename check
**Storage**: Original WAV retained, metadata in DB
**State**: `received` → `queued_for_transcription`

### Stage 3: Transcription
```
Queue → faster-whisper → Store transcript → Queue for callsign extraction
```

**Backends**: faster-whisper (GPU), whisper.cpp (CPU), OpenAI Whisper
**Idempotency**: Check if transcript exists before processing
**State**: `transcribing` → `transcribed` → `queued_for_extraction`

### Stage 4: Callsign Extraction
```
Transcript → Phonetic correction → Regex extraction → vLLM validation
```

**Two-Stage**:
1. Regex patterns detect candidates
2. vLLM (Qwen2.5-7B) validates each candidate

**Caching**: Validated callsigns cached in memory + DB
**State**: `extracting_callsigns` → `callsigns_extracted` → `queued_for_analysis`

### Stage 5: Deep Analysis
```
Transcript + Callsigns → vLLM → Topics + Summaries + Profiles
```

**Analyses**:
- **Topic Classification**: Antennas, Propagation, Equipment, etc.
- **Net Detection**: Heuristics + AI confirmation
- **Participant Profiling**: Activity patterns, interests, expertise
- **Trend Analysis**: Topic evolution, participation growth

**State**: `analyzing` → `analyzed` → `complete`

## State Machine

```
File States:
pending → uploading → uploaded → received → queued_transcription →
transcribing → transcribed → queued_extraction → extracting →
callsigns_extracted → queued_analysis → analyzing → analyzed → complete

Error States (from any stage):
→ failed_upload / failed_transcription / failed_extraction / failed_analysis
   (with retry_count, max 3 attempts)
```

## Database Schema

### Core Tables

**audio_files**
- `id` (UUID primary key)
- `filename` (unique, indexed)
- `sha256` (unique, for deduplication)
- `file_size` (bytes)
- `duration_sec` (from audio metadata)
- `uploaded_at` (timestamp)
- `node_id` (source node identifier)
- `state` (enum: pending, uploading, uploaded, etc.)
- `retry_count` (for failure recovery)
- `metadata` (JSONB for extensibility)

**transcriptions**
- `id` (UUID primary key)
- `audio_file_id` (FK to audio_files)
- `transcript_text` (TEXT)
- `language` (VARCHAR, default 'en')
- `backend` (enum: faster-whisper, whisper.cpp, openai)
- `confidence` (float, if available)
- `transcribed_at` (timestamp)
- `processing_time_ms` (for metrics)

**callsigns**
- `id` (UUID primary key)
- `callsign` (VARCHAR, unique, uppercase, indexed)
- `validated` (boolean)
- `validation_method` (enum: qrz, vllm, manual)
- `first_seen` (timestamp)
- `last_seen` (timestamp)
- `seen_count` (integer)
- `metadata` (JSONB for QRZ data, etc.)

**callsign_log** (append-only event log)
- `id` (UUID primary key)
- `callsign_id` (FK to callsigns)
- `transcription_id` (FK to transcriptions)
- `detected_at` (timestamp, indexed)
- `confidence` (float, from vLLM)
- `context_snippet` (TEXT, surrounding words)

**callsign_topics**
- `id` (UUID primary key)
- `callsign_id` (FK)
- `transcription_id` (FK)
- `topic` (VARCHAR, enum-like)
- `confidence` (float)
- `excerpt` (TEXT)
- `detected_at` (timestamp)

**net_sessions**
- `id` (UUID primary key)
- `net_name` (VARCHAR)
- `club_name` (VARCHAR, nullable)
- `ncs_callsign_id` (FK to callsigns, nullable)
- `start_time` (timestamp)
- `end_time` (timestamp)
- `duration_sec` (integer)
- `participant_count` (integer)
- `confidence` (float, from detection heuristics)
- `summary` (TEXT, from vLLM)

**net_participations**
- `net_session_id` (FK)
- `callsign_id` (FK)
- `first_seen` (timestamp)
- `last_seen` (timestamp)
- `transmission_count` (integer)
- `estimated_talk_seconds` (integer)
- `checkin_type` (enum: regular, late, io, proxy, etc.)

**callsign_profiles** (aggregated)
- `callsign_id` (FK, primary key)
- `profile_summary` (TEXT, from vLLM)
- `primary_topics` (TEXT[], top interests)
- `activity_score` (float, computed)
- `last_updated` (timestamp)

### Support Tables

**phonetic_corrections**
- `detect` (VARCHAR, indexed)
- `correct` (VARCHAR)

**processing_metrics**
- `id` (UUID)
- `stage` (enum: transcription, extraction, analysis)
- `processing_time_ms` (integer)
- `success` (boolean)
- `error_message` (TEXT, nullable)
- `timestamp` (timestamp)

**system_health**
- `component` (VARCHAR, e.g., 'node-1', 'transcriber', 'vllm')
- `status` (enum: healthy, degraded, down)
- `last_heartbeat` (timestamp)
- `cpu_percent` (float)
- `memory_mb` (integer)
- `metrics` (JSONB)

## Technology Stack

### Node Side
- **Language**: Python 3.11+
- **Framework**: asyncio
- **Libraries**:
  - `watchfiles`: Efficient file watching
  - `httpx`: Async HTTP client
  - `paramiko`: SFTP transfers
  - `pydantic`: Config management
  - `structlog`: Structured logging

### Server Side
- **Language**: Python 3.11+
- **Framework**: FastAPI (for optional REST API)
- **Database**: MySQL 8.0+ / MariaDB 10.6+
- **ORM**: SQLAlchemy 2.0 (async)
- **Migrations**: Alembic
- **Task Queue**: asyncio + MySQL (no Redis dependency)
- **Libraries**:
  - `faster-whisper`: GPU transcription
  - `openai`: vLLM-compatible client
  - `httpx`: Async HTTP
  - `prometheus_client`: Metrics export
  - `structlog`: Logging

### AI/ML
- **Transcription**: faster-whisper (Whisper medium.en on GPU)
- **Callsign Validation**: vLLM with Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4
- **Analysis**: Same vLLM endpoint
- **Fallback**: OpenAI API (gpt-4o-mini)

## Configuration Hierarchy

```python
# Environment → YAML → Defaults
class Settings(BaseSettings):
    # Database
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "tsn"
    db_user: str = "tsn_user"
    db_password: SecretStr
    
    # Node
    node_id: str = "node-1"
    audio_incoming_dir: Path
    audio_archive_dir: Path
    
    # Server
    sftp_host: str
    sftp_port: int = 22
    sftp_username: str
    sftp_key_path: Path
    
    # Transcription
    whisper_model: str = "medium.en"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"
    
    # vLLM
    vllm_base_url: str = "http://192.168.0.104:8001/v1"
    vllm_model: str = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
    vllm_api_key: SecretStr = "sk-no-auth"
    
    # Processing
    max_concurrent_transcriptions: int = 4
    max_concurrent_validations: int = 10
    batch_size: int = 100
    
    # Retry
    max_retries: int = 3
    retry_backoff_base: float = 2.0
    retry_max_wait: int = 300  # 5 minutes
    
    class Config:
        env_prefix = "TSN_"
        env_file = ".env"
```

## Deployment

### Node (Repeater Site)
```bash
# Systemd service
[Unit]
Description=TSN Node Watcher
After=network.target

[Service]
Type=simple
User=tsn
WorkingDirectory=/opt/tsn-node
ExecStart=/opt/tsn-node/venv/bin/python -m tsn_node.watcher
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Server (Centralized)
```bash
# Docker Compose
services:
  tsn_server:
    image: tsn/server:latest
    environment:
      TSN_DB_HOST: ${TSN_DB_HOST}  # MySQL endpoint
      TSN_DB_USER: ${TSN_DB_USER}
      TSN_DB_PASSWORD: ${TSN_DB_PASSWORD}
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    volumes:
      - audio_storage:/data/audio
    depends_on:
      - vllm
  
  tsn_web:
    image: tsn/web:latest
    environment:
      TSN_DB_HOST: ${TSN_DB_HOST}
      TSN_DB_USER: ${TSN_DB_USER}
      TSN_DB_PASSWORD: ${TSN_DB_PASSWORD}
      TSN_WEB_SESSION_SECRET: ${TSN_WEB_SESSION_SECRET}
    ports:
      - "8081:8080"

  vllm:
    image: vllm/vllm-openai:latest
    command: --model Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 --gpu-memory-utilization 0.9
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
  
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prom_data:/prometheus
  
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
```

## Observability

### Logging
```json
{
  "timestamp": "2026-01-21T12:34:56.789Z",
  "level": "info",
  "logger": "tsn.transcriber",
  "event": "transcription_complete",
  "audio_file_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "20260121_123456.wav",
  "processing_time_ms": 2345,
  "transcript_length": 412,
  "backend": "faster-whisper"
}
```

### Metrics (Prometheus)
```
# Counters
tsn_files_ingested_total{node_id="node-1"}
tsn_transcriptions_total{backend="faster-whisper",status="success"}
tsn_callsigns_extracted_total{validation_method="vllm"}
tsn_errors_total{stage="transcription",error_type="timeout"}

# Gauges
tsn_queue_depth{stage="transcription"}
tsn_processing_active{stage="analysis"}
tsn_database_connections{pool="main"}

# Histograms
tsn_processing_duration_seconds{stage="transcription"}
tsn_file_size_bytes
tsn_transcript_length_chars
```

### Health Checks
```
GET /health
{
  "status": "healthy",
  "components": {
    "database": {"status": "up", "latency_ms": 2},
    "vllm": {"status": "up", "latency_ms": 45},
    "disk_space": {"status": "ok", "free_gb": 234}
  },
  "queue_depths": {
    "transcription": 5,
    "extraction": 12,
    "analysis": 3
  }
}
```

## Performance Targets

- **Ingestion**: 100 files/minute sustained
- **Transcription**: 4 concurrent, ~30sec per 1min audio (GPU)
- **Callsign Extraction**: 50 validations/minute (vLLM batch)
- **Analysis**: 20 transcripts/minute (vLLM batch)
- **End-to-End Latency**: < 5 minutes (file arrival → analysis complete)
- **Uptime**: 99.9% (< 9 hours downtime/year)

## Testing Strategy

### Unit Tests
- Database models (CRUD operations)
- Business logic (callsign extraction regex)
- Configuration validation
- Utility functions

### Integration Tests
- File watcher → Transfer agent flow
- Transcription pipeline with real Whisper
- vLLM callsign validation
- Database migrations

### End-to-End Tests
- Node → Server → Analysis full pipeline
- Failure scenarios (network drop, service crash)
- Performance benchmarks

### Property-Based Tests
- Callsign regex patterns (hypothesis)
- State machine transitions (valid paths only)

## Migration from Original

1. **Run both systems in parallel** (shadow mode)
2. **Compare outputs** (transcripts, callsigns, topics)
3. **Gradually shift traffic** (10% → 50% → 100%)
4. **Backfill historical data** (import old transcriptions)
5. **Decommission original** (archive codebase)

## Future Enhancements

- **Multi-node coordination**: Distributed locking for shared resources
- **Real-time streaming**: WebSocket API for live transcripts
- **Advanced analytics**: Network graph analysis, social patterns
- **Mobile app**: iOS/Android client for monitoring
- **Federation**: Multi-repeater networks, inter-site coordination
