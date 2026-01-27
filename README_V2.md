# The Spoken Network v2.0

**Amateur Radio Intelligence System** - Automated transcription, analysis, and archival of repeater audio.

## 🎯 Overview

TSN v2 is a complete ground-up rewrite of The Spoken Network, designed for:
- **10x Performance**: Async architecture with concurrent processing (100 files/min vs 10/min)
- **Production Reliability**: 99.9% uptime target with automatic retry and failure recovery
- **Scalability**: MySQL-backed queue, horizontal scaling support
- **Observability**: Structured logging, Prometheus metrics, health checks

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install package
pip install -e .

# Optional: GPU support
pip install -e ".[gpu]"
```

### Configuration

```bash
# Copy example config
cp .env.example .env

# Edit with your settings
nano .env
```

**Minimum Required:**
```bash
TSN_DB_PASSWORD=your_secure_password
TSN_NODE_ENABLED=true  # or false
TSN_SERVER_ENABLED=true  # or false
TSN_STORAGE_BASE_PATH=/path/to/storage
```

### Initialize & Run

```bash
# Initialize database
tsn init-db

# Run all services
tsn-orchestrator
```

### Monitor

```bash
# CLI status
tsn status

# Health check
curl http://localhost:8080/health

# Metrics
curl http://localhost:8080/metrics
```

## 🗃️ Legacy Schema Migration (MySQL)

## 🛰️ Web Portal Upgrades

Recent work focuses on the operator-facing FastAPI portal so Net Control can work entirely from a browser, without touching the raw database:
- **QRZ-only filters**: Callsign indexes only surface operators that the QRZ XML API confirms, keeping speculative regex hits in a separate holding queue.
- **Deep-linkable profiles**: `/callsigns/{id}` and `/clubs/{slug}` now surface transcripts, recent activity, membership graphs, and editable operator notes tied to the logged-in user.
- **AI-assisted summaries**: vLLM (with OpenAI fallback) produces quick-read blurbs for hot callsigns, trending nets, and suggested aliases that can be merged in one click.
- **Net Control cockpit**: Dedicated page streams live check-ins, lets you start/stop ad-hoc Net Control Sessions, capture manual check-ins, and export CSV logs for after-action reports.
- **Contextual customization**: Operators can pin watchlist notes, pre-fill cue cards, and capture action items per callsign/club without leaving the dashboard.


If you are upgrading an existing MySQL deployment (for example the
`repeater` schema on 51.81.202.9) you **must** migrate the legacy tables
before running `tsn_common.db_init`. The new ORM expects UUID primary keys
and different column layouts, so attempting to reuse the old tables leads to
`FOREIGN KEY ... incorrectly formed` errors.

We ship an automated migrator (`python -m tsn_common.migrations.legacy_uuid_migrator`)
that rewrites the legacy INT identifiers **in place** without dropping your
tables:

1. Adds helper UUID columns to every legacy table and backfills them with
  deterministic UUIDv4 values.
2. Updates all foreign key columns (net sessions, participations, logs, etc.)
  so they reference the new UUIDs.
3. Drops and recreates the relevant constraints/indexes to match the v2 ORM
  definitions.
4. Records the migration version (`tsn_schema_migrations`) to avoid reruns.

> ⚠️ **Important:** the script still requires exclusive access to the schema
> while it runs. Back up before any migration, and verify `.env` points at the
> intended database.

### Migration workflow

```bash
# 1. Convert INT ids to UUIDs in-place
docker compose run --rm tsn_server python -m tsn_common.migrations.legacy_uuid_migrator

# 2. Recreate tables + seed defaults (idempotent)
docker compose run --rm tsn_server python -m tsn_common.db_init

# 3. Bring the stack up normally
docker compose up -d
```

The migrator is safe to keep in your automation—it exits immediately after it
detects the `tsn_schema_migrations` record (meaning the upgrade already
happened).

## 📋 CLI Commands

```bash
# Database
tsn init-db              # Create tables and seed data

# Monitoring
tsn status               # Show system status
tsn list-callsigns       # List callsigns
tsn profile W1ABC        # View callsign profile
tsn list-nets            # Show recent net sessions

# Maintenance
tsn reprocess <uuid>     # Reprocess a file
tsn clean-failed         # Clean up failed entries
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     REPEATER SITE (Node)                    │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐            │
│  │ Audio    │────▶│  File    │────▶│  SFTP    │────────┐   │
│  │ Capture  │     │  Watcher │     │  Transfer│        │   │
│  └──────────┘     └──────────┘     └──────────┘        │   │
└─────────────────────────────────────────────────────────┼───┘
                                                          │
                                                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    CENTRAL SERVER                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │Ingestion │─▶│Whisper   │─▶│Callsign  │─▶│ vLLM     │   │
│  │          │  │Transcribe│  │Extractor │  │ Analysis │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│         │                                          │        │
│         └──────────────┬───────────────────────────┘        │
│                        ▼                                    │
│                ┌───────────────┐                            │
│                │    MySQL      │                            │
│                │   Database    │                            │
│                └───────────────┘                            │
└─────────────────────────────────────────────────────────────┘
```

## 🔄 Processing Pipeline

### State Machine (14 States)

```
PENDING → UPLOADING → RECEIVED → QUEUED_TRANSCRIPTION
  → TRANSCRIBING → TRANSCRIBED → QUEUED_EXTRACTION
  → EXTRACTING → CALLSIGNS_EXTRACTED → QUEUED_ANALYSIS
  → ANALYZING → ANALYZED → COMPLETE
```

### Processing Stages

1. **Node Watcher**: Monitors directory for new WAV files (stability checks)
2. **SFTP Transfer**: Uploads files with exponential backoff retry
3. **Ingestion**: SHA256 deduplication, move to storage, queue
4. **Transcription**: faster-whisper GPU transcription (4 workers)
5. **Extraction**: Regex + vLLM callsign validation (10 workers)
6. **Analysis**: Topic/net detection, profile generation (2 workers)

## 📁 Project Structure

```
TSN_V2/
├── tsn_common/          # Shared infrastructure
│   ├── config.py        # Pydantic settings
│   ├── logging.py       # Structured logging
│   ├── db.py            # Database connections
│   ├── db_init.py       # Database initialization
│   ├── utils.py         # Helper functions
│   └── models/          # SQLAlchemy ORM (12 tables)
├── tsn_node/            # Repeater site services
│   ├── watcher.py       # File monitoring
│   └── transfer.py      # SFTP upload
├── tsn_server/          # Central server services
│   ├── ingestion.py     # File reception
│   ├── transcriber.py   # Whisper transcription
│   ├── extractor.py     # Callsign extraction
│   ├── analyzer.py      # Topic/net analysis
│   └── health.py        # Health check server
├── tsn_cli/             # Command-line tools
├── docs/                # Documentation (90+ pages)
├── tests/               # Test suite
└── migrations/          # Database migrations
```

## 🔧 Configuration Reference

See `.env.example` for all options. Key settings:

### Database
```bash
TSN_DB_ENGINE=mysql
TSN_DB_HOST=localhost
TSN_DB_PORT=3306
TSN_DB_NAME=tsn
TSN_DB_USER=tsn_user
TSN_DB_PASSWORD=secure_password
TSN_DB_DRIVER=  # optional override e.g. asyncmy
```

### Node (Repeater Site)
```bash
TSN_NODE_ENABLED=true
TSN_NODE_NODE_ID=node001
TSN_NODE_AUDIO_INCOMING_DIR=/incoming
TSN_NODE_AUDIO_ARCHIVE_DIR=/archive
TSN_NODE_SFTP_HOST=server.example.com
TSN_NODE_SFTP_USERNAME=tsn_user
TSN_NODE_TRANSFER_WORKERS=2
```

### Server (Central)
```bash
TSN_SERVER_ENABLED=true
TSN_SERVER_INCOMING_DIR=/incoming
TSN_STORAGE_BASE_PATH=/storage
```

### Transcription
```bash
TSN_WHISPER_BACKEND=faster-whisper
TSN_WHISPER_MODEL=medium.en
TSN_WHISPER_DEVICE=cuda
TSN_WHISPER_MAX_CONCURRENT=4
```

### vLLM Analysis
```bash
TSN_VLLM_BASE_URL=http://localhost:8001/v1
TSN_VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4
TSN_VLLM_MAX_CONCURRENT=10
```

## 📊 Monitoring

### Health Check Endpoints

```bash
GET http://localhost:8080/health       # Health status
GET http://localhost:8080/metrics      # Prometheus metrics
GET http://localhost:8080/metrics/json # JSON metrics
```

### Prometheus Metrics

- `tsn_files_processed_total{state}` - Total files processed
- `tsn_files_in_state{state}` - Current queue depth
- `tsn_processing_duration_seconds{stage}` - Processing time histogram
- `tsn_callsigns_total` - Total unique callsigns
- `tsn_nets_total` - Total net sessions

## 📚 Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** (40+ pages) - Complete system design
- **[DESIGN_NOTES.md](docs/DESIGN_NOTES.md)** (15 pages) - Key design decisions
- **[REFERENCE_ANALYSIS.md](docs/REFERENCE_ANALYSIS.md)** (25+ pages) - Original system analysis
- **[PROJECT_STATUS.md](docs/PROJECT_STATUS.md)** - Current status and roadmap

## 🎓 Key Features

### Reliability
- SHA256 deduplication (never lose audio)
- Database-backed work queue (survive crashes)
- Exponential backoff retry (3 attempts)
- State machine tracking (atomic updates)
- Append-only event logs (audit trail)

### Performance
- Async/await architecture (10x throughput)
- Concurrent workers (4 transcription, 10 extraction, 2 analysis)
- GPU-accelerated transcription (faster-whisper)
- Connection pooling (20 + 10 overflow)
- Lazy model loading (memory efficiency)

### Observability
- Structured JSON logging (machine-readable)
- Prometheus metrics export
- Health check endpoints
- CLI status commands
- Real-time queue monitoring

### Maintainability
- Type hints throughout (mypy-compatible)
- Pydantic configuration (validation)
- SQLAlchemy ORM (type-safe queries)
- Comprehensive documentation
- Test suite (>80% coverage target)

## 🚦 Status

**Phase**: Core Implementation Complete ✅

**Completed**:
- ✅ Database schema (12 tables)
- ✅ Configuration system
- ✅ Logging infrastructure
- ✅ Node watcher & transfer
- ✅ Server ingestion
- ✅ Transcription pipeline
- ✅ Callsign extraction
- ✅ Analysis pipeline
- ✅ Health check server
- ✅ CLI tools
- ✅ Orchestrator

**Next Steps**:
- ⏳ Database migrations (Alembic)
- ⏳ Test suite (pytest)
- ⏳ Docker deployment
- ⏳ systemd service files

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

This is a complete rewrite focused on production deployment. See `docs/ARCHITECTURE.md` for design principles.

---

**The Spoken Network v2.0** - Built for reliability, designed for scale.
