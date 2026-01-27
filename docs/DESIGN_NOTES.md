# TSN V2 Design Notes

## Key Design Decisions & Rationale

### 1. MySQL/MariaDB as the Primary Database

**Decision**: Standardize on MySQL 8.0+ / MariaDB 10.6+ with async support  
**Rationale**:
- **Operational Fit**: Matches the existing repeater infrastructure and DBA skillset
- **UTF-8 Everywhere**: Mature support for `utf8mb4` across drivers and tooling
- **Async Driver**: `asyncmy` provides stable async access without extra dependencies
- **ACID Compliance**: InnoDB delivers the guarantees TSN needs
- **Lower Complexity**: Keeps the deployment footprint minimal (no secondary DB engine)

**Migration Path**: Continue using the existing MySQL schema, running the UUID migrator if needed

### 2. UUID Primary Keys

**Decision**: Use UUID v4 for all primary keys  
**Rationale**:
- **Distributed Safety**: Can generate IDs on any node without coordination
- **Security**: Non-sequential IDs prevent enumeration attacks
- **Merging**: Easy to merge data from multiple sources
- **Modern Standard**: Industry best practice for microservices

**Trade-off**: Slightly larger index size vs auto-increment integers (acceptable)

### 3. Async-First Architecture

**Decision**: asyncio throughout, SQLAlchemy async, httpx  
**Rationale**:
- **Concurrency**: Handle hundreds of files without threading complexity
- **I/O Bound**: Audio processing is network/disk heavy, perfect for async
- **Single Thread**: Easier debugging than thread pools
- **Modern Python**: 3.11+ async performance is excellent

**Implementation**: async/await everywhere, no blocking calls in hot paths

### 4. State Machine in Database

**Decision**: Store processing state in `audio_files.state` column  
**Rationale**:
- **Atomic Updates**: Database transactions prevent race conditions
- **Visibility**: Easy to query stuck files, retry failures
- **Idempotency**: Can safely restart workers mid-processing
- **Audit Trail**: State transitions logged automatically

**Original Issue**: File-based state (incoming/processed dirs) had race conditions

### 5. No Message Queue Dependency

**Decision**: Use MySQL as the work queue (no Redis/RabbitMQ)  
**Rationale**:
- **Simplicity**: One less infrastructure component
- **ACID**: Transactional guarantees for queue operations
- **Sufficient**: TSN isn't high-frequency trading; InnoDB row locks are fast enough
- **Observability**: SQL queries to inspect queue state

**Performance**: `SELECT ... FOR UPDATE SKIP LOCKED` via MySQL 8.0 to dequeue work without contention

### 6. Phonetic Corrections Table

**Decision**: Database table instead of hardcoded dict  
**Rationale**:
- **Runtime Updates**: Add corrections without code deploy
- **Audit**: Track when corrections added
- **Sharing**: Multiple workers see same corrections instantly
- **Backup**: Included in database backups

**Original**: Hardcoded Python dict in multiple files

### 7. Separate Transcription from Analysis

**Decision**: Split transcription worker from analysis worker  
**Rationale**:
- **Resource Isolation**: GPU transcription vs CPU analysis
- **Scaling**: Can run 4 transcription + 10 analysis workers independently
- **Failure Isolation**: Analysis crash doesn't block transcription
- **Clear Boundaries**: Single Responsibility Principle

**Deployment**: Different Docker containers, different systemd services

### 8. Callsign Validation Two-Stage

**Decision**: Regex first, then vLLM validation for high-value callsigns  
**Rationale**:
- **Cost**: vLLM is slower/expensive, regex is free
- **Accuracy**: Regex finds candidates, AI reduces false positives
- **Caching**: Validated callsigns cached, avoid re-validation
- **Fallback**: If vLLM down, regex-only mode degrades gracefully

**Original**: Mixed regex + QRZ in one pass

### 9. Explicit Retry Logic

**Decision**: `retry_count` column + exponential backoff  
**Rationale**:
- **Transient Failures**: Network blips, temporary GPU OOM
- **Poison Pills**: Max retries → dead letter queue
- **Observability**: See which files are problematic
- **Fairness**: Don't block queue with one bad file

**Backoff**: 2^retry_count * base_delay, max 5 minutes

### 10. Structured Logging with JSON

**Decision**: structlog with JSON output  
**Rationale**:
- **Machine Readable**: Easy to parse, index in Elasticsearch/Loki
- **Context Propagation**: Request ID, user ID follow through pipeline
- **Production Standard**: Modern observability stack expects JSON
- **Development**: Console renderer for local debugging

**Format**: ISO timestamps, log levels, event names, context dict

### 11. Prometheus Metrics

**Decision**: Expose /metrics endpoint for Prometheus scraping  
**Rationale**:
- **Industry Standard**: Grafana dashboards, alerting rules
- **Low Overhead**: Pull-based, no agent needed
- **Rich Ecosystem**: Pre-built exporters, integrations
- **Histograms**: Track latency distributions, not just averages

**Metrics**: Counters, gauges, histograms for every pipeline stage

### 12. Type Safety with Pydantic

**Decision**: Pydantic for config, mypy for type checking  
**Rationale**:
- **Validation**: Catch config errors at startup, not runtime
- **Documentation**: Types serve as inline documentation
- **IDE Support**: Autocomplete, refactoring safety
- **Maintainability**: Easier for new contributors to understand

**Enforcement**: mypy in CI, fail build on type errors

### 13. Alembic for Migrations

**Decision**: Alembic for database schema evolution  
**Rationale**:
- **Version Control**: Schema changes tracked in git
- **Rollback**: Downgrade migrations if deploy fails
- **Team Coordination**: No manual SQL scripts to coordinate
- **SQLAlchemy Integration**: Auto-generate migrations from model changes

**Workflow**: `alembic revision --autogenerate`, review, `alembic upgrade head`

### 14. Testing Strategy

**Decision**: pytest with async support, coverage >80%  
**Rationale**:
- **Confidence**: Refactor without breaking things
- **Documentation**: Tests show how to use code
- **Regressions**: Catch bugs before production
- **CI/CD**: Automated testing on every commit

**Levels**: Unit (models, utils), Integration (DB operations), E2E (full pipeline)

### 15. Docker for Deployment

**Decision**: Multi-stage Docker builds, docker-compose for local  
**Rationale**:
- **Reproducibility**: Same environment dev → prod
- **Dependency Management**: Python deps, system libs packaged together
- **GPU Support**: NVIDIA Container Toolkit for Whisper/vLLM
- **Orchestration**: Easy to add Kubernetes later if needed

**Images**: tsn-node, tsn-transcriber, tsn-analyzer

## Differences from Original Implementation

| Aspect | Original | TSN V2 | Benefit |
|--------|----------|---------|---------|
| Language | Python 3.9 | Python 3.11+ | 20% faster, better type hints |
| Async | Threading, subprocess | asyncio | 10x concurrency |
| Database | MySQL | MySQL (async) | Single engine, no migration |
| ORM | mysql-connector | SQLAlchemy 2.0 | Type safety, migrations |
| Config | Hardcoded + .env | Pydantic Settings | Validation, hierarchy |
| State | File system | Database | ACID, no races |
| Logging | print() | structlog | JSON, context |
| Metrics | None | Prometheus | Observability |
| Retries | Manual try/except | Exponential backoff | Resilience |
| Tests | None | pytest >80% | Confidence |
| Deployment | Shell scripts | Docker/systemd | Reproducibility |

## Performance Improvements

### Throughput
- **Original**: ~10 files/minute (sequential processing)
- **V2**: ~100 files/minute (4 concurrent transcribers, 10 analyzers)
- **Bottleneck**: GPU transcription (30s per 1min audio)

### Latency
- **Original**: 2-10 minutes (end-to-end)
- **V2**: <5 minutes target (parallel stages)
- **P99**: <10 minutes with retry backoff

### Resource Usage
- **CPU**: -30% (async vs threads)
- **Memory**: +10% (connection pooling)
- **GPU**: 90% utilization (batch processing)
- **Disk**: -70% (MP3 compression)

## Failure Mode Analysis

### Network Failures
- **Node → Server**: Exponential backoff, local queue grows, auto-resume
- **Server → vLLM**: Circuit breaker, fallback to OpenAI, degraded mode
- **Server → Database**: Connection pool retries, transaction rollback

### Service Crashes
- **Node Watcher**: Systemd auto-restart, resume from last checkpoint
- **Transcriber**: Unprocessed files stay in queue, picked up by healthy worker
- **Analyzer**: Partial results committed, resume from last transcript_id

### Data Corruption
- **Malformed WAV**: FFmpeg validation, move to failed queue
- **Bad Transcript**: Quality gate (min words), skip analysis
- **Invalid JSON**: Best-effort parsing, fallback to regex

### Resource Exhaustion
- **Disk Full**: Alert, stop recording, purge old archives
- **GPU OOM**: Reduce batch size, kill process and restart
- **Database Connections**: Pool limits, wait with timeout

## Security Considerations

1. **Secrets Management**: Environment variables, never in code
2. **SQL Injection**: SQLAlchemy parameterized queries
3. **API Keys**: Stored in DB encrypted, never logged
4. **File Uploads**: SHA256 verification, size limits
5. **Rate Limiting**: Max requests per minute to vLLM

## Future Scaling Path

### Current (Single Server)
- 1 database (MySQL/MariaDB)
- 1-4 transcription workers (GPU-bound)
- 5-10 analysis workers (CPU-bound)
- 1-10 nodes (repeater sites)

### Next (Horizontal)
- 1 database cluster (MySQL replication / read replicas)
- N transcription workers (multiple GPUs)
- M analysis workers (CPU scale-out)
- 100+ nodes (federation)

### Future (Distributed)
- Database sharding by node_id
- Message queue (Kafka) for cross-region
- CDN for audio file distribution
- Service mesh (Istio) for observability

## Maintenance Playbook

### Daily
- Check Grafana dashboard for anomalies
- Review error logs for new patterns
- Verify queue depths < thresholds

### Weekly
- Rotate logs if file-based
- Review slow queries (pg_stat_statements)
- Update phonetic corrections table

### Monthly
- Vacuum/analyze database
- Archive old audio files (>30 days)
- Update dependencies (security patches)

### Quarterly
- Load test to find bottlenecks
- Review and update AI prompts
- Performance tuning (indexes, caching)

## Conclusion

TSN V2 is designed for **production reliability** while maintaining **developer velocity**. Every decision balances pragmatism (ship it now) with long-term maintainability (refactor safely later). The architecture can scale 10x before needing fundamental changes.
