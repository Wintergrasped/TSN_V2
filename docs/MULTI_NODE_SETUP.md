# Multi-Node Audio File Handling

## Overview

TSN V2 supports multiple input nodes, where each node can independently capture and upload audio files. To ensure proper separation and attribution of data, audio files must be tagged with their originating node.

## Node ID System

### Filename Format

Audio files follow this naming convention:

```
<node_id>_<timestamp>.wav
```

**Examples:**
- `node1_20260128120000.wav` - File from node1 captured at 2026-01-28 12:00:00 UTC
- `repeater1_20260128120500.wav` - File from repeater1 captured at 2026-01-28 12:05:00 UTC
- `20260128120000.wav` - Legacy format without node ID (assigned node_id="unknown")

### Database Storage

The `audio_files` table includes a `node_id` column (indexed) that stores the originating node identifier:

```sql
CREATE TABLE audio_files (
    id UUID PRIMARY KEY,
    filename VARCHAR(255) UNIQUE NOT NULL,
    node_id VARCHAR(64) NOT NULL,  -- Extracted from filename prefix
    ...
    INDEX ix_audio_files_node_created (node_id, created_at)
);
```

## Automatic Node ID Extraction

### During Ingestion

The ingestion service automatically extracts node IDs from filenames:

1. **Parse filename** - `parse_audio_filename_metadata()` extracts node_id and timestamp
2. **Validate format** - Checks for `node_timestamp` pattern
3. **Store in database** - Sets `audio_files.node_id` field
4. **Log extraction** - Records node_id extraction for audit trail

**Code path:** `tsn_server/ingestion.py` → `ingest_file()` → line 91-93

### Startup Repair

On service startup, the ingestion service runs a repair operation to backfill node IDs for any historical records with `node_id="unknown"`:

```python
# Automatically runs on first ingestion loop
await repair_node_ids(limit=1000)
```

This ensures legacy files or files ingested before node_id extraction was implemented get properly tagged.

## Manual Repair

You can manually fix node IDs using the repair script:

```bash
# Dry run - show what would change
python -m scripts.fix_audio_node_ids --dry-run

# Apply changes
python -m scripts.fix_audio_node_ids

# Limit to specific number of records
python -m scripts.fix_audio_node_ids --limit 500
```

The script:
- Finds all records with `node_id="unknown"`
- Parses the filename to extract node_id
- Updates the database if a valid node_id is found
- Logs all changes for audit trail

## Data Separation

With node_id properly set, all related data is automatically separated:

### Transcriptions
```sql
-- Transcriptions inherit node context via audio_files join
SELECT t.*, a.node_id 
FROM transcriptions t
JOIN audio_files a ON t.audio_file_id = a.id
WHERE a.node_id = 'node1';
```

### Callsign Logs
```sql
-- Callsign logs track which node heard the transmission
SELECT cl.*, t.audio_file_id, a.node_id
FROM callsign_logs cl
JOIN transcriptions t ON cl.transcription_id = t.id
JOIN audio_files a ON t.audio_file_id = a.id
WHERE a.node_id = 'repeater1';
```

### Net Sessions
```sql
-- Nets are associated with primary node that captured them
SELECT ns.*, COUNT(DISTINCT a.node_id) as node_count
FROM net_sessions ns
JOIN net_participations np ON np.net_session_id = ns.id
JOIN callsign_logs cl ON cl.id = np.checkin_log_id
JOIN transcriptions t ON t.id = cl.transcription_id
JOIN audio_files a ON a.id = t.audio_file_id
GROUP BY ns.id;
```

## Best Practices

1. **Consistent Naming** - Always use `<node_id>_<timestamp>.wav` format
2. **Unique Node IDs** - Each physical node should have a unique identifier
3. **Monitor Extraction** - Check logs for `node_id_extracted_from_filename` events
4. **Run Repairs** - Periodically run repair script after bulk imports
5. **Query by Node** - Always include node_id in WHERE clauses when analyzing specific nodes

## Troubleshooting

### Files showing node_id="unknown"

**Cause:** Filename doesn't match expected pattern

**Solutions:**
1. Check filename format - must be `<node>_<timestamp>.wav`
2. Node portion must contain alphanumeric characters and hyphens only
3. Timestamp must be exactly 14 digits (YYYYMMDDHHMMSS)

### Node ID not updating automatically

**Cause:** File was ingested before extraction logic was added

**Solution:** Run the repair script:
```bash
python -m scripts.fix_audio_node_ids
```

### Multiple nodes showing same data

**Cause:** Queries not filtering by node_id

**Solution:** Add node_id filter to queries:
```sql
WHERE audio_files.node_id = 'your_node_id'
```

## Configuration

No configuration needed - node ID extraction is automatic. The system uses regex pattern matching:

```python
_NEW_FILENAME_PATTERN = re.compile(
    r"^(?P<node>[A-Za-z0-9\-]+)_(?P<ts>\d{14})(?:[._-].+)?$"
)
```

## Migration Notes

If migrating from single-node to multi-node:

1. **Backup database** before making changes
2. **Rename historical files** with node prefix if needed
3. **Run repair script** to backfill node IDs
4. **Verify separation** by querying distinct node_ids:
   ```sql
   SELECT node_id, COUNT(*) FROM audio_files GROUP BY node_id;
   ```
5. **Update dashboards** to filter by node_id
