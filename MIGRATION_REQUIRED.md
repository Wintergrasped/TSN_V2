# URGENT: Database Migration Required

## Problem
The web container is crashing because the new `formal_structure`, `ncs_script`, and `checkin_sequence` columns don't exist in the `net_sessions` table yet.

**Error**: `Unknown column 'net_sessions.formal_structure' in 'SELECT'`

## Solution - Run on Linux Server

### Option 1: Quick Fix (Recommended)
```bash
# SSH into your Linux server, then run:

docker exec -i $(docker ps -q -f name=tsn_web) mysql -h 172.19.0.1 -u tsn -ptsn123 repeater << 'EOF'
ALTER TABLE net_sessions 
ADD COLUMN formal_structure JSON NULL,
ADD COLUMN ncs_script JSON NULL,
ADD COLUMN checkin_sequence JSON NULL;
EOF

# Then restart the web container:
docker-compose restart tsn_web
```

### Option 2: Using Script
```bash
# Copy the migration script to your server:
# Upload: scripts/migrate_net_structure.sh

# Make it executable:
chmod +x scripts/migrate_net_structure.sh

# Run it:
./scripts/migrate_net_structure.sh
```

### Option 3: Direct MySQL
```bash
# Connect to MySQL:
docker exec -it $(docker ps -q -f name=tsn_web) mysql -h 172.19.0.1 -u tsn -ptsn123 repeater

# Run this SQL:
ALTER TABLE net_sessions 
ADD COLUMN formal_structure JSON NULL,
ADD COLUMN ncs_script JSON NULL,
ADD COLUMN checkin_sequence JSON NULL;

# Exit MySQL and restart:
docker-compose restart tsn_web
```

## Verify Success

After running the migration, check that the columns exist:
```bash
docker exec -i $(docker ps -q -f name=tsn_web) mysql -h 172.19.0.1 -u tsn -ptsn123 repeater -e "DESCRIBE net_sessions;"
```

You should see:
- `formal_structure` | json | YES
- `ncs_script` | json | YES  
- `checkin_sequence` | json | YES

## Why This Happened

The auto-migration system only adds columns when the table is first created. Since `net_sessions` already existed, it didn't add the new columns. We need to add them manually this one time.

## After Migration

Once the columns are added and the container restarted:
1. ✅ Web portal will load without errors
2. ✅ New net detections will populate formal structure data
3. ✅ Display pages will show opening/closing quotes and check-in sequences
4. ✅ Existing nets will show "—" for missing data (graceful degradation)

## If You Can't Access Server Now

You can temporarily comment out the new columns in the model to make the web portal work:

```python
# In tsn_common/models/net.py, comment these lines:
# formal_structure: Mapped[dict | None] = mapped_column(JSON, nullable=True)
# ncs_script: Mapped[list | None] = mapped_column(JSON, nullable=True)
# checkin_sequence: Mapped[list | None] = mapped_column(JSON, nullable=True)
```

Then rebuild the Docker image. But you'll lose the new features until you run the migration.
