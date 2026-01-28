#!/bin/bash
# Run this script on the Linux server to add formal structure columns

echo "🔧 Adding formal structure columns to net_sessions table..."

# Get the web container name
CONTAINER=$(docker ps -q -f name=tsn_web)

if [ -z "$CONTAINER" ]; then
    echo "❌ tsn_web container not found!"
    exit 1
fi

# Add the columns
docker exec -i $CONTAINER mysql -h 172.19.0.1 -u tsn -ptsn123 repeater << 'EOF'
-- Add formal structure columns
ALTER TABLE net_sessions 
ADD COLUMN IF NOT EXISTS formal_structure JSON NULL,
ADD COLUMN IF NOT EXISTS ncs_script JSON NULL,
ADD COLUMN IF NOT EXISTS checkin_sequence JSON NULL;

-- Verify columns
SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = 'repeater'
AND TABLE_NAME = 'net_sessions'
AND COLUMN_NAME IN ('formal_structure', 'ncs_script', 'checkin_sequence');
EOF

echo "✅ Migration complete! Restart the web container:"
echo "   docker-compose restart tsn_web"
