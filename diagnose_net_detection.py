"""
Check if net autodetect is actually running and why it's not working
"""
import pymysql
from datetime import datetime, timedelta

# Database connection from .env.example
conn = pymysql.connect(
    host='51.81.202.9',
    port=3306,
    user='server',
    password='wowhead1',
    database='repeater',
    charset='utf8mb4'
)

cursor = conn.cursor(pymysql.cursors.DictCursor)

print("=" * 80)
print("NET AUTODETECT DIAGNOSTIC")
print("=" * 80)

# Check if there are recent transcriptions with net-related content
print("\n--- RECENT NET-RELATED TRANSCRIPTIONS (Last 24 hours) ---")
yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
cursor.execute("""
    SELECT 
        t.audio_file_id,
        t.transcript_text,
        a.node_id,
        a.created_at,
        a.filename
    FROM transcriptions t
    JOIN audio_files a ON t.audio_file_id = a.id
    WHERE a.created_at > %s
    AND (
        t.transcript_text LIKE '%%net control%%'
        OR t.transcript_text LIKE '%%check in%%'
        OR t.transcript_text LIKE '%%check-in%%'
        OR t.transcript_text LIKE '%%traffic%%'
        OR t.transcript_text LIKE '%%net is now%%'
    )
    ORDER BY a.created_at DESC
    LIMIT 10
""", (yesterday,))

net_transcripts = cursor.fetchall()
print(f"Found {len(net_transcripts)} net-related transcripts in last 24 hours")
for row in net_transcripts:
    print(f"\nNode: {row['node_id']}")
    print(f"Time: {row['created_at']}")
    print(f"File: {row['filename']}")
    print(f"Text: {row['transcript_text'][:200]}...")

# Check analysis_audits for net detection attempts
print("\n--- CHECKING ANALYSIS_AUDITS FOR NET DETECTION ---")
cursor.execute("""
    SELECT analysis_type, COUNT(*) as count
    FROM analysis_audits
    GROUP BY analysis_type
""")
for row in cursor.fetchall():
    print(f"{row['analysis_type']}: {row['count']}")

# Check if there are any analysis_audits with net-related results
print("\n--- RECENT ANALYSIS AUDITS (Last 100) ---")
cursor.execute("""
    SELECT 
        aa.analysis_type,
        aa.result_data,
        aa.created_at,
        a.node_id
    FROM analysis_audits aa
    JOIN audio_files a ON aa.audio_file_id = a.id
    ORDER BY aa.created_at DESC
    LIMIT 100
""")
audits = cursor.fetchall()
print(f"Total recent audits: {len(audits)}")

# Count by type
from collections import Counter
type_counts = Counter([a['analysis_type'] for a in audits])
print("Analysis types:")
for atype, count in type_counts.items():
    print(f"  {atype}: {count}")

# Check for any net-related results
import json
net_related_count = 0
for audit in audits:
    try:
        if audit['result_data']:
            data = json.loads(audit['result_data'])
            if isinstance(data, dict) and data.get('is_net_related'):
                net_related_count += 1
    except:
        pass

print(f"\nNet-related results found: {net_related_count}")

# Check callsign_log for recent activity
print("\n--- RECENT CALLSIGN DETECTIONS (Last 24 hours) ---")
cursor.execute("""
    SELECT 
        cl.callsign_id,
        cl.audio_file_id,
        cl.heard_at,
        c.callsign,
        a.node_id
    FROM callsign_log cl
    JOIN callsigns c ON cl.callsign_id = c.id
    JOIN audio_files a ON cl.audio_file_id = a.id
    WHERE cl.heard_at > %s
    ORDER BY cl.heard_at DESC
    LIMIT 20
""", (yesterday,))

callsign_logs = cursor.fetchall()
print(f"Found {len(callsign_logs)} callsign detections in last 24 hours")
if callsign_logs:
    for row in callsign_logs[:5]:
        print(f"  {row['callsign']} on {row['node_id']} at {row['heard_at']}")

# Check what nodes are active
print("\n--- ACTIVE NODES (Last 24 hours) ---")
cursor.execute("""
    SELECT 
        node_id,
        COUNT(*) as file_count,
        MAX(created_at) as last_file,
        MIN(created_at) as first_file
    FROM audio_files
    WHERE created_at > %s
    GROUP BY node_id
    ORDER BY file_count DESC
""", (yesterday,))

nodes = cursor.fetchall()
print(f"Active nodes: {len(nodes)}")
for row in nodes:
    print(f"  {row['node_id']}: {row['file_count']} files, last: {row['last_file']}")

cursor.close()
conn.close()

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
