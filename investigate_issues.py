"""
Investigate callsign extraction and net detection issues
Read-only database queries to understand the problem
"""
import pymysql
import json
from datetime import datetime

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
print("DATABASE INVESTIGATION - READ ONLY")
print("=" * 80)

# Check what tables exist
print("\n--- AVAILABLE TABLES ---")
cursor.execute("SHOW TABLES")
tables = cursor.fetchall()
for table in tables:
    table_name = list(table.values())[0]
    cursor.execute(f"SELECT COUNT(*) as count FROM `{table_name}`")
    count = cursor.fetchone()['count']
    print(f"{table_name}: {count} rows")

# Check schema first
print("\n--- AUDIO_FILES TABLE SCHEMA ---")
cursor.execute("DESCRIBE audio_files")
schema = cursor.fetchall()
for col in schema:
    print(f"{col['Field']}: {col['Type']}")

# Check total counts
print("\n--- TABLE COUNTS ---")
cursor.execute("SELECT COUNT(*) as count FROM audio_files")
print(f"Total audio files: {cursor.fetchone()['count']}")

cursor.execute("SELECT COUNT(*) as count FROM callsigns")
print(f"Total callsigns detected: {cursor.fetchone()['count']}")

# Sample transcripts with potential callsign issues
print("\n--- SAMPLE TRANSCRIPTION DATA ---")
cursor.execute("DESCRIBE transcriptions")
trans_schema = cursor.fetchall()
print("Transcriptions table columns:")
for col in trans_schema:
    print(f"  {col['Field']}: {col['Type']}")

cursor.execute("""
    SELECT t.id, t.audio_file_id, t.transcript_text, a.filename, a.created_at
    FROM transcriptions t
    JOIN audio_files a ON t.audio_file_id = a.id
    WHERE t.transcript_text IS NOT NULL 
    AND t.transcript_text != ''
    ORDER BY a.created_at DESC
    LIMIT 5
""")
for row in cursor.fetchall():
    print(f"\nTranscription ID: {row['id']}")
    print(f"Audio File: {row['filename']}")
    print(f"Date: {row['created_at']}")
    print(f"Text: {row['transcript_text'][:300]}...")

# Look at callsigns table structure and samples
print("\n--- CALLSIGNS TABLE SCHEMA ---")
cursor.execute("DESCRIBE callsigns")
callsign_schema = cursor.fetchall()
for col in callsign_schema:
    print(f"{col['Field']}: {col['Type']}")

print("\n--- CALLSIGNS TABLE SAMPLES ---")
cursor.execute("""
    SELECT * FROM callsigns
    ORDER BY last_seen DESC
    LIMIT 10
""")
callsigns = cursor.fetchall()
if callsigns:
    for row in callsigns:
        print(f"\nCallsign: {row.get('callsign')}")
        print(f"Last Seen: {row.get('last_seen')}")
        print(f"Seen Count: {row.get('seen_count')}")
        print(f"Validated: {row.get('validated')}")
else:
    print("No callsigns found in database")

# Look for transcripts that might contain net-related keywords
print("\n--- TRANSCRIPTS WITH NET-RELATED KEYWORDS ---")
cursor.execute("""
    SELECT t.id, t.audio_file_id, t.transcript_text, a.filename, a.created_at
    FROM transcriptions t
    JOIN audio_files a ON t.audio_file_id = a.id
    WHERE t.transcript_text IS NOT NULL 
    AND (t.transcript_text LIKE '%net%' OR t.transcript_text LIKE '%check-in%' 
         OR t.transcript_text LIKE '%check in%' OR t.transcript_text LIKE '%traffic%'
         OR t.transcript_text LIKE '%control%')
    ORDER BY a.created_at DESC
    LIMIT 5
""")
for row in cursor.fetchall():
    print(f"\nTranscription ID: {row['id']}")
    print(f"Audio File: {row['filename']}")
    print(f"Date: {row['created_at']}")
    print(f"Text: {row['transcript_text'][:300]}...")

# Check AI analysis results - use analysis_audits instead
print("\n--- ANALYSIS_AUDITS SCHEMA ---")
cursor.execute("DESCRIBE analysis_audits")
analysis_schema = cursor.fetchall()
for col in analysis_schema:
    print(f"{col['Field']}: {col['Type']}")

print("\n--- ANALYSIS_AUDITS SAMPLES ---")
cursor.execute("""
    SELECT * FROM analysis_audits
    ORDER BY created_at DESC
    LIMIT 5
""")
analyses = cursor.fetchall()
if analyses:
    for row in analyses:
        print(f"\nAudio ID: {row.get('audio_file_id')}")
        print(f"Analysis Type: {row.get('analysis_type')}")
        print(f"Result: {str(row.get('result_data'))[:200]}...")
        print(f"Created: {row.get('created_at')}")
else:
    print("No analysis audits found")

# Check for any net detection history
print("\n--- CHECKING NET-RELATED TABLES ---")
cursor.execute("SHOW TABLES LIKE '%net%'")
net_tables = cursor.fetchall()
for table in net_tables:
    table_name = list(table.values())[0]
    print(f"\nFound table: {table_name}")
    cursor.execute(f"DESCRIBE `{table_name}`")
    schema = cursor.fetchall()
    for col in schema:
        print(f"  {col['Field']}: {col['Type']}")
    
    cursor.execute(f"SELECT COUNT(*) as count FROM `{table_name}`")
    count = cursor.fetchone()['count']
    print(f"  Total rows: {count}")
    
    if count > 0:
        cursor.execute(f"SELECT * FROM `{table_name}` LIMIT 3")
        samples = cursor.fetchall()
        print(f"  Sample data:")
        for row in samples:
            print(f"    {row}")

cursor.close()
conn.close()

print("\n" + "=" * 80)
print("INVESTIGATION COMPLETE")
print("=" * 80)
