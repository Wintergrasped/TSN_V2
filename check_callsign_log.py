"""
Check CallsignLog schema
"""
import pymysql

conn = pymysql.connect(
    host='51.81.202.9',
    port=3306,
    user='server',
    password='wowhead1',
    database='repeater',
    charset='utf8mb4'
)

cursor = conn.cursor(pymysql.cursors.DictCursor)

print("CALLSIGN_LOG SCHEMA:")
cursor.execute("DESCRIBE callsign_log")
schema = cursor.fetchall()
for col in schema:
    print(f"  {col['Field']}: {col['Type']}")

print("\nSample data:")
cursor.execute("SELECT * FROM callsign_log LIMIT 3")
for row in cursor.fetchall():
    print(row)

cursor.close()
conn.close()
