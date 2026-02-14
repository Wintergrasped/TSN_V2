"""
Fix stuck WARMUP candidates by manually activating them if they meet relaxed criteria.
"""

import pymysql
import json
from datetime import datetime

conn = pymysql.connect(
    host='51.81.202.9',
    port=3306,
    user='server',
    password='wowhead1',
    database='repeater',
    charset='utf8mb4'
)

cursor = conn.cursor(pymysql.cursors.DictCursor)

print("🔧 Fixing Stuck WARMUP Candidates")
print("=" * 80)

# Get all WARMUP candidates
cursor.execute("SELECT * FROM net_candidates WHERE status = 'WARMUP'")
warmup_candidates = cursor.fetchall()

print(f"\nFound {len(warmup_candidates)} WARMUP candidates\n")

fixed_count = 0

for candidate in warmup_candidates:
    features = json.loads(candidate['features_json'] or '{}')
    unique_callsigns = len(features.get("unique_callsigns", []))
    peak = candidate['vllm_confidence_peak'] or 0
    avg = candidate['vllm_confidence_avg'] or 0
    
    print(f"Candidate {candidate['id']}:")
    print(f"  Node: {candidate['node_id']}")
    print(f"  Start: {candidate['start_ts']}")
    print(f"  Evaluations: {candidate['vllm_evaluation_count']}")
    print(f"  Peak likelihood: {peak}")
    print(f"  Avg likelihood: {avg}")
    print(f"  Unique callsigns: {unique_callsigns}")
    
    # Relaxed activation criteria:
    # - At least 10 evaluations (shows sustained activity)
    # - Peak >= 60 OR avg >= 30 (some net signals)
    # - At least 4 unique callsigns
    
    should_activate = (
        candidate['vllm_evaluation_count'] >= 10 and
        (peak >= 60 or avg >= 30) and
        unique_callsigns >= 4
    )
    
    if should_activate:
        print(f"  ✅ ACTIVATING (meets relaxed criteria)")
        cursor.execute(
            "UPDATE net_candidates SET status = 'ACTIVE', updated_at = %s WHERE id = %s",
            (datetime.utcnow(), candidate['id'])
        )
        fixed_count += 1
    else:
        print(f"  ⏭️  Skipping (insufficient signals)")
    
    print()

if fixed_count > 0:
    conn.commit()
    print(f"\n✅ Activated {fixed_count} candidates!")
else:
    print(f"\n⚠️  No candidates met activation criteria")

conn.close()

print("\n" + "=" * 80)
print("Next step: Restart tsn_server to apply code fixes:")
print("  cd /opt/tsn-server")
print("  docker compose restart tsn_server")
