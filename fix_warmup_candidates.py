"""
Fix stuck WARMUP candidates by manually activating them if they meet relaxed criteria.
"""

import asyncio
from datetime import datetime, timezone
from tsn_common.db import async_session_maker
from tsn_common.models.net import NetCandidate
from sqlalchemy import select, update

async def fix_warmup_candidates():
    """Manually activate WARMUP candidates that meet relaxed criteria."""
    
    print("🔧 Fixing Stuck WARMUP Candidates")
    print("=" * 80)
    
    async with async_session_maker() as session:
        # Get all WARMUP candidates
        stmt = select(NetCandidate).where(NetCandidate.status == "WARMUP")
        result = await session.execute(stmt)
        warmup_candidates = list(result.scalars().all())
        
        print(f"\nFound {len(warmup_candidates)} WARMUP candidates\n")
        
        fixed_count = 0
        
        for candidate in warmup_candidates:
            features = candidate.features_json or {}
            unique_callsigns = len(features.get("unique_callsigns", []))
            peak = candidate.vllm_confidence_peak or 0
            avg = candidate.vllm_confidence_avg or 0
            
            print(f"Candidate {candidate.id}:")
            print(f"  Node: {candidate.node_id}")
            print(f"  Start: {candidate.start_ts}")
            print(f"  Evaluations: {candidate.vllm_evaluation_count}")
            print(f"  Peak likelihood: {peak}")
            print(f"  Avg likelihood: {avg}")
            print(f"  Unique callsigns: {unique_callsigns}")
            
            # Relaxed activation criteria:
            # - At least 10 evaluations (shows sustained activity)
            # - Peak >= 60 OR avg >= 30 (some net signals)
            # - At least 4 unique callsigns
            
            should_activate = (
                candidate.vllm_evaluation_count >= 10 and
                (peak >= 60 or avg >= 30) and
                unique_callsigns >= 4
            )
            
            if should_activate:
                print(f"  ✅ ACTIVATING (meets relaxed criteria)")
                candidate.status = "ACTIVE"
                fixed_count += 1
            else:
                print(f"  ⏭️  Skipping (insufficient signals)")
            
            print()
        
        if fixed_count > 0:
            await session.commit()
            print(f"\n✅ Activated {fixed_count} candidates!")
        else:
            print(f"\n⚠️  No candidates met activation criteria")
        
        print("\n" + "=" * 80)
        print("Next step: Restart tsn_server to apply code fixes:")
        print("  cd /opt/tsn-server")
        print("  docker compose restart tsn_server")

if __name__ == "__main__":
    asyncio.run(fix_warmup_candidates())
