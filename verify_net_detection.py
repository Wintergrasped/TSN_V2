"""
Quick verification that net detection overhaul is working.
Run this after deploying changes to verify formal structure detection.
"""

import asyncio
import json
from tsn_common.db import async_session_maker
from tsn_common.models.net import NetSession
from sqlalchemy import select, desc


async def check_net_detection_overhaul():
    """Verify formal structure fields are present and working."""
    
    print("🔍 Checking Net Detection Overhaul Status...\n")
    
    async with async_session_maker() as session:
        # Get the 5 most recent nets
        stmt = select(NetSession).order_by(desc(NetSession.created_at)).limit(5)
        result = await session.execute(stmt)
        nets = list(result.scalars().all())
        
        if not nets:
            print("❌ No nets found in database")
            return
        
        print(f"✅ Found {len(nets)} recent nets\n")
        
        # Check each net for new formal structure fields
        for i, net in enumerate(nets, 1):
            print(f"--- Net #{i}: {net.net_name} ---")
            print(f"    Start: {net.start_time}")
            print(f"    Confidence: {net.confidence:.1%}")
            
            # Check formal_structure
            if net.formal_structure:
                fs = net.formal_structure
                print(f"    ✅ Formal Structure Detected:")
                print(f"       - Opening: {'✓' if fs.get('has_opening') else '✗'}")
                print(f"       - Check-ins: {'✓' if fs.get('has_checkins') else '✗'}")
                print(f"       - Closing: {'✓' if fs.get('has_closing') else '✗'}")
                
                if fs.get('opening_text'):
                    print(f"       - Opening Quote: \"{fs['opening_text'][:80]}...\"")
                if fs.get('closing_text'):
                    print(f"       - Closing Quote: \"{fs['closing_text'][:80]}...\"")
            else:
                print(f"    ⚠️  No formal structure data (old net or detection pending)")
            
            # Check NCS script
            if net.ncs_script:
                print(f"    ✅ NCS Script: {len(net.ncs_script)} statements captured")
                if net.ncs_script:
                    print(f"       - Example: \"{net.ncs_script[0][:60]}...\"")
            else:
                print(f"    ⚠️  No NCS script data")
            
            # Check checkin sequence
            if net.checkin_sequence:
                print(f"    ✅ Formal Check-ins: {len(net.checkin_sequence)} detected")
                if net.checkin_sequence:
                    first = net.checkin_sequence[0]
                    print(f"       - First: {first.get('callsign')} - \"{first.get('statement', '')[:50]}...\"")
            else:
                print(f"    ⚠️  No formal check-in sequence")
            
            print()
        
        # Summary
        nets_with_structure = sum(1 for n in nets if n.formal_structure)
        nets_with_script = sum(1 for n in nets if n.ncs_script)
        nets_with_checkins = sum(1 for n in nets if n.checkin_sequence)
        
        print("=" * 60)
        print("📊 SUMMARY")
        print("=" * 60)
        print(f"Total nets checked: {len(nets)}")
        print(f"With formal structure: {nets_with_structure}/{len(nets)}")
        print(f"With NCS script: {nets_with_script}/{len(nets)}")
        print(f"With checkin sequence: {nets_with_checkins}/{len(nets)}")
        print()
        
        if nets_with_structure > 0:
            print("✅ NET DETECTION OVERHAUL IS WORKING!")
            print("   New formal structure fields are being populated.")
        elif nets:
            print("⚠️  OVERHAUL DEPLOYED BUT NO NEW NETS YET")
            print("   Wait for new audio files to be analyzed.")
            print("   Or manually trigger re-analysis of recent transcripts.")
        else:
            print("❌ NO NETS TO CHECK")


if __name__ == "__main__":
    asyncio.run(check_net_detection_overhaul())
