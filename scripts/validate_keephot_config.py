#!/usr/bin/env python3
"""
Quick validation script for vLLM keep-hot system configuration.
Run this after deployment to verify settings are loaded correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tsn_common.config import get_settings
from tsn_common.logging import get_logger

logger = get_logger(__name__)


def validate_configuration():
    """Validate that keep-hot system configuration is properly loaded."""
    
    print("=" * 80)
    print("TSN V2 - vLLM Keep-Hot System Configuration Validator")
    print("=" * 80)
    print()
    
    try:
        settings = get_settings()
        analysis = settings.analysis
        
        # Check critical settings
        checks = [
            ("Idle Poll Interval", analysis.idle_poll_interval_sec, 0.1, "<=", "Should be ≤0.1s for aggressive mode"),
            ("Aggressive Backfill Enabled", analysis.aggressive_backfill_enabled, True, "==", "Must be enabled"),
            ("Idle Work Chain Limit", analysis.idle_work_chain_limit, 1, ">=", "Should be ≥1"),
            ("GPU Watch Enabled", analysis.gpu_watch_enabled, True, "==", "Recommended for monitoring"),
            ("GPU Low Utilization Threshold", analysis.gpu_low_utilization_pct, 65.0, "<=", "Trigger backfill below this"),
            ("Overdrive Batch Size", analysis.overdrive_batch_size, 1, ">=", "Should be ≥1"),
            ("Refinement Batch Size", analysis.refinement_batch_size, 1, ">=", "Should be ≥1"),
            ("Profile Batch Size", analysis.profile_batch_size, 1, ">=", "Should be ≥1"),
            ("Transcript Smoothing Enabled", analysis.transcript_smoothing_enabled, True, "==", "Recommended"),
        ]
        
        passed = 0
        failed = 0
        warnings = 0
        
        for name, actual, expected, operator, description in checks:
            status = "✓ PASS"
            level = "INFO"
            
            if operator == "==":
                check_result = actual == expected
            elif operator == "<=":
                check_result = actual <= expected
            elif operator == ">=":
                check_result = actual >= expected
            else:
                check_result = False
            
            if not check_result:
                if "Recommended" in description:
                    status = "⚠ WARN"
                    level = "WARN"
                    warnings += 1
                else:
                    status = "✗ FAIL"
                    level = "ERROR"
                    failed += 1
            else:
                passed += 1
            
            print(f"{status:10} | {name:35} | {actual:20} | {description}")
        
        print()
        print("=" * 80)
        print(f"Results: {passed} passed, {warnings} warnings, {failed} failed")
        print("=" * 80)
        print()
        
        if failed > 0:
            print("❌ CRITICAL: Configuration issues detected!")
            print("   Please review .env file and ensure:")
            print("   - TSN_ANALYSIS_IDLE_POLL_INTERVAL_SEC=0.1")
            print("   - TSN_ANALYSIS_AGGRESSIVE_BACKFILL_ENABLED=true")
            print("   - TSN_ANALYSIS_IDLE_WORK_CHAIN_LIMIT=10")
            print()
            return False
        
        if warnings > 0:
            print("⚠️  WARNING: Some recommended settings are not optimal.")
            print("   System will work but may not achieve maximum GPU utilization.")
            print()
        else:
            print("✅ SUCCESS: All keep-hot system settings are properly configured!")
            print()
        
        # Print summary
        print("Configuration Summary:")
        print("-" * 80)
        print(f"  Worker Count:           {analysis.worker_count}")
        print(f"  Max Batch Size:         {analysis.max_batch_size}")
        print(f"  Context Char Budget:    {analysis.context_char_budget:,}")
        print(f"  Overdrive Budget:       {analysis.gpu_overdrive_budget:,}")
        print(f"  Idle Poll Interval:     {analysis.idle_poll_interval_sec}s")
        print(f"  Aggressive Backfill:    {analysis.aggressive_backfill_enabled}")
        print(f"  Work Chain Limit:       {analysis.idle_work_chain_limit}")
        print(f"  GPU Low Util Threshold: {analysis.gpu_low_utilization_pct}%")
        print("-" * 80)
        print()
        
        # Print vLLM settings
        print("vLLM Configuration:")
        print("-" * 80)
        print(f"  Base URL:               {settings.vllm.base_url}")
        print(f"  Model:                  {settings.vllm.model}")
        print(f"  Timeout:                {settings.vllm.timeout_sec}s")
        print(f"  Max Concurrent:         {settings.vllm.max_concurrent}")
        print("-" * 80)
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Failed to load configuration: {e}")
        print()
        print("Make sure:")
        print("  1. You are in the TSN_V2 project directory")
        print("  2. .env file exists and is properly formatted")
        print("  3. All required environment variables are set")
        print()
        return False


if __name__ == "__main__":
    success = validate_configuration()
    sys.exit(0 if success else 1)
