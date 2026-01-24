"""
Quick deployment verification test - checks basic functionality without external dependencies
"""
import ast
import sys
from pathlib import Path

def test_syntax_all_files():
    """Check Python syntax on all .py files"""
    errors = []
    root = Path(".")
    
    for py_file in root.rglob("*.py"):
        if "venv" in str(py_file) or "__pycache__" in str(py_file):
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                ast.parse(f.read(), filename=str(py_file))
        except SyntaxError as e:
            errors.append(f"{py_file}: {e}")
    
    return errors

def test_orchestrator_logic():
    """Verify orchestrator has correct attribute access"""
    with open("tsn_orchestrator.py", 'r') as f:
        content = f.read()
        
    # Check for the bug we just fixed
    if "watcher.queue" in content:
        return ["ERROR: orchestrator still uses watcher.queue instead of watcher.transfer_queue"]
    
    if "watcher.transfer_queue" in content:
        return []
    
    return ["WARNING: Neither watcher.queue nor watcher.transfer_queue found in orchestrator"]

def test_watcher_has_transfer_queue():
    """Verify FileWatcher has transfer_queue attribute"""
    with open("tsn_node/watcher.py", 'r') as f:
        content = f.read()
    
    if "self.transfer_queue" not in content:
        return ["ERROR: FileWatcher missing transfer_queue attribute"]
    
    if "asyncio.Queue" not in content:
        return ["ERROR: transfer_queue not initialized as asyncio.Queue"]
    
    return []

def test_config_structure():
    """Verify config has required settings"""
    with open("tsn_common/config.py", 'r') as f:
        content = f.read()
    
    required = [
        "class NodeSettings",
        "class ServerSettings",
        "class DatabaseSettings",
        "enabled: bool",  # Check for enabled field
    ]
    
    errors = []
    for req in required:
        if req not in content:
            errors.append(f"Missing in config.py: {req}")
    
    return errors

def main():
    print("=" * 60)
    print("TSN v2 Deployment Verification")
    print("=" * 60)
    
    tests = [
        ("Syntax Check", test_syntax_all_files),
        ("Orchestrator Logic", test_orchestrator_logic),
        ("Watcher Queue Attribute", test_watcher_has_transfer_queue),
        ("Config Structure", test_config_structure),
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        print(f"\n[TEST] {test_name}...", end=" ")
        errors = test_func()
        
        if errors:
            print("❌ FAILED")
            for error in errors:
                print(f"  - {error}")
            all_passed = False
        else:
            print("✅ PASSED")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL CHECKS PASSED - Ready for deployment")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Build Docker image: docker compose build")
        print("2. Deploy to node: docker compose up -d")
        print("3. Check logs: docker logs -f tsn_node")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - Fix issues before deployment")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
