"""
Quick test to verify watcher.py fix
"""
import asyncio
import tempfile
import time
from pathlib import Path

# Simulate the buggy version
def buggy_version():
    pending_files = {
        Path("/tmp/test.wav"): time.time() - 5.0
    }
    ready_files = []
    
    for file_path, detected_at in list(pending_files.items()):
        ready_files.append(file_path)  # BUG: only storing path
    
    # Queue ready files
    for file_path in ready_files:
        del pending_files[file_path]
        # BUG: file_path not in pending_files anymore!
        pending_time = time.time() - pending_files.get(file_path, time.time())
        print(f"Buggy version - pending_time: {pending_time:.2f}s")

# Simulate the fixed version
def fixed_version():
    pending_files = {
        Path("/tmp/test.wav"): time.time() - 5.0
    }
    ready_files = []
    
    for file_path, detected_at in list(pending_files.items()):
        ready_files.append((file_path, detected_at))  # FIX: store tuple
    
    # Queue ready files
    for file_path, detected_at in ready_files:
        del pending_files[file_path]
        # FIX: we have detected_at from tuple
        pending_time = time.time() - detected_at
        print(f"Fixed version - pending_time: {pending_time:.2f}s")

if __name__ == "__main__":
    print("Testing buggy version:")
    try:
        buggy_version()
    except Exception as e:
        print(f"ERROR: {e}")
    
    print("\nTesting fixed version:")
    try:
        fixed_version()
    except Exception as e:
        print(f"ERROR: {e}")
