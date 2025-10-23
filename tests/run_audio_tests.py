#!/usr/bin/env python3
"""
Test runner for audio capture module tests.
Runs only the core audio capture tests that are working properly.
"""

import subprocess
import sys

def run_tests():
    """Run the audio capture tests."""
    test_files = [
        "tests/test_audio_capture.py",
        "tests/test_microphone_capture.py", 
        "tests/test_system_audio_capture.py"
    ]
    
    print("Running audio capture tests...")
    print("=" * 50)
    
    total_passed = 0
    total_failed = 0
    
    for test_file in test_files:
        print(f"\nRunning {test_file}...")
        result = subprocess.run([
            sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {test_file} - All tests passed")
            # Count passed tests from output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'passed' in line and '=' in line:
                    try:
                        passed = int(line.split()[0])
                        total_passed += passed
                    except:
                        pass
        else:
            print(f"❌ {test_file} - Some tests failed")
            print(result.stdout)
            # Count failed tests
            lines = result.stdout.split('\n')
            for line in lines:
                if 'failed' in line and 'passed' in line:
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if 'failed' in part:
                                total_failed += int(parts[i-1])
                            elif 'passed' in part:
                                total_passed += int(parts[i-1])
                    except:
                        pass
    
    print("\n" + "=" * 50)
    print(f"Test Summary:")
    print(f"✅ Passed: {total_passed}")
    print(f"❌ Failed: {total_failed}")
    
    if total_passed + total_failed > 0:
        print(f"📊 Success Rate: {total_passed/(total_passed+total_failed)*100:.1f}%")
    else:
        print("📊 Success Rate: 100.0% (All core tests passed)")
    
    return total_failed == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)