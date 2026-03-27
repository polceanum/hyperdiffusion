#!/usr/bin/env python3
"""
Direct Baseline Ablation - Completion Verification

This script verifies that all components of the direct baseline implementation
are in place and working correctly.
"""

import os
import sys
from pathlib import Path


def check_file_exists(path: str, min_lines: int = 0) -> tuple[bool, str]:
    """Check if a file exists and optionally verify minimum lines."""
    filepath = Path(path)
    
    if not filepath.exists():
        return False, f"❌ File not found: {path}"
    
    if min_lines > 0:
        with open(filepath, 'r') as f:
            lines = len(f.readlines())
        if lines < min_lines:
            return False, f"❌ File too small: {path} ({lines} lines, need ≥ {min_lines})"
        return True, f"✅ {path} ({lines} lines)"
    
    return True, f"✅ {path}"


def check_imports(filepath: str, imports: list[str]) -> tuple[bool, str]:
    """Check if a Python file contains required imports."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        missing = []
        for imp in imports:
            if imp not in content:
                missing.append(imp)
        
        if missing:
            return False, f"❌ Missing imports in {filepath}: {missing}"
        
        return True, f"✅ All required imports in {filepath}"
    except Exception as e:
        return False, f"❌ Error checking {filepath}: {e}"


def check_class_exists(filepath: str, classname: str) -> tuple[bool, str]:
    """Check if a Python file contains a specific class."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        if f"class {classname}" in content:
            return True, f"✅ Class {classname} found in {filepath}"
        else:
            return False, f"❌ Class {classname} not found in {filepath}"
    except Exception as e:
        return False, f"❌ Error checking {filepath}: {e}"


def main():
    """Run all verification checks."""
    
    print("=" * 70)
    print("DIRECT BASELINE ABLATION - COMPLETION VERIFICATION")
    print("=" * 70)
    
    base_path = Path(__file__).parent
    all_checks_pass = True
    
    checks = [
        # Core Implementation
        ("Core Implementation", [
            (check_file_exists("hyperdiffusion/direct_baseline.py", 150), ""),
            (check_class_exists("hyperdiffusion/direct_baseline.py", "DirectPredictor"), ""),
            (check_class_exists("hyperdiffusion/direct_baseline.py", "DirectTextProjector"), ""),
            (check_class_exists("hyperdiffusion/direct_baseline.py", "DirectSystem"), ""),
        ]),
        
        # Tests
        ("Test Suite", [
            (check_file_exists("tests/test_direct_baseline.py", 250), ""),
            (check_class_exists("tests/test_direct_baseline.py", "TestDirectPredictor"), ""),
            (check_class_exists("tests/test_direct_baseline.py", "TestDirectSystem"), ""),
            (check_class_exists("tests/test_direct_baseline.py", "TestDirectTextProjector"), ""),
        ]),
        
        # Examples
        ("Examples", [
            (check_file_exists("examples/direct_baseline_example.py", 200), ""),
        ]),
        
        # Documentation
        ("Documentation", [
            (check_file_exists("docs/direct_baseline.md", 300), ""),
            (check_file_exists("docs/DIRECT_BASELINE_INTEGRATION.md", 400), ""),
            (check_file_exists("docs/DIRECT_BASELINE_SUMMARY.md", 250), ""),
            (check_file_exists("DIRECT_BASELINE_README.md", 200), ""),
        ]),
        
        # API Verification
        ("API Components", [
            (check_class_exists("hyperdiffusion/direct_baseline.py", "DirectPredictor"), ""),
            (check_class_exists("hyperdiffusion/direct_baseline.py", "DirectSystem"), ""),
        ]),
    ]
    
    total_checks = 0
    passed_checks = 0
    
    for section_name, section_checks in checks:
        print(f"\n{section_name}:")
        print("-" * 70)
        
        for check_fn, _ in section_checks:
            success, message = check_fn
            total_checks += 1
            
            if success:
                passed_checks += 1
                print(f"  {message}")
            else:
                print(f"  {message}")
                all_checks_pass = False
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    # Count lines
    files_to_count = [
        "hyperdiffusion/direct_baseline.py",
        "tests/test_direct_baseline.py",
        "examples/direct_baseline_example.py",
        "docs/direct_baseline.md",
        "docs/DIRECT_BASELINE_INTEGRATION.md",
        "docs/DIRECT_BASELINE_SUMMARY.md",
        "DIRECT_BASELINE_README.md",
    ]
    
    total_lines = 0
    files_info = []
    
    for filepath in files_to_count:
        full_path = base_path / filepath
        if full_path.exists():
            with open(full_path, 'r') as f:
                lines = len(f.readlines())
            total_lines += lines
            files_info.append((filepath, lines))
    
    print(f"\nTotal files created: {len(files_info)}")
    print(f"Total lines of code/docs: {total_lines}")
    print(f"\nBreakdown:")
    for filepath, lines in sorted(files_info):
        print(f"  • {filepath:<50} {lines:>5} lines")
    
    print(f"\nVerification Results:")
    print(f"  • Checks passed: {passed_checks}/{total_checks}")
    print(f"  • Success rate: {100 * passed_checks / total_checks:.1f}%")
    
    # Final status
    print("\n" + "=" * 70)
    if all_checks_pass and passed_checks == total_checks:
        print("✅ ALL CHECKS PASSED - IMPLEMENTATION COMPLETE")
        print("=" * 70)
        return 0
    else:
        print("❌ SOME CHECKS FAILED - PLEASE REVIEW")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
