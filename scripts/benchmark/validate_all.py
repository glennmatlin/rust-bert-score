#!/usr/bin/env python3
"""
Comprehensive validation suite for BERTScore.
Runs all tests and generates a detailed report.
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
os.chdir(project_dir)

# Test modules to run
TEST_MODULES = [
    ("Edge Cases", "test_edge_cases.py"),
    ("Configuration Matrix", "test_configurations.py"),
    ("Numerical Stability", "test_numerical_stability.py"),
]

def check_prerequisites():
    """Check if all prerequisites are met."""
    issues = []
    
    # Check Python dependencies
    try:
        import bert_score
        import pandas
        import numpy
        import scipy
    except ImportError as e:
        issues.append(f"Missing Python dependency: {e}")
    
    # Check Rust binary
    rust_binary = "./target/release/bert-score"
    if not os.path.exists(rust_binary):
        issues.append(f"Rust binary not found: {rust_binary}")
        issues.append("  Run: cargo build --release")
    
    # Check libtorch
    libtorch_path = "/home/gmatlin/Codespace/rust-bert-score/libtorch"
    if not os.path.exists(libtorch_path):
        issues.append(f"libtorch not found: {libtorch_path}")
        issues.append("  Download and extract libtorch 2.4.0")
    
    # Check test data
    test_data = "data/benchmark/direct_eval_pairs.tsv"
    if not os.path.exists(test_data):
        issues.append(f"Test data not found: {test_data}")
    
    return issues

def run_test_module(name: str, module: str) -> dict:
    """Run a test module and capture results."""
    print(f"\n{'='*80}")
    print(f"Running {name}")
    print(f"{'='*80}")
    
    start_time = datetime.now()
    
    # Run the test
    result = subprocess.run(
        ["python", f"scripts/benchmark/{module}"],
        capture_output=True,
        text=True
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Parse output for summary
    output_lines = result.stdout.split('\n')
    stderr_lines = result.stderr.split('\n')
    
    # Try to extract pass/fail statistics
    passed = failed = total = 0
    for line in output_lines:
        if "passed" in line.lower() and "/" in line:
            try:
                # Look for patterns like "45/50 passed"
                parts = line.split()
                for part in parts:
                    if "/" in part:
                        nums = part.split("/")
                        if len(nums) == 2 and nums[0].isdigit() and nums[1].isdigit():
                            passed = int(nums[0])
                            total = int(nums[1])
                            failed = total - passed
                            break
            except:
                pass
    
    return {
        "name": name,
        "module": module,
        "success": result.returncode == 0,
        "duration": duration,
        "passed": passed,
        "failed": failed,
        "total": total,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "return_code": result.returncode
    }

def generate_summary_report(results: list, report_dir: str):
    """Generate a comprehensive summary report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_lines = [
        "="*80,
        "BERTSCORE VALIDATION REPORT",
        "="*80,
        f"Generated: {timestamp}",
        "",
        "EXECUTIVE SUMMARY",
        "-"*40,
    ]
    
    # Overall statistics
    total_tests = sum(r["total"] for r in results)
    total_passed = sum(r["passed"] for r in results)
    total_failed = sum(r["failed"] for r in results)
    total_modules_passed = sum(1 for r in results if r["success"])
    
    report_lines.extend([
        f"Total test modules: {len(results)}",
        f"Modules passed: {total_modules_passed}/{len(results)}",
        f"Total tests run: {total_tests}",
        f"Tests passed: {total_passed} ({total_passed/total_tests*100:.1f}%)" if total_tests > 0 else "Tests passed: N/A",
        f"Tests failed: {total_failed} ({total_failed/total_tests*100:.1f}%)" if total_tests > 0 else "Tests failed: N/A",
        "",
    ])
    
    # Per-module results
    report_lines.extend([
        "MODULE RESULTS",
        "-"*40,
    ])
    
    for result in results:
        status = "✓ PASS" if result["success"] else "✗ FAIL"
        report_lines.append(
            f"\n{result['name']} ({result['module']}):"
        )
        report_lines.append(f"  Status: {status}")
        report_lines.append(f"  Duration: {result['duration']:.1f}s")
        if result["total"] > 0:
            report_lines.append(
                f"  Tests: {result['passed']}/{result['total']} passed "
                f"({result['passed']/result['total']*100:.1f}%)"
            )
        else:
            report_lines.append("  Tests: No statistics available")
        
        if result["return_code"] != 0:
            report_lines.append(f"  Return code: {result['return_code']}")
            if result["stderr"]:
                report_lines.append(f"  Error: {result['stderr'].split(chr(10))[0][:100]}...")
    
    # Check for specific output files
    report_lines.extend([
        "",
        "OUTPUT FILES",
        "-"*40,
    ])
    
    expected_outputs = [
        ("Edge case results", "reports/edge_case_validation.csv"),
        ("Configuration matrix", "reports/configuration_matrix_results.csv"),
        ("Numerical stability", "reports/numerical_stability_results.csv"),
        ("Direct comparison", "reports/direct_agreement.txt"),
    ]
    
    for desc, path in expected_outputs:
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024  # KB
            report_lines.append(f"✓ {desc}: {path} ({size:.1f} KB)")
        else:
            report_lines.append(f"✗ {desc}: {path} (not found)")
    
    # Recommendations
    report_lines.extend([
        "",
        "RECOMMENDATIONS",
        "-"*40,
    ])
    
    if total_failed > 0:
        report_lines.extend([
            "1. Review failed test cases in the detailed reports",
            "2. Check numerical precision settings",
            "3. Verify model weights are identical between implementations",
            "4. Ensure tokenization is consistent",
        ])
    else:
        report_lines.extend([
            "✓ All tests passed! The Rust implementation matches Python bert_score.",
            "Consider running additional stress tests with larger datasets.",
        ])
    
    # Write report
    report_content = "\n".join(report_lines)
    report_path = os.path.join(report_dir, "validation_summary.txt")
    with open(report_path, "w") as f:
        f.write(report_content)
    
    print(report_content)
    print(f"\n✓ Summary report saved to: {report_path}")

def main():
    """Run all validation tests."""
    print("🚀 BERTScore Comprehensive Validation Suite")
    print("="*60)
    
    # Check prerequisites
    print("\nChecking prerequisites...")
    issues = check_prerequisites()
    if issues:
        print("\n❌ Prerequisites not met:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)
    print("✓ All prerequisites met")
    
    # Create reports directory
    os.makedirs("reports", exist_ok=True)
    
    # Run all test modules
    results = []
    for name, module in TEST_MODULES:
        try:
            result = run_test_module(name, module)
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error running {name}: {e}")
            results.append({
                "name": name,
                "module": module,
                "success": False,
                "duration": 0,
                "passed": 0,
                "failed": 0,
                "total": 0,
                "stdout": "",
                "stderr": str(e),
                "return_code": -1
            })
    
    # Run direct comparison as a special case
    print(f"\n{'='*80}")
    print("Running Direct Comparison")
    print(f"{'='*80}")
    
    try:
        # First run Python scoring
        py_result = subprocess.run(
            ["uv", "run", "python", "scripts/benchmark/run_direct_py.py"],
            capture_output=True,
            text=True
        )
        
        if py_result.returncode == 0:
            # Then run comparison
            comp_result = subprocess.run(
                ["uv", "run", "python", "scripts/benchmark/compare_direct.py"],
                capture_output=True,
                text=True
            )
            
            # Extract statistics from output
            output = comp_result.stdout
            passed = "ALL TESTS PASS" in output
            
            results.append({
                "name": "Direct Comparison",
                "module": "compare_direct.py",
                "success": comp_result.returncode == 0 and passed,
                "duration": 0,
                "passed": 21 if passed else 0,
                "failed": 0 if passed else 21,
                "total": 21,
                "stdout": comp_result.stdout,
                "stderr": comp_result.stderr,
                "return_code": comp_result.returncode
            })
    except Exception as e:
        print(f"Error in direct comparison: {e}")
    
    # Generate summary report
    generate_summary_report(results, "reports")
    
    # Save detailed results
    results_json = os.path.join("reports", "validation_results.json")
    with open(results_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Detailed results saved to: {results_json}")
    
    # Exit with appropriate code
    all_passed = all(r["success"] for r in results)
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()