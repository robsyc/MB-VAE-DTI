#!/usr/bin/env python3
"""Simple test script for all DTI model variants."""

import argparse
import subprocess
import sys
from pathlib import Path

def get_valid_combinations():
    """Generate all valid model/phase/dataset combinations."""
    combinations = []
    
    # baseline, multi_modal: finetune only
    for model in ["baseline", "multi_modal"]:
        for dataset in ["DAVIS", "KIBA"]:
            for split in ["rand", "cold"]:
                combinations.append([
                    "--model", model,
                    "--phase", "finetune", 
                    "--dataset", dataset,
                    "--split", split
                ])
    
    # multi_output: train + finetune
    for dataset in ["DAVIS", "KIBA"]:
        for split in ["rand", "cold"]:
            # train phase
            combinations.append([
                "--model", "multi_output",
                "--phase", "train",
                "--split", split
            ])
            # finetune phase
            combinations.append([
                "--model", "multi_output", 
                "--phase", "finetune",
                "--dataset", dataset,
                "--split", split
            ])
    
    # multi_hybrid, full: pretrain + train + finetune
    for model in ["multi_hybrid", "full"]:
        # pretrain phases
        for target in ["drug", "target"]:
            combinations.append([
                "--model", model,
                "--phase", "pretrain",
                "--pretrain_target", target
            ])
        
        # train + finetune phases
        for dataset in ["DAVIS", "KIBA"]:
            for split in ["rand", "cold"]:
                # train phase
                combinations.append([
                    "--model", model,
                    "--phase", "train", 
                    "--split", split
                ])
                # finetune phase
                combinations.append([
                    "--model", model,
                    "--phase", "finetune",
                    "--dataset", dataset,
                    "--split", split
                ])
    
    # remove duplicates without changing order (convert to tuples for hashing)
    seen = set()
    unique_combinations = []
    for combo in combinations:
        combo_tuple = tuple(combo)
        if combo_tuple not in seen:
            seen.add(combo_tuple)
            unique_combinations.append(combo)
    return unique_combinations

def get_combo_description(combo):
    """Get a readable description of the combination."""
    model = combo[combo.index("--model") + 1]
    phase = combo[combo.index("--phase") + 1]
    
    desc_parts = [model, phase]
    
    if "--dataset" in combo:
        dataset = combo[combo.index("--dataset") + 1]
        desc_parts.append(dataset)
    
    if "--split" in combo:
        split = combo[combo.index("--split") + 1]
        desc_parts.append(split)
        
    if "--pretrain_target" in combo:
        target = combo[combo.index("--pretrain_target") + 1]
        desc_parts.append(f"target={target}")
    
    return "_".join(desc_parts)

def run_test(args, test_type="single"):
    """Run a single test and return success status and error info."""
    cmd = ["python", str(Path(__file__).parent / "run.py")] + args + ["--debug"]
    
    if test_type == "gridsearch":
        cmd.extend(["--gridsearch", "--batch_index", "0", "--total_batches", "1"])
    elif test_type == "ensemble":
        cmd.append("--ensemble")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            return True, None, None
        else:
            # Extract last few lines of stderr for error info
            stderr_lines = result.stderr.strip().split('\n')[-3:] if result.stderr.strip() else []
            error_info = '\n'.join(stderr_lines) if stderr_lines else "No error details"
            return False, cmd, error_info
            
    except subprocess.TimeoutExpired:
        return False, cmd, "Timeout after 300 seconds"
    except Exception as e:
        return False, cmd, str(e)

def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description="Test DTI model variants")
    parser.add_argument("--skip", type=int, default=0, help="Number of tests to skip")
    args = parser.parse_args()
    
    combinations = get_valid_combinations()
    
    # Skip the specified number of combinations
    skip_count = min(args.skip, len(combinations))
    remaining_combinations = combinations[skip_count:]
    
    print(f"🚀 Testing {len(remaining_combinations)} combinations (skipping first {skip_count})")
    print(f"Running: single, gridsearch, ensemble for each")
    print(f"Total tests: {len(remaining_combinations) * 3}")
    
    results = {"single": [], "gridsearch": [], "ensemble": []}
    
    for i, combo in enumerate(remaining_combinations):
        desc = get_combo_description(combo)
        
        print(f"\n[{i+1}/{len(remaining_combinations)}] {desc}")
        
        # Test single run
        success, failed_cmd, error_info = run_test(combo, "single")
        results["single"].append(success)
        if success:
            print(f"  Single: ✅")
        else:
            print(f"  Single: ❌")
            print(f"    Command: {' '.join(failed_cmd)}")
            print(f"    Error: {error_info}")
        
        # Test gridsearch
        success, failed_cmd, error_info = run_test(combo, "gridsearch") 
        results["gridsearch"].append(success)
        if success:
            print(f"  Gridsearch: ✅")
        else:
            print(f"  Gridsearch: ❌")
            print(f"    Command: {' '.join(failed_cmd)}")
            print(f"    Error: {error_info}")
        
        # Test ensemble
        success, failed_cmd, error_info = run_test(combo, "ensemble")
        results["ensemble"].append(success)
        if success:
            print(f"  Ensemble: ✅")
        else:
            print(f"  Ensemble: ❌")
            print(f"    Command: {' '.join(failed_cmd)}")
            print(f"    Error: {error_info}")
    
    # Final report
    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")
    
    for test_type in ["single", "gridsearch", "ensemble"]:
        passed = sum(results[test_type])
        total = len(results[test_type])
        print(f"{test_type.capitalize()}: {passed}/{total} ({passed/total*100:.1f}%)")
    
    overall_passed = sum(sum(results[t]) for t in results)
    overall_total = sum(len(results[t]) for t in results)
    print(f"Overall: {overall_passed}/{overall_total} ({overall_passed/overall_total*100:.1f}%)")
    
    if overall_passed == overall_total:
        print("🎉 ALL TESTS PASSED!")
    else:
        print(f"⚠️  {overall_total - overall_passed} TESTS FAILED")

if __name__ == "__main__":
    main()
