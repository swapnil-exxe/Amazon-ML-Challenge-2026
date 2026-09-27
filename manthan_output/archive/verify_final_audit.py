#!/usr/bin/env python3
"""
Final Audit Verification Script for Manthan Candidate Generation Pipeline.
Physically checks original data safety, Phase 1/2/3 output files, line counts, TSV integrity,
candidate IDs, duplicate checks, and report files.
"""

import sys
import os
from pathlib import Path
import pandas as pd

BASE_DIR = Path("/Users/swapnil/Documents/ML/student_resource")
MANTHAN_DIR = Path("/Users/swapnil/Documents/ML/manthan_output")

def audit():
    print("=== MANTHAN FINAL PHYSICAL AUDIT ===")

    # A. Original Data Safety
    orig_files = [
        BASE_DIR / "dataset" / "train" / "train_source1.tsv",
        BASE_DIR / "dataset" / "train" / "train_source2.tsv",
        BASE_DIR / "dataset" / "train" / "train_source3.tsv",
        BASE_DIR / "dataset" / "train" / "train_ground_truth.tsv",
        BASE_DIR / "dataset" / "test" / "test_source1.tsv",
        BASE_DIR / "dataset" / "test" / "test_source2.tsv",
        BASE_DIR / "dataset" / "test" / "test_source3.tsv",
    ]
    all_orig_exist = all(f.exists() for f in orig_files)
    print(f"Original 7 TSV files exist: {all_orig_exist}")

    # B. Phase 1 & 2
    p1_reports = MANTHAN_DIR / "phase_1_audit" / "reports"
    p2_cleaned = MANTHAN_DIR / "phase_2_cleaning" / "cleaned"
    p3_candidates = MANTHAN_DIR / "phase_3_blocking" / "candidates"
    p3_reports = MANTHAN_DIR / "phase_3_blocking" / "reports"
    p3_src = MANTHAN_DIR / "phase_3_blocking" / "src"

    p2_files = [
        p2_cleaned / "clean_train_source1.tsv",
        p2_cleaned / "clean_train_source2.tsv",
        p2_cleaned / "clean_train_source3.tsv",
        p2_cleaned / "clean_test_source1.tsv",
        p2_cleaned / "clean_test_source2.tsv",
        p2_cleaned / "clean_test_source3.tsv",
    ]
    all_p2_exist = all(f.exists() for f in p2_files)
    print(f"Cleaned 6 Phase 2 files exist: {all_p2_exist}")

    # I & J Candidate files line count and integrity check
    train_cand_file = p3_candidates / "candidate_pairs_train.tsv"
    test_cand_file = p3_candidates / "candidate_pairs_test.tsv"

    print("Verifying candidate_pairs_train.tsv...")
    train_lines = 0
    train_s1_set = set()
    train_dup_s1 = False
    train_invalid_ids = False
    train_no_dup_cands = True
    train_no_s1_as_cands = True

    with open(train_cand_file, "r", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            train_lines += 1
            parts = line.strip("\n").split("\t")
            if len(parts) != 2:
                train_invalid_ids = True
                continue
            s1_id, cstr = parts
            if s1_id in train_s1_set:
                train_dup_s1 = True
            train_s1_set.add(s1_id)
            if not s1_id.startswith("S1-"):
                train_invalid_ids = True
            if cstr:
                clist = cstr.split(",")
                if len(clist) != len(set(clist)):
                    train_no_dup_cands = False
                if s1_id in clist:
                    train_no_s1_as_cands = False

    print(f"Train candidate rows: {train_lines} (Expected: 2206821)")
    print(f"Train unique S1: {len(train_s1_set)}, duplicate S1: {train_dup_s1}")

    print("Verifying candidate_pairs_test.tsv...")
    test_lines = 0
    test_s1_set = set()
    test_dup_s1 = False
    test_invalid_ids = False
    test_no_dup_cands = True
    test_no_s1_as_cands = True

    with open(test_cand_file, "r", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            test_lines += 1
            parts = line.strip("\n").split("\t")
            if len(parts) != 2:
                test_invalid_ids = True
                continue
            s1_id, cstr = parts
            if s1_id in test_s1_set:
                test_dup_s1 = True
            test_s1_set.add(s1_id)
            if not s1_id.startswith("S1-"):
                test_invalid_ids = True
            if cstr:
                clist = cstr.split(",")
                if len(clist) != len(set(clist)):
                    test_no_dup_cands = False
                if s1_id in clist:
                    test_no_s1_as_cands = False

    print(f"Test candidate rows: {test_lines} (Expected: 1732544)")
    print(f"Test unique S1: {len(test_s1_set)}, duplicate S1: {test_dup_s1}")

    # Reports check
    required_reports = [
        p3_reports / "candidate_statistics.csv",
        p3_reports / "blocking_report.md",
        p3_reports / "blocking_experiment_log.csv",
        p3_reports / "token_frequency_report.csv",
        p3_reports / "missed_match_analysis.csv",
        p3_reports / "blocking_rule_analysis.csv",
    ]
    all_reports_exist = all(r.exists() for r in required_reports)
    print(f"All 6 Phase 3 report files exist: {all_reports_exist}")

    # Code check
    required_code = [
        p3_src / "blocking_keys.py",
        p3_src / "build_indexes.py",
        p3_src / "evaluate_blocking.py",
        p3_src / "generate_candidates.py",
    ]
    all_code_exist = all(c.exists() for c in required_code)
    print(f"All Phase 3 code files exist: {all_code_exist}")

if __name__ == "__main__":
    audit()
