#!/usr/bin/env python3
"""
Blocking Error Analysis Script for Amazon ML Challenge 2026.
Analyzes the exact 1,671,394 missed true match pairs in train_ground_truth.tsv against candidate_pairs_train.tsv
and cleaned entity datasets to derive exact data-driven root causes, failure modes, and cap impacts.
"""

import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path("/Users/swapnil/Documents/ML/student_resource")
GO1_DIR = Path("/Users/swapnil/Documents/ML/go1_output")
CLEANED_DIR = GO1_DIR / "phase_2_cleaning" / "cleaned"
CANDIDATES_DIR = GO1_DIR / "phase_3_blocking" / "candidates"

GT_FILE = BASE_DIR / "dataset" / "train" / "train_ground_truth.tsv"
TRAIN_CAND_FILE = CANDIDATES_DIR / "candidate_pairs_train.tsv"

def run_error_analysis():
    print("Loading Ground Truth...")
    t0 = time.time()
    gt_df = pd.read_csv(GT_FILE, sep='\t', dtype=str, keep_default_na=False)
    
    gt_map = {}
    total_gt_pairs = 0
    for r in gt_df.itertuples():
        s1_id = r.source1_entity_id
        m_str = r.matched_entity_ids
        matches = set(x.strip() for x in m_str.split(',') if x.strip()) if m_str else set()
        if matches:
            gt_map[s1_id] = matches
            total_gt_pairs += len(matches)

    print(f"Loaded GT for {len(gt_map):,} S1 entities containing {total_gt_pairs:,} true match pairs.")

    print("Loading candidate_pairs_train.tsv...")
    cands_map = {}
    with open(TRAIN_CAND_FILE, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip("\n").split("\t")
            if len(parts) == 2:
                s1_id, cstr = parts
                cset = set(cstr.split(",")) if cstr else set()
                cands_map[s1_id] = cset

    print(f"Loaded candidates for {len(cands_map):,} S1 entities.")

    print("Loading cleaned entity metadata for deep diagnostic inspection...")
    df_s1 = pd.read_csv(CLEANED_DIR / "clean_train_source1.tsv", sep='\t', dtype=str, keep_default_na=False)
    df_s2 = pd.read_csv(CLEANED_DIR / "clean_train_source2.tsv", sep='\t', dtype=str, keep_default_na=False)
    df_s3 = pd.read_csv(CLEANED_DIR / "clean_train_source3.tsv", sep='\t', dtype=str, keep_default_na=False)

    s1_dict = {r.entity_id: r for r in df_s1.itertuples()}
    s23_dict = {}
    for r in df_s2.itertuples():
        s23_dict[r.entity_id] = r
    for r in df_s3.itertuples():
        s23_dict[r.entity_id] = r

    print("Analyzing missed true match pairs...")
    categories = defaultdict(int)
    s1_affected = defaultdict(set)
    examples = defaultdict(list)

    missed_count = 0
    analyzed_count = 0

    for s1_id, matches in gt_map.items():
        cset = cands_map.get(s1_id, set())
        missed = matches - cset
        if not missed:
            continue

        missed_count += len(missed)
        s1_obj = s1_dict.get(s1_id)

        for target_id in missed:
            analyzed_count += 1
            target_obj = s23_dict.get(target_id)

            category = "Other"

            if not s1_obj or not target_obj:
                category = "Missing Entity Record"
            elif len(cset) >= 350:
                s1_toks = set(s1_obj.name_significant_tokens.split())
                target_toks = set(target_obj.name_significant_tokens.split())
                s1_nums = set(s1_obj.address_numbers.split())
                target_nums = set(target_obj.address_numbers.split())

                if s1_toks.intersection(target_toks) or (s1_nums and s1_nums.intersection(target_nums)):
                    category = "1_Candidate_Cap_Truncation"
                else:
                    category = "2_Name_Token_Mismatch"
            else:
                s1_toks = set(s1_obj.name_significant_tokens.split())
                target_toks = set(target_obj.name_significant_tokens.split())
                s1_addr = set(s1_obj.address_tokens.split())
                target_addr = set(target_obj.address_tokens.split())

                if not target_obj.business_address:
                    category = "3_Missing_S23_Address"
                elif not s1_toks.intersection(target_toks):
                    category = "2_Name_Token_Mismatch"
                elif not s1_addr.intersection(target_addr):
                    category = "4_Address_Token_Mismatch"
                elif s1_obj.country_clean != target_obj.country_clean:
                    category = "5_Country_Partition_Mismatch"
                else:
                    category = "6_Phonetic_Prefix_Miss"

            categories[category] += 1
            s1_affected[category].add(s1_id)
            if len(examples[category]) < 3:
                examples[category].append({
                    "s1_id": s1_id,
                    "s1_name": s1_obj.business_name if s1_obj else "",
                    "target_id": target_id,
                    "target_name": target_obj.business_name if target_obj else "",
                    "cand_count": len(cset)
                })

    print("\n==================================================")
    print("MISSED MATCH ERROR ANALYSIS BREAKDOWN")
    print(f"Total True Match Pairs: {total_gt_pairs:,}")
    print(f"Total Missed True Match Pairs: {missed_count:,} ({(missed_count/total_gt_pairs*100):.2f}%)")
    print("==================================================")

    for cat, cnt in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        pct = (cnt / missed_count * 100) if missed_count > 0 else 0
        s1_cnt = len(s1_affected[cat])
        print(f"Category: {cat:<32} | Missed Pairs: {cnt:>8,} ({pct:>5.2f}%) | Affected S1: {s1_cnt:>7,}")
        for ex in examples[cat]:
            print(f"   Example: {ex['s1_id']} ({ex['s1_name']}) vs {ex['target_id']} ({ex['target_name']}) | Cands: {ex['cand_count']}")

run_error_analysis()
