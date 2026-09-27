#!/usr/bin/env python3
"""
Quantitative Missed True-Pair Blocking Failure Analysis — ML Challenge 2026
Categorizes why the ~1.67M true matches were missed by candidate generation.
"""

import os
import sys
import time
import pandas as pd
import numpy as np

GT_PATH = '/Users/swapnil/Documents/ML/student_resource/dataset/train/train_ground_truth.tsv'
CAND_TRAIN_PATH = '/Users/swapnil/Documents/ML/manthan_output/phase_3_blocking/candidates/candidate_pairs_train_v2.tsv'

print("=== QUANTITATIVE BLOCKING FAILURE ANALYSIS ===")
t0 = time.time()

# Load Ground Truth
gt_df = pd.read_csv(GT_PATH, sep='\t', dtype=str, keep_default_na=False)
gt_pairs = {}
total_gt = 0
for r in gt_df.itertuples():
    s1_id = r.source1_entity_id
    m_str = r.matched_entity_ids
    matches = set(x.strip() for x in m_str.split(',') if x.strip()) if m_str else set()
    if matches:
        gt_pairs[s1_id] = matches
        total_gt += len(matches)

print(f"Loaded {total_gt:,} total GT pairs across {len(gt_pairs):,} S1 entities.")

# Sample S1 entities for detailed failure breakdown
sample_s1 = list(gt_pairs.keys())[:50000]
sample_gt_pairs = {s: gt_pairs[s] for s in sample_s1}
sample_total_gt = sum(len(v) for v in sample_gt_pairs.values())

missed_count = 0
cap_pruned_count = 0
zero_cand_count = 0
sparse_cand_count = 0

print(f"Scanning candidate pairs for {len(sample_s1):,} sample queries...")

chunk_size = 100000
for chunk in pd.read_csv(CAND_TRAIN_PATH, sep='\t', chunksize=chunk_size, dtype=str, keep_default_na=False):
    for r in chunk.itertuples():
        s1_id = r.source1_entity_id
        if s1_id not in sample_gt_pairs:
            continue
        c_str = r.candidate_entity_ids
        cands = set(x.strip() for x in c_str.split(',') if x.strip()) if c_str else set()
        c_len = len(cands)
        true_set = sample_gt_pairs[s1_id]
        
        missed = true_set - cands
        if missed:
            missed_count += len(missed)
            if c_len == 350:
                cap_pruned_count += len(missed)
            elif c_len == 0:
                zero_cand_count += len(missed)
            else:
                sparse_cand_count += len(missed)

print("\n=== QUANTITATIVE FAILURE BREAKDOWN ===")
print(f"Sample Total GT True Pairs: {sample_total_gt:,}")
print(f"Sample Missed True Pairs: {missed_count:,} ({missed_count/sample_total_gt*100:.2f}%)")
print(f"  1. Candidate Cap (350 Cap Pruning): {cap_pruned_count:,} ({cap_pruned_count/missed_count*100:.2f}%)")
print(f"  2. Zero Candidates (No Index Match / Transliteration): {zero_cand_count:,} ({zero_cand_count/missed_count*100:.2f}%)")
print(f"  3. Index Pruning / Token Filtering (Sparse list < 350): {sparse_cand_count:,} ({sparse_cand_count/missed_count*100:.2f}%)")
print(f"Analysis Runtime: {time.time() - t0:.2f}s")
