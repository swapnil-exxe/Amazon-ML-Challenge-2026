#!/usr/bin/env python3
"""
Blocking Recall Audit Script (v2 Baseline Assessment) — ML Challenge 2026
Calculates exact Ground Truth candidate-blocking recall on train dataset and performs missing match error analysis.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

GT_PATH = '/Users/swapnil/Documents/ML/student_resource/dataset/train/train_ground_truth.tsv'
TRAIN_S1_PATH = '/Users/swapnil/Documents/ML/student_resource/dataset/train/train_source1.tsv'
TRAIN_S2_PATH = '/Users/swapnil/Documents/ML/student_resource/dataset/train/train_source2.tsv'
TRAIN_S3_PATH = '/Users/swapnil/Documents/ML/student_resource/dataset/train/train_source3.tsv'
CAND_TRAIN_PATH = '/Users/swapnil/Documents/ML/go1_output/phase_3_blocking/candidates/candidate_pairs_train_v2.tsv'

print("=== PHASE 2 — MEASURING BASELINE BLOCKING RECALL ===")
t0 = time.time()

# 1. Load S1 country map
print("Loading train_source1.tsv country labels...")
s1_df = pd.read_csv(TRAIN_S1_PATH, sep='\t', usecols=['entity_id', 'country'], dtype=str)
s1_country = dict(zip(s1_df['entity_id'], s1_df['country']))

# 2. Load Ground Truth
print("Loading train_ground_truth.tsv...")
gt_df = pd.read_csv(GT_PATH, sep='\t', dtype=str, keep_default_na=False)

gt_pairs = {} # s1_id -> set of true matched_ids
gt_country_pairs = {'US': 0, 'India': 0, 'France': 0}
total_gt_pairs = 0

for r in gt_df.itertuples():
    s1_id = r.source1_entity_id
    m_str = r.matched_entity_ids
    matches = set(x.strip() for x in m_str.split(',') if x.strip()) if m_str else set()
    if matches:
        gt_pairs[s1_id] = matches
        total_gt_pairs += len(matches)
        cntry = s1_country.get(s1_id, 'US')
        if cntry in gt_country_pairs:
            gt_country_pairs[cntry] += len(matches)

print(f"Total S1 entities in GT: {len(gt_df):,}")
print(f"Total True Matched Pairs: {total_gt_pairs:,}")
print(f"  - US True Pairs: {gt_country_pairs['US']:,}")
print(f"  - India True Pairs: {gt_country_pairs['India']:,}")
print(f"  - France True Pairs: {gt_country_pairs['France']:,}")

# 3. Stream candidate_pairs_train_v2.tsv
print(f"Reading candidates from {CAND_TRAIN_PATH}...")
hit_count = 0
hit_country_count = {'US': 0, 'India': 0, 'France': 0}
cand_counts = []
zero_cand_s1 = 0

chunk_size = 200000
for chunk in pd.read_csv(CAND_TRAIN_PATH, sep='\t', chunksize=chunk_size, dtype=str, keep_default_na=False):
    for r in chunk.itertuples():
        s1_id = r.source1_entity_id
        c_str = r.candidate_entity_ids
        cands = set(x.strip() for x in c_str.split(',') if x.strip()) if c_str else set()
        c_len = len(cands)
        cand_counts.append(c_len)
        if c_len == 0:
            zero_cand_s1 += 1
        
        if s1_id in gt_pairs:
            true_set = gt_pairs[s1_id]
            found = true_set.intersection(cands)
            f_len = len(found)
            hit_count += f_len
            cntry = s1_country.get(s1_id, 'US')
            if cntry in hit_country_count:
                hit_country_count[cntry] += f_len

cand_arr = np.array(cand_counts, dtype=np.int32)
mean_cands = float(np.mean(cand_arr))
median_cands = float(np.median(cand_arr))
p95_cands = float(np.percentile(cand_arr, 95))
max_cands = int(np.max(cand_arr))
zero_pct = (zero_cand_s1 / len(cand_arr)) * 100

overall_recall = (hit_count / total_gt_pairs) * 100
us_recall = (hit_country_count['US'] / gt_country_pairs['US']) * 100 if gt_country_pairs['US'] > 0 else 0.0
india_recall = (hit_country_count['India'] / gt_country_pairs['India']) * 100 if gt_country_pairs['India'] > 0 else 0.0

print("\n=== BASELINE BLOCKING RECALL RESULTS ===")
print(f"Total True Pairs: {total_gt_pairs:,}")
print(f"Reachable True Pairs: {hit_count:,}")
print(f"Missing True Pairs: {total_gt_pairs - hit_count:,}")
print(f"Overall Blocking Recall: {overall_recall:.4f}%")
print(f"US Blocking Recall: {us_recall:.4f}% ({hit_country_count['US']:,} / {gt_country_pairs['US']:,})")
print(f"India Blocking Recall: {india_recall:.4f}% ({hit_country_count['India']:,} / {gt_country_pairs['India']:,})")
print("France: not quantitatively measurable because labelled ground truth is unavailable.")

print("\n=== CANDIDATE DISTRIBUTION STATS ===")
print(f"Total S1 entities evaluated: {len(cand_arr):,}")
print(f"Total Candidates generated: {np.sum(cand_arr):,}")
print(f"Average candidates / S1: {mean_cands:.2f}")
print(f"Median candidates / S1: {median_cands:.1f}")
print(f"P95 candidates / S1: {p95_cands:.1f}")
print(f"Maximum candidates / S1: {max_cands}")
print(f"Zero candidates count: {zero_cand_s1:,} ({zero_pct:.2f}%)")
print(f"Elapsed time: {time.time() - t0:.2f}s")
