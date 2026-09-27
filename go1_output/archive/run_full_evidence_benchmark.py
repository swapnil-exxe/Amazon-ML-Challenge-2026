#!/usr/bin/env python3
"""
Full Evidence-Based Candidate Ranking & Capping Benchmark Script.
Evaluates recall, coverage, missed matches, and storage across candidate caps (100 to 1000 and Uncapped)
using deterministic evidence-based scoring vs. random truncation.
"""

import sys
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

CLEANED_DIR = Path("/Users/swapnil/Documents/ML/go1_output/phase_2_cleaning/cleaned")
GT_FILE = Path("/Users/swapnil/Documents/ML/student_resource/dataset/train/train_ground_truth.tsv")
REPORTS_DIR = Path("/Users/swapnil/Documents/ML/go1_output/phase_3_blocking/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading cleaned train files...")
t0 = time.time()
train_s1 = pd.read_csv(CLEANED_DIR / "clean_train_source1.tsv", sep='\t', dtype=str, keep_default_na=False)
train_s2 = pd.read_csv(CLEANED_DIR / "clean_train_source2.tsv", sep='\t', dtype=str, keep_default_na=False)
train_s3 = pd.read_csv(CLEANED_DIR / "clean_train_source3.tsv", sep='\t', dtype=str, keep_default_na=False)
print(f"Loaded train files in {time.time()-t0:.2f}s")

# Load Ground Truth
gt_df = pd.read_csv(GT_FILE, sep='\t', dtype=str, keep_default_na=False)
gt_map = {}
total_gt_pairs = 0
for r in gt_df.itertuples():
    m_str = r.matched_entity_ids
    matches = set(x.strip() for x in m_str.split(',') if x.strip()) if m_str else set()
    if matches:
        gt_map[r.source1_entity_id] = matches
        total_gt_pairs += len(matches)

print(f"Loaded GT for {len(gt_map):,} S1 entities containing {total_gt_pairs:,} total true match pairs.")

def run_benchmark():
    print("Building inverted indexes and token maps...")
    tb0 = time.time()
    s23_meta = {}
    name_idx = defaultdict(set)
    addr_idx = defaultdict(set)
    num_idx = defaultdict(set)
    prefix_idx = defaultdict(set)

    for df in (train_s2, train_s3):
        for r in df.itertuples():
            eid = r.entity_id
            country = r.country_clean
            sig_toks = set(t for t in r.name_significant_tokens.split() if len(t) >= 2)
            addr_toks = set(t for t in r.address_tokens.split() if len(t) >= 3)
            addr_nums = set(n for n in r.address_numbers.split() if len(n) >= 1)
            norm_name = r.business_name_clean

            s23_meta[eid] = (country, sig_toks, addr_toks, addr_nums, norm_name)

            for t in sig_toks:
                if len(t) >= 2:
                    name_idx[(country, t)].add(eid)
                    if len(t) >= 4:
                        prefix_idx[(country, t[:4])].add(eid)
            for t in addr_toks:
                if len(t) >= 3:
                    addr_idx[(country, t)].add(eid)
            for num in addr_nums:
                if len(num) >= 1:
                    num_idx[(country, num)].add(eid)

    f_name = {k: v for k, v in name_idx.items() if len(v) <= 3000}
    f_addr = {k: v for k, v in addr_idx.items() if len(v) <= 1000}
    f_num = {k: v for k, v in num_idx.items() if len(v) <= 1000}
    f_prefix = {k: v for k, v in prefix_idx.items() if len(v) <= 1000}

    print(f"Index build complete in {time.time()-tb0:.2f}s.")

    # Benchmark on 100,000 S1 sample
    sample_s1 = train_s1.sample(n=100000, random_state=42)
    sample_eids = set(sample_s1['entity_id'])
    sub_gt_count = sum(len(gt_map[eid]) for eid in sample_eids if eid in gt_map)
    sub_gt_s1_count = sum(1 for eid in sample_eids if eid in gt_map)

    caps_to_test = [100, 150, 200, 250, 300, 350, 500, 750, 1000, None]
    results = []

    print("\nRunning benchmark across candidate caps with Evidence-Based Ranking vs Random Truncation...")

    for cap in caps_to_test:
        matched_pairs = 0
        all_matched_s1 = 0
        total_cands = 0
        cand_counts = []

        for r in sample_s1.itertuples():
            s1_id = r.entity_id
            country = r.country_clean
            sig_toks = set(t for t in r.name_significant_tokens.split() if len(t) >= 2)
            addr_toks = set(t for t in r.address_tokens.split() if len(t) >= 3)
            addr_nums = set(n for n in r.address_numbers.split() if len(n) >= 1)
            norm_name = r.business_name_clean

            cand_hits = defaultdict(float)

            for t in sig_toks:
                if (country, t) in f_name:
                    for cid in f_name[(country, t)]:
                        cand_hits[cid] += 3.0

            for t in addr_toks:
                if (country, t) in f_addr:
                    for cid in f_addr[(country, t)]:
                        cand_hits[cid] += 1.5

            for num in addr_nums:
                if (country, num) in f_num:
                    for cid in f_num[(country, num)]:
                        cand_hits[cid] += 2.0

            if len(cand_hits) < 5:
                for t in sig_toks:
                    if len(t) >= 4 and (country, t[:4]) in f_prefix:
                        for cid in f_prefix[(country, t[:4])]:
                            cand_hits[cid] += 1.0

            raw_cands = set(cand_hits.keys())

            if cap is None:
                final_cands = raw_cands
            else:
                scored_cands = []
                for cid in raw_cands:
                    sc = cand_hits[cid]
                    c_country, c_sig, c_addr, c_nums, c_name = s23_meta[cid]
                    if norm_name and norm_name == c_name:
                        sc += 10.0
                    scored_cands.append((sc, cid))
                scored_cands.sort(key=lambda x: (-x[0], x[1]))
                final_cands = set(x[1] for x in scored_cands[:cap])

            nc = len(final_cands)
            cand_counts.append(nc)
            total_cands += nc

            gt_targets = gt_map.get(s1_id)
            if gt_targets:
                hit = len(gt_targets.intersection(final_cands))
                matched_pairs += hit
                if hit == len(gt_targets):
                    all_matched_s1 += 1

        pair_rec = matched_pairs / sub_gt_count * 100 if sub_gt_count > 0 else 0
        s1_cov = all_matched_s1 / sub_gt_s1_count * 100 if sub_gt_s1_count > 0 else 0
        s = pd.Series(cand_counts)
        mean_c = round(float(s.mean()), 2)
        med_c = float(s.median())
        p90_c = float(s.quantile(0.9))
        p95_c = float(s.quantile(0.95))
        p99_c = float(s.quantile(0.99))
        max_c = int(s.max())
        zero_c = int((s == 0).sum())
        est_total_gb = round(mean_c * 3.938e6 * 12 / 1e9, 2)

        cap_str = str(cap) if cap is not None else "Uncapped"
        results.append({
            "candidate_cap": cap_str,
            "ranking_method": "Deterministic Evidence-Based Scoring" if cap is not None else "None",
            "pair_recall": round(pair_rec, 4),
            "s1_coverage": round(s1_cov, 4),
            "missed_true_matches": sub_gt_count - matched_pairs,
            "mean_candidates": mean_c,
            "median": med_c,
            "p90": p90_c,
            "p95": p95_c,
            "p99": p99_c,
            "max": max_c,
            "zero_candidates": zero_c,
            "est_total_disk_gb": est_total_gb
        })

        print(f"Cap: {cap_str:>8} | Pair Recall: {pair_rec:.2f}% | S1 Coverage: {s1_cov:.2f}% | Mean Cands: {mean_c:.1f} | Est Disk: {est_total_gb:.2f} GB")

    df_res = pd.DataFrame(results)
    df_res.to_csv(REPORTS_DIR / "evidence_ranking_caps_benchmark.csv", index=False)
    print(f"\nSaved benchmark results to {REPORTS_DIR / 'evidence_ranking_caps_benchmark.csv'}")

if __name__ == "__main__":
    run_benchmark()
