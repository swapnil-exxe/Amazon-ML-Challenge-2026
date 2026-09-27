#!/usr/bin/env python3
import time
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

CLEANED_DIR = Path("/Users/swapnil/Documents/ML/go1_output/phase_2_cleaning/cleaned")
GT_FILE = Path("/Users/swapnil/Documents/ML/student_resource/dataset/train/train_ground_truth.tsv")

print("Loading cleaned train files...")
t0 = time.time()
train_s1 = pd.read_csv(CLEANED_DIR / "clean_train_source1.tsv", sep='\t', dtype=str, keep_default_na=False)
train_s2 = pd.read_csv(CLEANED_DIR / "clean_train_source2.tsv", sep='\t', dtype=str, keep_default_na=False)
train_s3 = pd.read_csv(CLEANED_DIR / "clean_train_source3.tsv", sep='\t', dtype=str, keep_default_na=False)

# Load GT
gt_df = pd.read_csv(GT_FILE, sep='\t', dtype=str, keep_default_na=False)
gt_map = {}
for r in gt_df.itertuples():
    m_str = r.matched_entity_ids
    matches = set(x.strip() for x in m_str.split(',') if x.strip()) if m_str else set()
    if matches:
        gt_map[r.source1_entity_id] = matches

def test_ranking():
    print("Building inverted indexes and token dictionaries...")
    # Store token maps for S2/S3 candidate entities for fast scoring
    s23_meta = {} # eid -> (country, sig_toks_set, addr_toks_set, addr_nums_set, norm_name)
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

    print("Evaluating ranking strategies on 50,000 S1 sample...")
    sample_s1 = train_s1.sample(n=50000, random_state=42)

    total_gt = sum(len(gt_map[r.entity_id]) for r in sample_s1.itertuples() if r.entity_id in gt_map)

    # Strategy 1: Uncapped (Baseline Result A)
    # Strategy 2: Random Truncation (Result B: list(cands)[:350])
    # Strategy 3: Deterministic Evidence-Based Scoring Top 350
    # Strategy 4: Deterministic Evidence-Based Scoring Top 500

    results = {}
    for strat in ["uncapped", "random_350", "evidence_350", "evidence_500"]:
        matched_gt = 0
        total_cands = 0

        for r in sample_s1.itertuples():
            s1_id = r.entity_id
            country = r.country_clean
            sig_toks = set(t for t in r.name_significant_tokens.split() if len(t) >= 2)
            addr_toks = set(t for t in r.address_tokens.split() if len(t) >= 3)
            addr_nums = set(n for n in r.address_numbers.split() if len(n) >= 1)
            norm_name = r.business_name_clean

            # Gather raw candidates from indexes
            cand_hits = defaultdict(float) # eid -> evidence score accumulator

            for t in sig_toks:
                if (country, t) in f_name:
                    for cid in f_name[(country, t)]:
                        cand_hits[cid] += 3.0 # name token match weight

            for t in addr_toks:
                if (country, t) in f_addr:
                    for cid in f_addr[(country, t)]:
                        cand_hits[cid] += 1.5 # addr token match weight

            for num in addr_nums:
                if (country, num) in f_num:
                    for cid in f_num[(country, num)]:
                        cand_hits[cid] += 2.0 # num match weight

            if len(cand_hits) < 5:
                for t in sig_toks:
                    if len(t) >= 4 and (country, t[:4]) in f_prefix:
                        for cid in f_prefix[(country, t[:4])]:
                            cand_hits[cid] += 1.0

            cands = set(cand_hits.keys())

            if strat == "uncapped":
                final_cands = cands
            elif strat == "random_350":
                final_cands = set(list(cands)[:350])
            elif strat == "evidence_350":
                # Rank candidates by score descending, then eid
                # Bonus for exact name match
                scored_cands = []
                for cid in cands:
                    sc = cand_hits[cid]
                    c_country, c_sig, c_addr, c_nums, c_name = s23_meta[cid]
                    if norm_name and norm_name == c_name:
                        sc += 10.0
                    scored_cands.append((sc, cid))
                scored_cands.sort(key=lambda x: (-x[0], x[1]))
                final_cands = set(x[1] for x in scored_cands[:350])
            elif strat == "evidence_500":
                scored_cands = []
                for cid in cands:
                    sc = cand_hits[cid]
                    c_country, c_sig, c_addr, c_nums, c_name = s23_meta[cid]
                    if norm_name and norm_name == c_name:
                        sc += 10.0
                    scored_cands.append((sc, cid))
                scored_cands.sort(key=lambda x: (-x[0], x[1]))
                final_cands = set(x[1] for x in scored_cands[:500])

            total_cands += len(final_cands)
            gt_targets = gt_map.get(s1_id)
            if gt_targets:
                hit = len(gt_targets.intersection(final_cands))
                matched_gt += hit

        rec = matched_gt / total_gt * 100 if total_gt > 0 else 0
        mean_c = total_cands / len(sample_s1)
        est_disk_gb = mean_c * 3.938e6 * 12 / 1e9
        results[strat] = (rec, mean_c, est_disk_gb)
        print(f"Strategy: {strat:<15} | Pair Recall: {rec:.2f}% | Mean Cands: {mean_c:.1f} | Est Total Disk: {est_disk_gb:.2f} GB")

test_ranking()
