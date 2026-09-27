#!/usr/bin/env python3
import time
import pandas as pd
from pathlib import Path
from collections import defaultdict

CLEANED_DIR = Path("/Users/swapnil/Documents/ML/manthan_output/phase_2_cleaning/cleaned")
GT_FILE = Path("/Users/swapnil/Documents/ML/student_resource/dataset/train/train_ground_truth.tsv")

print("Loading cleaned train files...")
t0 = time.time()
train_s1 = pd.read_csv(CLEANED_DIR / "clean_train_source1.tsv", sep='\t', dtype=str, keep_default_na=False)
train_s2 = pd.read_csv(CLEANED_DIR / "clean_train_source2.tsv", sep='\t', dtype=str, keep_default_na=False)
train_s3 = pd.read_csv(CLEANED_DIR / "clean_train_source3.tsv", sep='\t', dtype=str, keep_default_na=False)
print(f"Loaded train files in {time.time()-t0:.2f}s")

# Load GT
gt_df = pd.read_csv(GT_FILE, sep='\t', dtype=str, keep_default_na=False)
gt_map = {}
for r in gt_df.itertuples():
    m_str = r.matched_entity_ids
    matches = set(x.strip() for x in m_str.split(',') if x.strip()) if m_str else set()
    if matches:
        gt_map[r.source1_entity_id] = matches

def eval_config(max_name, max_addr, max_num, max_prefix, max_cands_per_s1=None):
    name_idx = defaultdict(set)
    addr_idx = defaultdict(set)
    num_idx = defaultdict(set)
    prefix_idx = defaultdict(set)

    for df in (train_s2, train_s3):
        for r in df.itertuples():
            eid = r.entity_id
            country = r.country_clean
            for t in r.name_significant_tokens.split():
                if len(t) >= 2:
                    name_idx[(country, t)].add(eid)
                    if len(t) >= 4:
                        prefix_idx[(country, t[:4])].add(eid)
            for t in r.address_tokens.split():
                if len(t) >= 3:
                    addr_idx[(country, t)].add(eid)
            for t in r.address_numbers.split():
                if len(t) >= 2:
                    num_idx[(country, t)].add(eid)

    f_name = {k: v for k, v in name_idx.items() if len(v) <= max_name}
    f_addr = {k: v for k, v in addr_idx.items() if len(v) <= max_addr}
    f_num = {k: v for k, v in num_idx.items() if len(v) <= max_num}
    f_prefix = {k: v for k, v in prefix_idx.items() if len(v) <= max_prefix}

    total_gt = 0
    matched_gt = 0
    total_cands = 0
    max_c = 0

    # Test on a 100,000 S1 sample
    sample_s1 = train_s1.sample(n=100000, random_state=42)

    for r in sample_s1.itertuples():
        s1_id = r.entity_id
        country = r.country_clean
        cands = set()

        for t in r.name_significant_tokens.split():
            if len(t) >= 2 and (country, t) in f_name:
                cands.update(f_name[(country, t)])

        for t in r.address_tokens.split():
            if len(t) >= 3 and (country, t) in f_addr:
                cands.update(f_addr[(country, t)])

        for t in r.address_numbers.split():
            if len(t) >= 2 and (country, t) in f_num:
                cands.update(f_num[(country, t)])

        if len(cands) < 5:
            for t in r.name_significant_tokens.split():
                if len(t) >= 4 and (country, t[:4]) in f_prefix:
                    cands.update(f_prefix[(country, t[:4])])

        if max_cands_per_s1 and len(cands) > max_cands_per_s1:
            cands = set(list(cands)[:max_cands_per_s1])

        nc = len(cands)
        total_cands += nc
        if nc > max_c:
            max_c = nc

        gt_targets = gt_map.get(s1_id)
        if gt_targets:
            total_gt += len(gt_targets)
            hit = len(gt_targets.intersection(cands))
            matched_gt += hit

    recall = matched_gt / total_gt * 100 if total_gt > 0 else 0
    mean_cands = total_cands / len(sample_s1)
    est_train_gb = (total_cands / len(sample_s1)) * 2.206e6 * 12 / 1e9
    est_test_gb = (total_cands / len(sample_s1)) * 1.732e6 * 12 / 1e9
    print(f"Postings ({max_name}, {max_addr}, {max_num}, {max_prefix}) | MaxPerS1: {str(max_cands_per_s1):>4} => Recall: {recall:.2f}% | Mean Cands: {mean_cands:.1f} | Max Cands: {max_c} | Est Total Disk: {(est_train_gb + est_test_gb):.2f} GB")

eval_config(3000, 1000, 1000, 1000, max_cands_per_s1=None)
eval_config(2000, 800, 800, 800, max_cands_per_s1=800)
eval_config(1500, 500, 500, 500, max_cands_per_s1=500)
eval_config(1000, 500, 500, 500, max_cands_per_s1=400)
