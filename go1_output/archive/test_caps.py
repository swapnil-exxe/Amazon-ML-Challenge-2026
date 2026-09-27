#!/usr/bin/env python3
import sys
import time
import pandas as pd
from pathlib import Path
from collections import defaultdict

CLEANED_DIR = Path("/Users/swapnil/Documents/ML/go1_output/phase_2_cleaning/cleaned")
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
    gt_map[r.source1_entity_id] = matches

def test_config(max_name, max_addr, max_num, max_prefix, max_cands_per_s1=None):
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

    # Test on a 100k sample
    sample_s1 = train_s1.sample(n=100000, random_state=42)
    sample_eids = set(sample_s1['entity_id'])

    total_cands = 0
    max_c_seen = 0

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
        if nc > max_c_seen:
            max_c_seen = nc

        gt_targets = gt_map.get(s1_id, set())
        if gt_targets:
            total_gt += len(gt_targets)
            hit = len(gt_targets.intersection(cands))
            matched_gt += hit

    recall = matched_gt / total_gt * 100 if total_gt > 0 else 0
    mean_cands = total_cands / len(sample_s1)
    est_file_size_gb = (total_cands / len(sample_s1)) * 3.938e6 * 12 / 1e9
    print(f"Caps ({max_name}, {max_addr}, {max_num}, {max_prefix}) | MaxPerS1: {max_cands_per_s1} => Recall: {recall:.2f}% | Mean Cands: {mean_cands:.1f} | Max Cands: {max_c_seen} | Est Disk: {est_file_size_gb:.2f} GB")

test_config(3000, 1000, 1000, 1000)
test_config(500, 200, 200, 200)
test_config(200, 100, 100, 100)
test_config(100, 50, 50, 50)
test_config(3000, 1000, 1000, 1000, max_cands_per_s1=200)
test_config(3000, 1000, 1000, 1000, max_cands_per_s1=100)
test_config(3000, 1000, 1000, 1000, max_cands_per_s1=50)
