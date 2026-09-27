#!/usr/bin/env python3
import sys
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

# Load GT
gt_df = pd.read_csv(GT_FILE, sep='\t', dtype=str, keep_default_na=False)
gt_map = {}
for r in gt_df.itertuples():
    m_str = r.matched_entity_ids
    matches = set(x.strip() for x in m_str.split(',') if x.strip()) if m_str else set()
    gt_map[r.source1_entity_id] = matches

def test_fast():
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

    f_name = {k: v for k, v in name_idx.items() if len(v) <= 1000}
    f_addr = {k: v for k, v in addr_idx.items() if len(v) <= 500}
    f_num = {k: v for k, v in num_idx.items() if len(v) <= 500}
    f_prefix = {k: v for k, v in prefix_idx.items() if len(v) <= 500}

    # Evaluate on 20k S1 sample
    sample_s1 = train_s1.sample(n=20000, random_state=42)

    for cap in [None, 200, 100, 50, 30]:
        total_gt = 0
        matched_gt = 0
        total_cands = 0
        max_c = 0
        counts = []

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

            if cap and len(cands) > cap:
                cands = set(list(cands)[:cap])

            nc = len(cands)
            counts.append(nc)
            total_cands += nc
            if nc > max_c:
                max_c = nc

            gt_targets = gt_map.get(s1_id, set())
            if gt_targets:
                total_gt += len(gt_targets)
                hit = len(gt_targets.intersection(cands))
                matched_gt += hit

        recall = matched_gt / total_gt * 100 if total_gt > 0 else 0
        mean_cands = total_cands / len(sample_s1)
        est_disk_gb = (total_cands / len(sample_s1)) * 3.938e6 * 12 / 1e9
        print(f"Cap: {str(cap):>4} | Recall: {recall:.2f}% | Mean Cands: {mean_cands:.1f} | Max Cands: {max_c} | Est Disk: {est_disk_gb:.2f} GB")

test_fast()
