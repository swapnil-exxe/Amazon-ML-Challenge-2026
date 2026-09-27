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

def run_bench():
    single_name_idx = defaultdict(set)
    comp_name_idx = defaultdict(set)
    name_num_idx = defaultdict(set)
    addr_num_idx = defaultdict(set)
    prefix_idx = defaultdict(set)

    print("Building composite indexes...")
    tb0 = time.time()
    for df in (train_s2, train_s3):
        for r in df.itertuples():
            eid = r.entity_id
            country = r.country_clean
            sig_toks = [t for t in r.name_significant_tokens.split() if len(t) >= 2]
            addr_toks = [t for t in r.address_tokens.split() if len(t) >= 3]
            addr_nums = [n for n in r.address_numbers.split() if len(n) >= 1]

            # 1. Single name keys
            for t in sig_toks:
                if len(t) >= 3:
                    single_name_idx[(country, t)].add(eid)
                if len(t) >= 4:
                    prefix_idx[(country, t[:4])].add(eid)

            # 2. Composite name keys (token pairs)
            for i in range(len(sig_toks)):
                for j in range(i+1, min(i+4, len(sig_toks))):
                    pair = tuple(sorted([sig_toks[i], sig_toks[j]]))
                    comp_name_idx[(country, pair[0], pair[1])].add(eid)

            # 3. Name + Number
            for t in sig_toks[:3]:
                for num in addr_nums[:2]:
                    name_num_idx[(country, t, num)].add(eid)

            # 4. Address + Number
            for at in addr_toks[:3]:
                for num in addr_nums[:2]:
                    addr_num_idx[(country, at, num)].add(eid)

    print(f"Index raw build complete in {time.time()-tb0:.2f}s")

    f_single = {k: v for k, v in single_name_idx.items() if len(v) <= 800}
    f_comp = {k: v for k, v in comp_name_idx.items() if len(v) <= 1500}
    f_name_num = {k: v for k, v in name_num_idx.items() if len(v) <= 1500}
    f_addr_num = {k: v for k, v in addr_num_idx.items() if len(v) <= 1000}
    f_prefix = {k: v for k, v in prefix_idx.items() if len(v) <= 300}

    total_gt = 0
    matched_gt = 0
    total_cands = 0
    max_c = 0

    sample_s1 = train_s1.sample(n=100000, random_state=42)

    for r in sample_s1.itertuples():
        s1_id = r.entity_id
        country = r.country_clean
        sig_toks = [t for t in r.name_significant_tokens.split() if len(t) >= 2]
        addr_toks = [t for t in r.address_tokens.split() if len(t) >= 3]
        addr_nums = [n for n in r.address_numbers.split() if len(n) >= 1]

        cands = set()

        # Composite name keys
        for i in range(len(sig_toks)):
            for j in range(i+1, min(i+4, len(sig_toks))):
                pair = tuple(sorted([sig_toks[i], sig_toks[j]]))
                key = (country, pair[0], pair[1])
                if key in f_comp:
                    cands.update(f_comp[key])

        # Name + Num keys
        for t in sig_toks[:3]:
            for num in addr_nums[:2]:
                key = (country, t, num)
                if key in f_name_num:
                    cands.update(f_name_num[key])

        # Single name keys
        for t in sig_toks:
            if len(t) >= 3:
                key = (country, t)
                if key in f_single:
                    cands.update(f_single[key])

        # Address + Num keys
        for at in addr_toks[:3]:
            for num in addr_nums[:2]:
                key = (country, at, num)
                if key in f_addr_num:
                    cands.update(f_addr_num[key])

        # Fallback if sparse
        if len(cands) < 5:
            for t in sig_toks:
                if len(t) >= 4:
                    key = (country, t[:4])
                    if key in f_prefix:
                        cands.update(f_prefix[key])

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
    print(f"Recall: {recall:.2f}% | Mean Cands: {mean_cands:.1f} | Max Cands: {max_c} | Est Train Disk: {est_train_gb:.2f} GB | Est Test Disk: {est_test_gb:.2f} GB | Total Disk: {(est_train_gb + est_test_gb):.2f} GB")

run_bench()
