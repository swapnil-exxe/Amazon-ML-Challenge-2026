#!/usr/bin/env python3
"""
Phase 2 Error Ceiling Analysis — ML Challenge 2026 S++ Optimization
Categorizes 20K validation error loss into:
1. Blocking / Retrieval Failure (Candidate missing from top-350 candidates)
2. Ranking Failure (Candidate present in top-350 but ranked low / lost at top-K)
3. Threshold Failure (Candidate present and ranked high, but rejected by threshold)
"""

import os
import sys
import time
import math
import joblib
import psutil
import numpy as np
import pandas as pd

def token_jaccard(set1, set2):
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def prefix_4_match(str1, str2):
    if len(str1) >= 4 and len(str2) >= 4:
        return 1.0 if str1[:4] == str2[:4] else 0.0
    return 0.0

def load_feature_tuples(tsv_path):
    usecols = ['entity_id', 'business_name_clean', 'name_significant_tokens', 'business_address_clean', 'country_clean', 'address_numbers']
    df = pd.read_csv(tsv_path, sep='\t', usecols=usecols, dtype=str).fillna('')
    ids = df['entity_id'].to_numpy()
    names = df['business_name_clean'].to_numpy()
    sigs = df['name_significant_tokens'].to_numpy()
    addrs = df['business_address_clean'].to_numpy()
    countries = df['country_clean'].to_numpy()
    nums_arr = df['address_numbers'].to_numpy()

    feats = {}
    for e_id, name, sig, addr, country, nums in zip(ids, names, sigs, addrs, countries, nums_arr):
        feats[e_id] = (name, sig, addr, country, nums)
    del df
    return feats

def parse_entity_features_from_tuple(tup):
    name, sig, addr, country, nums = tup
    return {
        'name': name,
        'sig_name': sig,
        'addr': addr,
        'country': country,
        'nums': nums,
        'name_tokens': set(name.split()) if name else set(),
        'sig_tokens': set(sig.split()) if sig else set(),
        'addr_tokens': set(addr.split()) if addr else set(),
        'num_tokens': set(nums.split()) if nums else set(),
        'name_len': len(name),
        'addr_len': len(addr)
    }

def main():
    exp_dir = '/Users/swapnil/Documents/ML/spp_optimization/experiments'
    val_gt_path = os.path.join(exp_dir, 'val_ground_truth.tsv')
    clean_dir = '/Users/swapnil/Documents/ML/manthan_output/phase_2_cleaning/cleaned'
    cand_train_path = '/Users/swapnil/Documents/ML/manthan_output/phase_3_blocking/candidates/candidate_pairs_train_v2.tsv'
    model_path = '/Users/swapnil/Documents/ML/arya_output/arya_model.joblib'

    num_queries = 20000
    print(f"=== PHASE 2: ERROR CEILING ANALYSIS ({num_queries:,} QUERIES) ===")

    val_gt_df = pd.read_csv(val_gt_path, sep='\t', dtype=str, keep_default_na=False).iloc[:num_queries]
    gt_map = {}
    for r in val_gt_df.itertuples():
        s1_id = r.source1_entity_id
        m_str = r.matched_entity_ids
        matches = set(x.strip() for x in m_str.split(',') if x.strip()) if m_str else set()
        gt_map[s1_id] = matches

    total_gt_pairs = sum(len(v) for v in gt_map.values())

    s1_feats = load_feature_tuples(os.path.join(clean_dir, 'clean_train_source1.tsv'))
    s2_feats = load_feature_tuples(os.path.join(clean_dir, 'clean_train_source2.tsv'))
    s3_feats = load_feature_tuples(os.path.join(clean_dir, 'clean_train_source3.tsv'))
    s2_s3_feats = {**s2_feats, **s3_feats}
    del s2_feats, s3_feats

    clf = joblib.load(model_path)

    eval_cands = {}
    with open(cand_train_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if not parts or not parts[0]:
                continue
            s1_id = parts[0]
            if s1_id in gt_map:
                cands = parts[1].split(',') if len(parts) > 1 and parts[1].strip() else []
                eval_cands[s1_id] = cands
                if len(eval_cands) >= len(gt_map):
                    break

    # Categorization counts for GT true pairs
    blocking_misses = 0      # Not in candidates at all
    present_in_cands = 0     # In candidates
    
    # Ranks of present true matches after classifier prediction
    rank_1 = 0
    rank_5 = 0
    rank_10 = 0
    rank_25 = 0
    rank_50 = 0
    rank_100 = 0
    rank_350 = 0

    threshold_misses = 0     # Present & ranked, but prob < threshold
    threshold_hits = 0       # Selected by model threshold (TP)
    false_positives = 0      # Predicted by model threshold but not in GT (FP)

    sub_chunk_size = 1000
    s1_ids_list = list(gt_map.keys())
    num_sub = math.ceil(len(s1_ids_list) / sub_chunk_size)

    for sc_i in range(num_sub):
        sub_s1 = s1_ids_list[sc_i * sub_chunk_size : (sc_i + 1) * sub_chunk_size]
        rows, meta = [], []

        for s1_id in sub_s1:
            if s1_id not in s1_feats:
                continue
            f1 = parse_entity_features_from_tuple(s1_feats[s1_id])
            cands = eval_cands.get(s1_id, [])
            total_cands = len(cands)

            for rank, cand_id in enumerate(cands, start=1):
                if cand_id not in s2_s3_feats:
                    continue
                f2 = parse_entity_features_from_tuple(s2_s3_feats[cand_id])
                row = [
                    1.0 if f1['name'] and f1['name'] == f2['name'] else 0.0,
                    1.0 if f1['sig_name'] and f1['sig_name'] == f2['sig_name'] else 0.0,
                    token_jaccard(f1['name_tokens'], f2['name_tokens']),
                    token_jaccard(f1['sig_tokens'], f2['sig_tokens']),
                    abs(f1['name_len'] - f2['name_len']),
                    prefix_4_match(f1['name'], f2['name']),
                    1.0 if f1['addr'] and f1['addr'] == f2['addr'] else 0.0,
                    token_jaccard(f1['addr_tokens'], f2['addr_tokens']),
                    1.0 if not f2['addr'] else 0.0,
                    1.0 if f1['nums'] and f1['nums'] == f2['nums'] else 0.0,
                    token_jaccard(f1['num_tokens'], f2['num_tokens']),
                    abs(f1['addr_len'] - f2['addr_len']),
                    1.0 if f1['country'] == f2['country'] else 0.0,
                    1.0 if cand_id.startswith('S3-') else 0.0,
                    float(rank),
                    float(total_cands)
                ]
                rows.append(row)
                meta.append((s1_id, cand_id))

        if rows:
            probs = clf.predict_proba(np.array(rows, dtype=np.float32))[:, 1]
            s1_probs = {}
            for (s1_id, cand_id), prob in zip(meta, probs):
                s1_probs.setdefault(s1_id, []).append((cand_id, float(prob)))

            for s1_id in sub_s1:
                true_set = gt_map[s1_id]
                cands = set(eval_cands.get(s1_id, []))
                f1 = parse_entity_features_from_tuple(s1_feats[s1_id]) if s1_id in s1_feats else {'country': 'US'}
                country = f1['country']

                # 1. Blocking Analysis
                present = true_set & cands
                missing = true_set - cands
                blocking_misses += len(missing)
                present_in_cands += len(present)

                # 2. Ranking & Threshold Analysis
                if s1_id in s1_probs:
                    c_probs = s1_probs[s1_id]
                    c_probs.sort(key=lambda x: x[1], reverse=True)

                    for rank_i, (cid, p) in enumerate(c_probs, start=1):
                        if cid in true_set:
                            if rank_i <= 1: rank_1 += 1
                            if rank_i <= 5: rank_5 += 1
                            if rank_i <= 10: rank_10 += 1
                            if rank_i <= 25: rank_25 += 1
                            if rank_i <= 50: rank_50 += 1
                            if rank_i <= 100: rank_100 += 1
                            if rank_i <= 350: rank_350 += 1

                    if country == 'US':
                        selected = set(cid for cid, p in c_probs if p >= 0.60)
                    elif country == 'India':
                        selected = set(cid for cid, p in c_probs if p >= 0.50)
                    else:
                        selected = set(cid for cid, p in c_probs if p >= 0.35)
                        if not selected and c_probs[0][1] >= 0.30:
                            selected = {c_probs[0][0]}
                else:
                    selected = set()

                tp = len(selected & true_set)
                fp = len(selected - true_set)
                thresh_miss = len(present - selected)

                threshold_hits += tp
                false_positives += fp
                threshold_misses += thresh_miss

    print("\n==================================================")
    print("  PHASE 2: TRUE MATCH ERROR LOSS CEILING BREAKDOWN")
    print("==================================================")
    print(f"Total Ground-Truth True Pairs: {total_gt_pairs:,} (100.00%)")
    print(f"\n1. BLOCKING FAILURE (Missing from candidates):")
    print(f"   - Missed Pairs: {blocking_misses:,} ({blocking_misses/total_gt_pairs*100:.2f}%)")
    print(f"   - Reachable Pairs: {present_in_cands:,} ({present_in_cands/total_gt_pairs*100:.2f}%)")

    print(f"\n2. RANKING RECALL CUMULATIVE CEILING (Among present candidates):")
    print(f"   - Recall@1:   {rank_1:,} ({rank_1/total_gt_pairs*100:.2f}%)")
    print(f"   - Recall@5:   {rank_5:,} ({rank_5/total_gt_pairs*100:.2f}%)")
    print(f"   - Recall@10:  {rank_10:,} ({rank_10/total_gt_pairs*100:.2f}%)")
    print(f"   - Recall@25:  {rank_25:,} ({rank_25/total_gt_pairs*100:.2f}%)")
    print(f"   - Recall@50:  {rank_50:,} ({rank_50/total_gt_pairs*100:.2f}%)")
    print(f"   - Recall@100: {rank_100:,} ({rank_100/total_gt_pairs*100:.2f}%)")
    print(f"   - Recall@350: {rank_350:,} ({rank_350/total_gt_pairs*100:.2f}%)")

    print(f"\n3. THRESHOLD DECISION BREAKDOWN:")
    print(f"   - True Positives Selected (Hits): {threshold_hits:,} ({threshold_hits/total_gt_pairs*100:.2f}%)")
    print(f"   - Threshold Rejected Matches:     {threshold_misses:,} ({threshold_misses/total_gt_pairs*100:.2f}%)")
    print(f"   - False Positive Extractions:     {false_positives:,}")
    print("==================================================")

    out_csv = os.path.join(exp_dir, 'baseline_streaming', 'phase2_error_ceiling.csv')
    df_err = pd.DataFrame([{
        'total_gt_pairs': total_gt_pairs,
        'blocking_misses': blocking_misses,
        'reachable_pairs': present_in_cands,
        'rank_1': rank_1,
        'rank_5': rank_5,
        'rank_10': rank_10,
        'rank_25': rank_25,
        'rank_50': rank_50,
        'rank_100': rank_100,
        'rank_350': rank_350,
        'threshold_hits': threshold_hits,
        'threshold_misses': threshold_misses,
        'false_positives': false_positives
    }])
    df_err.to_csv(out_csv, index=False)
    print(f"Saved error ceiling breakdown to: {out_csv}")

if __name__ == '__main__':
    main()
