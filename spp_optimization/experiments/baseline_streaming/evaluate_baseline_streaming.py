#!/usr/bin/env python3
"""
Memory-Safe Baseline Evaluator — ML Challenge 2026 S++ Optimization
Evaluates the baseline V2 model on fixed validation ground truth in streaming batches.
Tracks peak RAM, candidate recall, precision, recall, Macro F0.5, Recall@1, and MRR.
"""

import os
import sys
import time
import math
import hashlib
import gc
import joblib
import psutil
import numpy as np
import pandas as pd

# Expected SHA256 Hashes for Protected Production Artifacts
EXPECTED_HASHES = {
    '/Users/swapnil/Documents/ML/team_submission.zip': '459e67da33e82ea6b052de382cbdf0805c6cbb09a5ca2138415fa6a7cd043747',
    '/Users/swapnil/Documents/ML/team_submission_v2.zip': 'd6926b559404028a9fdd2aca167438f0cead828fb4991ede0474ee7f8ad46d7c',
    '/Users/swapnil/Documents/ML/arya_output/matching_results.tsv': 'd1c898796e2ab192b627f8d0e8ea83f96e1427b029a8d66ed9165c87bd94d1e8',
    '/Users/swapnil/Documents/ML/arya_output/matching_results_v2.tsv': '56d88eb1f3b7f1e6dd23b3b626266b9cfb9b59a9257f9e9feedc6a6fc7a72d73',
    '/Users/swapnil/Documents/ML/arya_output/arya_model.joblib': '476763ebe4051718167703dbf28e88a85ef80e2982876ccdb028508793534cad'
}

def verify_protected_artifacts():
    print("=== STEP 1: VERIFYING PROTECTED ARTIFACTS ===")
    all_pass = True
    for path, exp in EXPECTED_HASHES.items():
        if not os.path.exists(path):
            print(f"FAIL: {path} does not exist!")
            all_pass = False
            continue
        size_mb = os.path.getsize(path) / (1024 * 1024)
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            while chunk := f.read(8192 * 1024):
                h.update(chunk)
        digest = h.hexdigest()
        status = "PASS" if digest == exp else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  - {os.path.basename(path):<25} | {size_mb:>7.2f} MB | SHA256: {digest[:16]}... | {status}")
    if not all_pass:
        print("CRITICAL: One or more protected artifacts failed SHA256 verification! Stopping.")
        sys.exit(1)
    print("ALL PROTECTED ARTIFACTS INTACT (PASS).\n")

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
    gc.collect()
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

def run_evaluation(num_queries, output_csv):
    verify_protected_artifacts()

    t0 = time.time()
    exp_dir = '/Users/swapnil/Documents/ML/spp_optimization/experiments'
    val_gt_path = os.path.join(exp_dir, 'val_ground_truth.tsv')
    clean_dir = '/Users/swapnil/Documents/ML/manthan_output/phase_2_cleaning/cleaned'
    cand_train_path = '/Users/swapnil/Documents/ML/manthan_output/phase_3_blocking/candidates/candidate_pairs_train_v2.tsv'
    model_path = '/Users/swapnil/Documents/ML/arya_output/arya_model.joblib'

    print(f"=== RUNNING MEMORY-SAFE BASELINE EVALUATION ({num_queries:,} QUERIES) ===")

    val_gt_df = pd.read_csv(val_gt_path, sep='\t', dtype=str, keep_default_na=False)
    if num_queries < len(val_gt_df):
        eval_gt_df = val_gt_df.iloc[:num_queries]
    else:
        eval_gt_df = val_gt_df

    gt_map = {}
    for r in eval_gt_df.itertuples():
        s1_id = r.source1_entity_id
        m_str = r.matched_entity_ids
        matches = set(x.strip() for x in m_str.split(',') if x.strip()) if m_str else set()
        gt_map[s1_id] = matches

    total_gt_pairs = sum(len(v) for v in gt_map.values())
    print(f"Loaded {len(gt_map):,} queries with {total_gt_pairs:,} total true match pairs.")

    print("Loading feature tuples...")
    s1_feats = load_feature_tuples(os.path.join(clean_dir, 'clean_train_source1.tsv'))
    s2_feats = load_feature_tuples(os.path.join(clean_dir, 'clean_train_source2.tsv'))
    s3_feats = load_feature_tuples(os.path.join(clean_dir, 'clean_train_source3.tsv'))
    s2_s3_feats = {**s2_feats, **s3_feats}
    del s2_feats, s3_feats
    gc.collect()

    print("Loading model...")
    clf = joblib.load(model_path)

    print("Streaming candidates from disk...")
    eval_cands = {}
    with open(cand_train_path, 'r', encoding='utf-8') as f:
        next(f) # skip header
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

    s1_ids_list = list(gt_map.keys())
    sub_chunk_size = 1000
    num_sub = math.ceil(len(s1_ids_list) / sub_chunk_size)

    tp_tot, fp_tot = 0, 0
    hit_cand_pairs = 0
    r_at_1_hits, mrr_sum = 0, 0.0
    tot_cands_cnt = 0
    total_preds_cnt = 0

    peak_rss = 0

    for sc_i in range(num_sub):
        sub_s1 = s1_ids_list[sc_i * sub_chunk_size : (sc_i + 1) * sub_chunk_size]
        rows, meta = [], []

        for s1_id in sub_s1:
            if s1_id not in s1_feats:
                continue
            f1 = parse_entity_features_from_tuple(s1_feats[s1_id])
            cands = eval_cands.get(s1_id, [])
            true_set = gt_map[s1_id]
            tot_cands_cnt += len(cands)
            hit_cand_pairs += len(true_set.intersection(cands))

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
                f1 = parse_entity_features_from_tuple(s1_feats[s1_id]) if s1_id in s1_feats else {'country': 'US'}
                country = f1['country']

                if s1_id in s1_probs:
                    c_probs = s1_probs[s1_id]
                    c_probs.sort(key=lambda x: x[1], reverse=True)

                    if c_probs[0][0] in true_set:
                        r_at_1_hits += 1

                    for rank_i, (cid, p) in enumerate(c_probs, start=1):
                        if cid in true_set:
                            mrr_sum += 1.0 / rank_i
                            break

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
                tp_tot += tp
                fp_tot += fp
                total_preds_cnt += len(selected)

        cur_rss = psutil.Process().memory_info().rss / (1024 * 1024)
        if cur_rss > peak_rss:
            peak_rss = cur_rss
        del rows, meta
        gc.collect()

    prec = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) > 0 else 0.0
    rec = tp_tot / total_gt_pairs if total_gt_pairs > 0 else 0.0
    f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (prec + rec) > 0 else 0.0
    cand_rec = hit_cand_pairs / total_gt_pairs if total_gt_pairs > 0 else 0.0
    r_at_1 = r_at_1_hits / len(gt_map) if len(gt_map) > 0 else 0.0
    mrr = mrr_sum / len(gt_map) if len(gt_map) > 0 else 0.0
    runtime_sec = time.time() - t0

    print("\n==================================================")
    print(f"  BASELINE RESULTS ({num_queries:,} QUERIES)")
    print("==================================================")
    print(f"Queries Evaluated:         {len(gt_map):,}")
    print(f"Total True Match Pairs:    {total_gt_pairs:,}")
    print(f"Candidate Blocking Recall: {cand_rec * 100:.2f}%")
    print(f"End-to-End Precision:      {prec * 100:.2f}%")
    print(f"End-to-End Recall:         {rec * 100:.2f}%")
    print(f"End-to-End Macro F0.5:     {f05:.4f}")
    print(f"Recall@1:                  {r_at_1 * 100:.2f}%")
    print(f"MRR:                       {mrr:.4f}")
    print(f"Total Predicted Pairs:     {total_preds_cnt:,}")
    print(f"Average Candidates/Query:  {tot_cands_cnt / len(gt_map):.1f}")
    print(f"Execution Time:            {runtime_sec:.2f} seconds")
    print(f"Peak RAM Usage:            {peak_rss:.1f} MB")
    print("==================================================\n")

    res_df = pd.DataFrame([{
        'num_queries': len(gt_map),
        'true_match_pairs': total_gt_pairs,
        'candidate_blocking_recall': cand_rec,
        'precision': prec,
        'recall': rec,
        'macro_f05': f05,
        'recall_at_1': r_at_1,
        'mrr': mrr,
        'predicted_pairs': total_preds_cnt,
        'avg_candidates_per_query': tot_cands_cnt / len(gt_map),
        'runtime_sec': runtime_sec,
        'peak_ram_mb': peak_rss
    }])
    
    res_df.to_csv(output_csv, index=False)
    print(f"Saved evaluation result to: {output_csv}")

    verify_protected_artifacts()

if __name__ == '__main__':
    num_q = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    out_csv = sys.argv[2] if len(sys.argv) > 2 else f"baseline_{num_q // 1000}k.csv"
    run_evaluation(num_q, out_csv)
