#!/usr/bin/env python3
"""
1-HOUR EMERGENCY HIGH-SPEED S++ OPTIMIZER — ML Challenge 2026
Pre-indexes candidate inverted lookups, runs pre-ranked inference on 20K validation set,
and performs high-speed multi-country threshold grid calibration in seconds.
Strictly verifies SHA256 hashes before and after execution to protect production artifacts.
"""

import os
import sys
import time
import math
import json
import hashlib
import gc
import joblib
import psutil
import numpy as np
import pandas as pd
from collections import defaultdict

# Expected SHA256 Hashes for Protected Production Artifacts
EXPECTED_HASHES = {
    '/Users/swapnil/Documents/ML/team_submission.zip': '459e67da33e82ea6b052de382cbdf0805c6cbb09a5ca2138415fa6a7cd043747',
    '/Users/swapnil/Documents/ML/team_submission_v2.zip': 'd6926b559404028a9fdd2aca167438f0cead828fb4991ede0474ee7f8ad46d7c',
    '/Users/swapnil/Documents/ML/arya_output/matching_results.tsv': 'd1c898796e2ab192b627f8d0e8ea83f96e1427b029a8d66ed9165c87bd94d1e8',
    '/Users/swapnil/Documents/ML/arya_output/matching_results_v2.tsv': '56d88eb1f3b7f1e6dd23b3b626266b9cfb9b59a9257f9e9feedc6a6fc7a72d73',
    '/Users/swapnil/Documents/ML/arya_output/arya_model.joblib': '476763ebe4051718167703dbf28e88a85ef80e2982876ccdb028508793534cad'
}

EXP_DIR = '/Users/swapnil/Documents/ML/spp_optimization/experiments'
VAL_GT_PATH = os.path.join(EXP_DIR, 'val_ground_truth.tsv')
CLEAN_DIR = '/Users/swapnil/Documents/ML/manthan_output/phase_2_cleaning/cleaned'
MODEL_PATH = '/Users/swapnil/Documents/ML/arya_output/arya_model.joblib'

LOG_PATH = os.path.join(EXP_DIR, 'emergency_optimization.log')

def log(msg, file_obj):
    print(msg)
    file_obj.write(msg + '\n')
    file_obj.flush()

def verify_protected_artifacts(file_obj):
    log("=== VERIFYING PROTECTED PRODUCTION ARTIFACT INTEGRITY ===", file_obj)
    all_pass = True
    for path, exp in EXPECTED_HASHES.items():
        if not os.path.exists(path):
            log(f"FAIL: {path} missing!", file_obj)
            all_pass = False
            continue
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            while chunk := f.read(8192 * 1024):
                h.update(chunk)
        digest = h.hexdigest()
        status = "PASS" if digest == exp else "FAIL"
        if status == "FAIL":
            all_pass = False
        log(f"  - {os.path.basename(path):<25} | SHA256: {digest[:16]}... | {status}", file_obj)
    if not all_pass:
        log("CRITICAL ERROR: SHA256 Verification Failed! Stopping immediately.", file_obj)
        sys.exit(1)
    log("ALL PROTECTED ARTIFACTS 100% INTACT (PASS).\n", file_obj)

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

def parse_entity_features(tup):
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
    t0 = time.time()
    log_file = open(LOG_PATH, 'w', encoding='utf-8')
    log("==================================================", log_file)
    log("  1-HOUR EMERGENCY HIGH-SPEED S++ OPTIMIZER", log_file)
    log("==================================================", log_file)

    verify_protected_artifacts(log_file)

    num_queries = 20000
    log(f"Loading 20,000 validation queries from {VAL_GT_PATH}...", log_file)
    val_gt_df = pd.read_csv(VAL_GT_PATH, sep='\t', dtype=str, keep_default_na=False).iloc[:num_queries]
    gt_map = {}
    s1_country_map = {}
    for r in val_gt_df.itertuples():
        s1_id = r.source1_entity_id
        m_str = r.matched_entity_ids
        matches = set(x.strip() for x in m_str.split(',') if x.strip()) if m_str else set()
        gt_map[s1_id] = matches

    total_gt_pairs = sum(len(v) for v in gt_map.values())
    log(f"Loaded {len(gt_map):,} validation queries with {total_gt_pairs:,} true match pairs.", log_file)

    log("Loading cleaned entity feature tuples...", log_file)
    usecols = ['entity_id', 'business_name_clean', 'name_significant_tokens', 'business_address_clean', 'country_clean', 'address_numbers']
    s1_df = pd.read_csv(os.path.join(CLEAN_DIR, 'clean_train_source1.tsv'), sep='\t', usecols=usecols, dtype=str).fillna('')
    s2_df = pd.read_csv(os.path.join(CLEAN_DIR, 'clean_train_source2.tsv'), sep='\t', usecols=usecols, dtype=str).fillna('')
    s3_df = pd.read_csv(os.path.join(CLEAN_DIR, 'clean_train_source3.tsv'), sep='\t', usecols=usecols, dtype=str).fillna('')

    s1_feats = {}
    for r in s1_df.itertuples():
        s1_feats[r.entity_id] = (r.business_name_clean, r.name_significant_tokens, r.business_address_clean, r.country_clean, r.address_numbers)
        s1_country_map[r.entity_id] = r.country_clean

    s23_meta = {}
    for df in (s2_df, s3_df):
        for r in df.itertuples():
            eid = r.entity_id
            country = r.country_clean
            sig_toks = set(t for t in r.name_significant_tokens.split() if len(t) >= 2)
            addr_toks = set(t for t in r.business_address_clean.split() if len(t) >= 3)
            addr_nums = set(n for n in r.address_numbers.split() if len(n) >= 1)
            norm_name = r.business_name_clean
            s23_meta[eid] = (country, sig_toks, addr_toks, addr_nums, norm_name, (r.business_name_clean, r.name_significant_tokens, r.business_address_clean, r.country_clean, r.address_numbers))

    del s2_df, s3_df
    gc.collect()

    log("Building high-speed inverted indexes...", log_file)
    name_idx = defaultdict(set)
    addr_idx = defaultdict(set)
    num_idx = defaultdict(set)
    prefix_idx = defaultdict(set)

    for eid, (country, sig_toks, addr_toks, addr_nums, norm_name, _) in s23_meta.items():
        for t in sig_toks:
            if len(t) >= 2:
                name_idx[(country, t)].add(eid)
                if len(t) >= 4:
                    prefix_idx[(country, t[:4])].add(eid)
        for t in addr_toks:
            if len(t) >= 3:
                addr_idx[(country, t)].add(eid)
        for n in addr_nums:
            if len(n) >= 1:
                num_idx[(country, n)].add(eid)

    f_name = {k: v for k, v in name_idx.items() if len(v) <= 3000}
    f_addr = {k: v for k, v in addr_idx.items() if len(v) <= 1500}
    f_num = {k: v for k, v in num_idx.items() if len(v) <= 1500}
    f_prefix = {k: v for k, v in prefix_idx.items() if len(v) <= 1500}

    log("Generating pre-ranked candidate sets for 20,000 validation queries...", log_file)
    eval_cands = {}
    hit_cand_pairs = 0
    tot_cands_cnt = 0

    for s1_id in gt_map.keys():
        if s1_id not in s1_feats:
            eval_cands[s1_id] = []
            continue
        f1 = parse_entity_features(s1_feats[s1_id])
        country = f1['country']
        norm_name = f1['name']

        cand_hits = defaultdict(float)
        for t in f1['sig_tokens']:
            if (country, t) in f_name:
                for cid in f_name[(country, t)]:
                    cand_hits[cid] += 3.0
        for t in f1['addr_tokens']:
            if (country, t) in f_addr:
                for cid in f_addr[(country, t)]:
                    cand_hits[cid] += 1.5
        for num in f1['num_tokens']:
            if (country, num) in f_num:
                for cid in f_num[(country, num)]:
                    cand_hits[cid] += 2.0
        if len(cand_hits) < 5:
            for t in f1['sig_tokens']:
                if len(t) >= 4 and (country, t[:4]) in f_prefix:
                    for cid in f_prefix[(country, t[:4])]:
                        cand_hits[cid] += 1.0

        scored_cands = []
        for cid in cand_hits.keys():
            sc = cand_hits[cid]
            c_country, c_sig, c_addr, c_nums, c_name, _ = s23_meta[cid]
            if norm_name and norm_name == c_name:
                sc += 10.0
            scored_cands.append((sc, cid))

        scored_cands.sort(key=lambda x: (-x[0], x[1]))
        top_cands = [x[1] for x in scored_cands[:350]]
        eval_cands[s1_id] = top_cands

        true_set = gt_map[s1_id]
        tot_cands_cnt += len(top_cands)
        hit_cand_pairs += len(true_set.intersection(top_cands))

    cand_blocking_recall = hit_cand_pairs / total_gt_pairs
    log(f"Pre-ranked Candidate Blocking Recall: {cand_blocking_recall*100:.2f}% (Hit Pairs: {hit_cand_pairs:,} / {total_gt_pairs:,})", log_file)
    log(f"Average Candidates / Query: {tot_cands_cnt / len(gt_map):.1f}", log_file)

    log(f"Loading trained classifier model from {MODEL_PATH}...", log_file)
    clf = joblib.load(MODEL_PATH)

    log("Extracting features and predicting candidate probabilities in batch...", log_file)
    rows, meta = [], []
    for s1_id in gt_map.keys():
        if s1_id not in s1_feats:
            continue
        f1 = parse_entity_features(s1_feats[s1_id])
        cands = eval_cands.get(s1_id, [])
        total_cands_len = len(cands)

        for rank, cand_id in enumerate(cands, start=1):
            if cand_id not in s23_meta:
                continue
            f2_tup = s23_meta[cand_id][5]
            f2 = parse_entity_features(f2_tup)
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
                float(total_cands_len)
            ]
            rows.append(row)
            meta.append((s1_id, cand_id))

    X_mat = np.array(rows, dtype=np.float32)
    probs = clf.predict_proba(X_mat)[:, 1]

    s1_probs = {}
    for (s1_id, cand_id), p in zip(meta, probs):
        s1_probs.setdefault(s1_id, []).append((cand_id, float(p)))

    # Compute Recall@1, Recall@5, MRR
    r_at_1, r_at_5, mrr_sum = 0, 0, 0.0
    for s1_id in gt_map.keys():
        true_set = gt_map[s1_id]
        if s1_id in s1_probs:
            c_p = sorted(s1_probs[s1_id], key=lambda x: x[1], reverse=True)
            for rank_i, (cid, p) in enumerate(c_p, start=1):
                if cid in true_set:
                    if rank_i == 1: r_at_1 += 1
                    if rank_i <= 5: r_at_5 += 1
                    mrr_sum += 1.0 / rank_i
                    break

    rec_1 = r_at_1 / len(gt_map)
    rec_5 = r_at_5 / len(gt_map)
    mrr_val = mrr_sum / len(gt_map)

    log(f"Recall@1: {rec_1*100:.2f}% | Recall@5: {rec_5*100:.2f}% | MRR: {mrr_val:.4f}", log_file)

    log("\n--- EXECUTING FAST MULTI-COUNTRY THRESHOLD GRID CALIBRATION ---", log_file)
    us_grid = [0.40, 0.45, 0.50, 0.55, 0.60]
    india_grid = [0.30, 0.35, 0.40, 0.45, 0.50]
    other_grid = [0.20, 0.25, 0.30, 0.35, 0.40]

    best_score = 0.0
    best_tuple = None

    for t_us in us_grid:
        for t_in in india_grid:
            for t_ot in other_grid:
                tp_tot, fp_tot = 0, 0
                for s1_id in gt_map.keys():
                    true_set = gt_map[s1_id]
                    country = s1_country_map.get(s1_id, 'US')
                    if s1_id in s1_probs:
                        c_p = s1_probs[s1_id]
                        if country == 'US':
                            sel = set(cid for cid, p in c_p if p >= t_us)
                        elif country == 'India':
                            sel = set(cid for cid, p in c_p if p >= t_in)
                        else:
                            sel = set(cid for cid, p in c_p if p >= t_ot)
                            if not sel and sorted(c_p, key=lambda x: x[1], reverse=True)[0][1] >= (t_ot - 0.05):
                                sel = {sorted(c_p, key=lambda x: x[1], reverse=True)[0][0]}
                    else:
                        sel = set()

                    tp = len(sel & true_set)
                    fp = len(sel - true_set)
                    tp_tot += tp
                    fp_tot += fp

                prec = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) > 0 else 0.0
                rec = tp_tot / total_gt_pairs if total_gt_pairs > 0 else 0.0
                f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (prec + rec) > 0 else 0.0

                if f05 > best_score:
                    best_score = f05
                    best_tuple = (t_us, t_in, t_ot, prec, rec, f05, tp_tot, fp_tot)

    t_us, t_in, t_ot, prec_best, rec_best, f05_best, tp_best, fp_best = best_tuple
    runtime_total = time.time() - t0
    peak_ram_mb = psutil.Process().memory_info().rss / (1024 * 1024)

    log(f"\n==================================================", log_file)
    log(f"  EMERGENCY OPTIMIZATION BEST RESULT ACHIEVED", log_file)
    log(f"==================================================", log_file)
    log(f"Optimal Thresholds:        US (p >= {t_us:.2f}) | India (p >= {t_in:.2f}) | Other (p >= {t_ot:.2f})", log_file)
    log(f"Candidate Blocking Recall: {cand_blocking_recall*100:.2f}%", log_file)
    log(f"End-to-End Precision:      {prec_best*100:.2f}%", log_file)
    log(f"End-to-End Recall:         {rec_best*100:.2f}%", log_file)
    log(f"End-to-End Macro F0.5:     {f05_best:.6f} (Baseline: 0.864000 | Delta: {f05_best - 0.8640:+.6f})", log_file)
    log(f"Recall@1:                  {rec_1*100:.2f}%", log_file)
    log(f"Recall@5:                  {rec_5*100:.2f}%", log_file)
    log(f"MRR:                       {mrr_val:.4f}", log_file)
    log(f"Average Candidates/Query:  {tot_cands_cnt / len(gt_map):.1f}", log_file)
    log(f"Total Execution Time:      {runtime_total:.2f} seconds", log_file)
    log(f"Peak RAM Usage:            {peak_ram_mb:.1f} MB", log_file)

    pass_primary = "PASS" if f05_best >= 0.992000 else "FAIL"
    pass_stretch = "PASS" if f05_best >= 0.992500 else "FAIL"
    pass_aggressive = "PASS" if f05_best >= 0.993000 else "FAIL"

    log(f"\nTarget Status:", log_file)
    log(f"  PRIMARY    (>= 0.992000): {pass_primary}", log_file)
    log(f"  STRETCH    (>= 0.992500): {pass_stretch}", log_file)
    log(f"  AGGRESSIVE (>= 0.993000): {pass_aggressive}", log_file)

    # Save artifacts
    best_config_data = {
        'experiment_id': 'EMERGENCY_C1_THRESHOLD_OPT',
        'optimal_thresholds': {'US': t_us, 'India': t_in, 'France_Other': t_ot},
        'metrics': {
            'candidate_blocking_recall': cand_blocking_recall,
            'precision': prec_best,
            'recall': rec_best,
            'macro_f05': f05_best,
            'recall_at_1': rec_1,
            'recall_at_5': rec_5,
            'mrr': mrr_val,
            'avg_candidates_per_query': tot_cands_cnt / len(gt_map),
            'runtime_sec': runtime_total,
            'peak_ram_mb': peak_ram_mb
        },
        'target_status': {
            'primary_0_992000': pass_primary,
            'stretch_0_992500': pass_stretch,
            'aggressive_0_993000': pass_aggressive
        }
    }

    with open(os.path.join(EXP_DIR, 'best_config.json'), 'w', encoding='utf-8') as f:
        json.dump(best_config_data, f, indent=2)

    df_final = pd.DataFrame([best_config_data['metrics']])
    df_final.to_csv(os.path.join(EXP_DIR, 'final_metrics.csv'), index=False)

    verify_protected_artifacts(log_file)
    log_file.close()

    print(f"\n========================================")
    print(f"ML CHALLENGE 2026 — EMERGENCY FINAL")
    print(f"========================================")
    print(f"Baseline Macro F0.5 : 0.864000")
    print(f"Best Macro F0.5     : {f05_best:.6f}")
    print(f"Primary  >= 0.992000 : {pass_primary}")
    print(f"Stretch  >= 0.992500 : {pass_stretch}")
    print(f"Aggressive >= 0.993000 : {pass_aggressive}")
    print(f"Best Experiment     : EMERGENCY_C1_THRESHOLD_OPT")
    print(f"Candidate Blocking Recall: {cand_blocking_recall*100:.2f}%")
    print(f"Precision           : {prec_best*100:.2f}%")
    print(f"Recall              : {rec_best*100:.2f}%")
    print(f"Recall@1            : {rec_1*100:.2f}%")
    print(f"Recall@5            : {rec_5*100:.2f}%")
    print(f"MRR                 : {mrr_val:.4f}")
    print(f"Runtime             : {runtime_total:.2f}s")
    print(f"Peak RAM            : {peak_rss_or_mb(peak_ram_mb)}")
    print(f"Final Output        : {os.path.join(EXP_DIR, 'final_metrics.csv')}")
    print(f"Protected SHA256    : PASS")
    print(f"========================================")

def peak_rss_or_mb(val):
    return f"{val:.1f} MB"

if __name__ == '__main__':
    main()
