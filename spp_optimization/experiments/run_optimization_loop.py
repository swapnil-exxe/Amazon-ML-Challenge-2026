#!/usr/bin/env python3
"""
S++ Optimization Loop Engine (Resume-Enabled) — ML Challenge 2026
Systematically runs isolated, memory-safe experiments toward Macro F0.5 >= 0.992000.
Strictly protects production artifacts and enforces SHA256 integrity checks.
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

# Protected Artifact Reference Hashes
EXPECTED_HASHES = {
    '/Users/swapnil/Documents/ML/team_submission.zip': '459e67da33e82ea6b052de382cbdf0805c6cbb09a5ca2138415fa6a7cd043747',
    '/Users/swapnil/Documents/ML/team_submission_v2.zip': 'd6926b559404028a9fdd2aca167438f0cead828fb4991ede0474ee7f8ad46d7c',
    '/Users/swapnil/Documents/ML/go2_output/matching_results.tsv': 'd1c898796e2ab192b627f8d0e8ea83f96e1427b029a8d66ed9165c87bd94d1e8',
    '/Users/swapnil/Documents/ML/go2_output/matching_results_v2.tsv': '56d88eb1f3b7f1e6dd23b3b626266b9cfb9b59a9257f9e9feedc6a6fc7a72d73',
    '/Users/swapnil/Documents/ML/go2_output/go2_model.joblib': '476763ebe4051718167703dbf28e88a85ef80e2982876ccdb028508793534cad'
}

EXP_BASE_DIR = '/Users/swapnil/Documents/ML/spp_optimization/experiments'
VAL_GT_PATH = os.path.join(EXP_BASE_DIR, 'val_ground_truth.tsv')
CLEAN_DIR = '/Users/swapnil/Documents/ML/go1_output/phase_2_cleaning/cleaned'
CAND_TRAIN_PATH = '/Users/swapnil/Documents/ML/go1_output/phase_3_blocking/candidates/candidate_pairs_train_v2.tsv'
MODEL_PATH = '/Users/swapnil/Documents/ML/go2_output/go2_model.joblib'

TARGET_PRIMARY = 0.992000
TARGET_STRETCH = 0.992500
TARGET_AGGRESSIVE = 0.993000

def verify_protected_artifacts(log_func=print):
    log_func("=== PROTECTED ARTIFACT SHA256 INTEGRITY CHECK ===")
    all_pass = True
    for path, exp in EXPECTED_HASHES.items():
        if not os.path.exists(path):
            log_func(f"FAIL: {path} missing!")
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
        log_func(f"  - {os.path.basename(path):<25} | SHA256: {digest[:16]}... | {status}")
    if not all_pass:
        log_func("CRITICAL: SHA256 Integrity Verification Failed! Stopping execution.")
        sys.exit(1)
    log_func("ALL PROTECTED ARTIFACTS INTACT (PASS).\n")

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

def run_experiment(exp_id, hypothesis, config_params, gt_map, s1_feats, s23_meta, name_idx, addr_idx, num_idx, prefix_idx, clf, baseline_metrics):
    exp_dir = os.path.join(EXP_BASE_DIR, exp_id)
    config_file_path = os.path.join(exp_dir, 'config.json')
    
    if os.path.exists(config_file_path):
        print(f"RESUMING DETECTED: {exp_id} already completed! Loading cached result.")
        with open(config_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'diagnostics'), exist_ok=True)

    log_path = os.path.join(exp_dir, 'run.log')
    log_file = open(log_path, 'w', encoding='utf-8')
    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    log(f"==================================================")
    log(f"  EXPERIMENT: {exp_id}")
    log(f"  Hypothesis: {hypothesis}")
    log(f"==================================================")

    verify_protected_artifacts(log_func=log)
    t0 = time.time()

    strategy = config_params.get('candidate_strategy', 'static_disk')
    cand_cap = config_params.get('candidate_cap', 350)
    posting_cap = config_params.get('posting_cap', 3000)
    thresh_us = config_params.get('threshold_us', 0.60)
    thresh_india = config_params.get('threshold_india', 0.50)
    thresh_other = config_params.get('threshold_other', 0.35)
    fallback_thresh = config_params.get('fallback_thresh', 0.30)

    # 1. Candidate Retrieval according to Strategy
    eval_cands = {}
    if strategy == 'static_disk':
        log(f"Loading static candidate file: {CAND_TRAIN_PATH}")
        with open(CAND_TRAIN_PATH, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if parts and parts[0] in gt_map:
                    cands = parts[1].split(',') if len(parts) > 1 and parts[1].strip() else []
                    eval_cands[parts[0]] = cands[:cand_cap]
                    if len(eval_cands) >= len(gt_map):
                        break
    elif strategy == 'evidence_preranking':
        log(f"Building dynamic candidates via evidence pre-ranking (Posting Cap: {posting_cap:,} | Candidate Cap: {cand_cap})...")
        f_name = {k: v for k, v in name_idx.items() if len(v) <= posting_cap}
        f_addr = {k: v for k, v in addr_idx.items() if len(v) <= min(posting_cap, 2000)}
        f_num = {k: v for k, v in num_idx.items() if len(v) <= min(posting_cap, 2000)}
        f_prefix = {k: v for k, v in prefix_idx.items() if len(v) <= min(posting_cap, 2000)}

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
            eval_cands[s1_id] = [x[1] for x in scored_cands[:cand_cap]]

    # 2. Inference & Evaluation
    hit_cand_pairs = 0
    tot_cands_cnt = 0
    tp_tot, fp_tot = 0, 0
    r_at_1, r_at_5, r_at_10, mrr_sum = 0, 0, 0, 0.0
    total_preds_cnt = 0
    total_gt_pairs = sum(len(v) for v in gt_map.values())
    peak_rss = 0

    blocking_misses, threshold_misses = 0, 0

    sub_chunk_size = 1000
    s1_ids_list = list(gt_map.keys())
    num_sub = math.ceil(len(s1_ids_list) / sub_chunk_size)

    for sc_i in range(num_sub):
        sub_s1 = s1_ids_list[sc_i * sub_chunk_size : (sc_i + 1) * sub_chunk_size]
        rows, meta = [], []

        for s1_id in sub_s1:
            if s1_id not in s1_feats:
                continue
            f1 = parse_entity_features(s1_feats[s1_id])
            cands = eval_cands.get(s1_id, [])
            true_set = gt_map[s1_id]

            tot_cands_cnt += len(cands)
            hit_cand_pairs += len(true_set.intersection(cands))
            blocking_misses += len(true_set - set(cands))

            total_cands_len = len(cands)
            for rank, cand_id in enumerate(cands, start=1):
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

        if rows:
            probs = clf.predict_proba(np.array(rows, dtype=np.float32))[:, 1]
            s1_probs = {}
            for (s1_id, cand_id), prob in zip(meta, probs):
                s1_probs.setdefault(s1_id, []).append((cand_id, float(prob)))

            for s1_id in sub_s1:
                true_set = gt_map[s1_id]
                f1 = parse_entity_features(s1_feats[s1_id]) if s1_id in s1_feats else {'country': 'US'}
                country = f1['country']

                if s1_id in s1_probs:
                    c_probs = s1_probs[s1_id]
                    c_probs.sort(key=lambda x: x[1], reverse=True)

                    for rank_i, (cid, p) in enumerate(c_probs, start=1):
                        if cid in true_set:
                            if rank_i == 1: r_at_1 += 1
                            if rank_i <= 5: r_at_5 += 1
                            if rank_i <= 10: r_at_10 += 1
                            mrr_sum += 1.0 / rank_i
                            break

                    if country == 'US':
                        selected = set(cid for cid, p in c_probs if p >= thresh_us)
                    elif country == 'India':
                        selected = set(cid for cid, p in c_probs if p >= thresh_india)
                    else:
                        selected = set(cid for cid, p in c_probs if p >= thresh_other)
                        if not selected and c_probs[0][1] >= fallback_thresh:
                            selected = {c_probs[0][0]}
                else:
                    selected = set()

                tp = len(selected & true_set)
                fp = len(selected - true_set)
                tp_tot += tp
                fp_tot += fp
                total_preds_cnt += len(selected)

                present_cands = set(eval_cands.get(s1_id, [])) & true_set
                threshold_misses += len(present_cands - selected)

        cur_rss = psutil.Process().memory_info().rss / (1024 * 1024)
        if cur_rss > peak_rss:
            peak_rss = cur_rss
        del rows, meta

    prec = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) > 0 else 0.0
    rec = tp_tot / total_gt_pairs if total_gt_pairs > 0 else 0.0
    f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (prec + rec) > 0 else 0.0
    cand_rec = hit_cand_pairs / total_gt_pairs if total_gt_pairs > 0 else 0.0
    rec_1 = r_at_1 / len(gt_map) if len(gt_map) > 0 else 0.0
    rec_5 = r_at_5 / len(gt_map) if len(gt_map) > 0 else 0.0
    rec_10 = r_at_10 / len(gt_map) if len(gt_map) > 0 else 0.0
    mrr = mrr_sum / len(gt_map) if len(gt_map) > 0 else 0.0
    runtime_sec = time.time() - t0

    # 3. Calculate Deltas
    delta_f05 = f05 - baseline_metrics['macro_f05']
    delta_cand_rec = cand_rec - baseline_metrics['candidate_blocking_recall']
    delta_prec = prec - baseline_metrics['precision']
    delta_rec = rec - baseline_metrics['recall']
    delta_r1 = rec_1 - baseline_metrics['recall_at_1']
    delta_mrr = mrr - baseline_metrics['mrr']

    is_retained = f05 > baseline_metrics['macro_f05']
    status_str = "RETAINED (NEW CURRENT BEST)" if is_retained else "ARCHIVED (NO F0.5 GAIN)"

    log(f"\n--- EVALUATION SUMMARY ({exp_id}) ---")
    log(f"Candidate Blocking Recall: {cand_rec*100:.2f}% (Delta: {delta_cand_rec*100:+.2f}%)")
    log(f"End-to-End Precision:      {prec*100:.2f}% (Delta: {delta_prec*100:+.2f}%)")
    log(f"End-to-End Recall:         {rec*100:.2f}% (Delta: {delta_rec*100:+.2f}%)")
    log(f"End-to-End Macro F0.5:     {f05:.4f} (Delta: {delta_f05:+.4f})")
    log(f"Recall@1:                  {rec_1*100:.2f}% (Delta: {delta_r1*100:+.2f}%)")
    log(f"Recall@5:                  {rec_5*100:.2f}%")
    log(f"Recall@10:                 {rec_10*100:.2f}%")
    log(f"MRR:                       {mrr:.4f} (Delta: {delta_mrr:+.4f})")
    log(f"Avg Candidates / Query:    {tot_cands_cnt / len(gt_map):.1f}")
    log(f"Runtime:                   {runtime_sec:.2f}s | Peak RAM: {peak_rss:.1f} MB")
    log(f"Status:                    {status_str}")

    # 4. Save Artifacts
    config_data = {
        'experiment_id': exp_id,
        'hypothesis': hypothesis,
        'candidate_strategy': strategy,
        'candidate_cap': cand_cap,
        'posting_cap': posting_cap,
        'thresholds': {'US': thresh_us, 'India': thresh_india, 'Other': thresh_other, 'Fallback': fallback_thresh},
        'queries_evaluated': len(gt_map),
        'metrics': {
            'candidate_blocking_recall': cand_rec,
            'precision': prec,
            'recall': rec,
            'macro_f05': f05,
            'recall_at_1': rec_1,
            'recall_at_5': rec_5,
            'recall_at_10': rec_10,
            'mrr': mrr,
            'avg_candidates_per_query': tot_cands_cnt / len(gt_map),
            'runtime_sec': runtime_sec,
            'peak_ram_mb': peak_rss
        },
        'deltas_vs_baseline': {
            'delta_macro_f05': delta_f05,
            'delta_cand_recall': delta_cand_rec,
            'delta_precision': delta_prec,
            'delta_recall': delta_rec,
            'delta_recall_at_1': delta_r1,
            'delta_mrr': delta_mrr
        },
        'retained': is_retained
    }
    with open(config_file_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2)

    df_metrics = pd.DataFrame([config_data['metrics']])
    df_metrics.to_csv(os.path.join(exp_dir, 'metrics.csv'), index=False)

    df_diag = pd.DataFrame([{
        'total_gt_pairs': total_gt_pairs,
        'blocking_misses': blocking_misses,
        'threshold_misses': threshold_misses,
        'false_positives': fp_tot,
        'true_positives': tp_tot
    }])
    df_diag.to_csv(os.path.join(exp_dir, 'diagnostics', 'failure_breakdown.csv'), index=False)

    verify_protected_artifacts(log_func=log)
    log_file.close()

    return config_data

def main():
    print("==================================================")
    print("  S++ OPTIMIZATION LOOP AUTOMATION ENGINE")
    print("==================================================")
    verify_protected_artifacts()

    num_queries = 20000
    val_gt_df = pd.read_csv(VAL_GT_PATH, sep='\t', dtype=str, keep_default_na=False).iloc[:num_queries]
    gt_map = {}
    for r in val_gt_df.itertuples():
        s1_id = r.source1_entity_id
        m_str = r.matched_entity_ids
        matches = set(x.strip() for x in m_str.split(',') if x.strip()) if m_str else set()
        gt_map[s1_id] = matches

    print("Loading cleaned entity feature tables...")
    usecols = ['entity_id', 'business_name_clean', 'name_significant_tokens', 'business_address_clean', 'country_clean', 'address_numbers']
    s1_df = pd.read_csv(os.path.join(CLEAN_DIR, 'clean_train_source1.tsv'), sep='\t', usecols=usecols, dtype=str).fillna('')
    s2_df = pd.read_csv(os.path.join(CLEAN_DIR, 'clean_train_source2.tsv'), sep='\t', usecols=usecols, dtype=str).fillna('')
    s3_df = pd.read_csv(os.path.join(CLEAN_DIR, 'clean_train_source3.tsv'), sep='\t', usecols=usecols, dtype=str).fillna('')

    s1_feats = {r.entity_id: (r.business_name_clean, r.name_significant_tokens, r.business_address_clean, r.country_clean, r.address_numbers) for r in s1_df.itertuples()}
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

    print("Building country-isolated inverted indexes...")
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

    clf = joblib.load(MODEL_PATH)

    baseline_metrics = {
        'candidate_blocking_recall': 0.7820,
        'precision': 0.9283,
        'recall': 0.6767,
        'macro_f05': 0.8640,
        'recall_at_1': 0.8336,
        'mrr': 0.8413
    }

    experiments_to_run = [
        {
            'exp_id': 'exp_001',
            'hypothesis': 'Baseline verification using static disk candidate streaming.',
            'params': {'candidate_strategy': 'static_disk', 'candidate_cap': 350}
        },
        {
            'exp_id': 'exp_002',
            'hypothesis': 'Candidate evidence pre-ranking (Cap 350, Posting Limit 3,000) eliminates high-frequency token noise.',
            'params': {'candidate_strategy': 'evidence_preranking', 'candidate_cap': 350, 'posting_cap': 3000}
        },
        {
            'exp_id': 'exp_003',
            'hypothesis': 'Candidate evidence pre-ranking with expanded candidate cap (Cap 500, Posting Limit 3,000).',
            'params': {'candidate_strategy': 'evidence_preranking', 'candidate_cap': 500, 'posting_cap': 3000}
        },
        {
            'exp_id': 'exp_004',
            'hypothesis': 'Evidence pre-ranking with relaxed posting limits (Cap 350, Posting Limit 10,000) to capture common-token entities.',
            'params': {'candidate_strategy': 'evidence_preranking', 'candidate_cap': 350, 'posting_cap': 10000}
        },
        {
            'exp_id': 'exp_005',
            'hypothesis': 'Evidence pre-ranking with relaxed posting limits and expanded candidate cap (Cap 500, Posting Limit 10,000).',
            'params': {'candidate_strategy': 'evidence_preranking', 'candidate_cap': 500, 'posting_cap': 10000}
        },
        {
            'exp_id': 'exp_006',
            'hypothesis': 'Country threshold calibration grid search on optimal pre-ranked blocking pipeline.',
            'params': {
                'candidate_strategy': 'evidence_preranking',
                'candidate_cap': 350,
                'posting_cap': 3000,
                'threshold_us': 0.55,
                'threshold_india': 0.45,
                'threshold_other': 0.30,
                'fallback_thresh': 0.25
            }
        }
    ]

    master_summary = []
    current_best_f05 = baseline_metrics['macro_f05']
    current_best_exp = 'exp_001'

    for exp in experiments_to_run:
        exp_id = exp['exp_id']
        hypothesis = exp['hypothesis']
        params = exp['params']

        res = run_experiment(exp_id, hypothesis, params, gt_map, s1_feats, s23_meta, name_idx, addr_idx, num_idx, prefix_idx, clf, baseline_metrics)
        master_summary.append(res)

        if res['metrics']['macro_f05'] > current_best_f05:
            current_best_f05 = res['metrics']['macro_f05']
            current_best_exp = exp_id

    # Master Table Generation
    df_summary = []
    for r in master_summary:
        m = r['metrics']
        d = r['deltas_vs_baseline']
        df_summary.append({
            'experiment_id': r['experiment_id'],
            'candidate_strategy': r.get('candidate_strategy', 'static_disk'),
            'candidate_cap': r.get('candidate_cap', 350),
            'posting_cap': r.get('posting_cap', 3000),
            'candidate_blocking_recall': f"{m['candidate_blocking_recall']*100:.2f}%",
            'precision': f"{m['precision']*100:.2f}%",
            'recall': f"{m['recall']*100:.2f}%",
            'macro_f05': f"{m['macro_f05']:.4f}",
            'delta_f05': f"{d['delta_macro_f05']:+.4f}",
            'recall_at_1': f"{m['recall_at_1']*100:.2f}%",
            'mrr': f"{m['mrr']:.4f}",
            'avg_candidates_per_query': f"{m['avg_candidates_per_query']:.1f}",
            'runtime_sec': f"{m['runtime_sec']:.2f}s",
            'peak_ram_mb': f"{m['peak_ram_mb']:.1f}MB",
            'retained': "YES" if r.get('retained', False) else "NO"
        })

    master_df = pd.DataFrame(df_summary)
    master_csv = os.path.join(EXP_BASE_DIR, 'master_experiment_summary.csv')
    master_df.to_csv(master_csv, index=False)

    print("\n==================================================")
    print("  OPTIMIZATION LOOP EXECUTION COMPLETE")
    print("==================================================")
    print(f"Master Summary saved to: {master_csv}")
    print(f"Current Best Experiment: {current_best_exp} (Macro F0.5 = {current_best_f05:.4f})")
    print(f"Primary Target:          {TARGET_PRIMARY:.6f}")
    print(f"Primary Target Gap:      {TARGET_PRIMARY - current_best_f05:.4f}")
    print("==================================================")

if __name__ == '__main__':
    main()
