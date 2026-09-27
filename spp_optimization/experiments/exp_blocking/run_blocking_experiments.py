#!/usr/bin/env python3
"""
Phase 3 Candidate Blocking Optimization Experiments — ML Challenge 2026 S++ Optimization
Tests evidence-based pre-ranking and variable posting limits on the 20K validation benchmark.
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
from collections import defaultdict

# Expected SHA256 Hashes for Protected Production Artifacts
EXPECTED_HASHES = {
    '/Users/swapnil/Documents/ML/team_submission.zip': '459e67da33e82ea6b052de382cbdf0805c6cbb09a5ca2138415fa6a7cd043747',
    '/Users/swapnil/Documents/ML/team_submission_v2.zip': 'd6926b559404028a9fdd2aca167438f0cead828fb4991ede0474ee7f8ad46d7c',
    '/Users/swapnil/Documents/ML/arya_output/matching_results.tsv': 'd1c898796e2ab192b627f8d0e8ea83f96e1427b029a8d66ed9165c87bd94d1e8',
    '/Users/swapnil/Documents/ML/arya_output/matching_results_v2.tsv': '56d88eb1f3b7f1e6dd23b3b626266b9cfb9b59a9257f9e9feedc6a6fc7a72d73',
    '/Users/swapnil/Documents/ML/arya_output/arya_model.joblib': '476763ebe4051718167703dbf28e88a85ef80e2982876ccdb028508793534cad'
}

def verify_protected_artifacts():
    print("=== VERIFYING PROTECTED ARTIFACT INTEGRITY ===")
    all_pass = True
    for path, exp in EXPECTED_HASHES.items():
        if not os.path.exists(path):
            print(f"FAIL: {path} does not exist!")
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
        print(f"  - {os.path.basename(path):<25} | SHA256: {digest[:16]}... | {status}")
    if not all_pass:
        print("CRITICAL: SHA256 Verification Failed! Stopping.")
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
    verify_protected_artifacts()

    exp_dir = '/Users/swapnil/Documents/ML/spp_optimization/experiments'
    blocking_dir = os.path.join(exp_dir, 'exp_blocking')
    os.makedirs(blocking_dir, exist_ok=True)

    val_gt_path = os.path.join(exp_dir, 'val_ground_truth.tsv')
    clean_dir = '/Users/swapnil/Documents/ML/manthan_output/phase_2_cleaning/cleaned'
    model_path = '/Users/swapnil/Documents/ML/arya_output/arya_model.joblib'

    num_queries = 20000
    print(f"=== PHASE 3: BLOCKING OPTIMIZATION EXPERIMENTS ({num_queries:,} QUERIES) ===")

    # 1. Load Ground Truth
    val_gt_df = pd.read_csv(val_gt_path, sep='\t', dtype=str, keep_default_na=False).iloc[:num_queries]
    gt_map = {}
    for r in val_gt_df.itertuples():
        s1_id = r.source1_entity_id
        m_str = r.matched_entity_ids
        matches = set(x.strip() for x in m_str.split(',') if x.strip()) if m_str else set()
        gt_map[s1_id] = matches

    total_gt_pairs = sum(len(v) for v in gt_map.values())
    print(f"Loaded {len(gt_map):,} queries ({total_gt_pairs:,} true match pairs).")

    # 2. Load Cleaned Feature Data
    print("Loading cleaned entity feature tables...")
    usecols = ['entity_id', 'business_name_clean', 'name_significant_tokens', 'business_address_clean', 'country_clean', 'address_numbers']
    
    s1_df = pd.read_csv(os.path.join(clean_dir, 'clean_train_source1.tsv'), sep='\t', usecols=usecols, dtype=str).fillna('')
    s2_df = pd.read_csv(os.path.join(clean_dir, 'clean_train_source2.tsv'), sep='\t', usecols=usecols, dtype=str).fillna('')
    s3_df = pd.read_csv(os.path.join(clean_dir, 'clean_train_source3.tsv'), sep='\t', usecols=usecols, dtype=str).fillna('')

    s1_feats = {r.entity_id: (r.business_name_clean, r.name_significant_tokens, r.business_address_clean, r.country_clean, r.address_numbers) for r in s1_df.itertuples()}
    
    s23_meta = {} # eid -> (country, sig_toks_set, addr_toks_set, addr_nums_set, norm_name, tuple_feats)
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

    clf = joblib.load(model_path)

    # Controlled Configurations to Evaluate
    configs = [
        {'exp_id': 'EXP-B1', 'posting_cap': 3000, 'cand_cap': 350, 'pre_ranking': True},
        {'exp_id': 'EXP-B2', 'posting_cap': 3000, 'cand_cap': 500, 'pre_ranking': True},
        {'exp_id': 'EXP-B3', 'posting_cap': 10000, 'cand_cap': 350, 'pre_ranking': True},
        {'exp_id': 'EXP-B4', 'posting_cap': 10000, 'cand_cap': 500, 'pre_ranking': True}
    ]

    all_results = []
    baseline_f05 = 0.8640
    baseline_cand_rec = 0.7820

    for cfg in configs:
        exp_id = cfg['exp_id']
        post_cap = cfg['posting_cap']
        cand_cap = cfg['cand_cap']
        
        print(f"\n==================================================")
        print(f"  RUNNING {exp_id} (Posting Cap: {post_cap:,} | Candidate Cap: {cand_cap})")
        print(f"==================================================")
        t0 = time.time()

        # Build filtered indexes according to posting cap
        f_name = {k: v for k, v in name_idx.items() if len(v) <= post_cap}
        f_addr = {k: v for k, v in addr_idx.items() if len(v) <= min(post_cap, 2000)}
        f_num = {k: v for k, v in num_idx.items() if len(v) <= min(post_cap, 2000)}
        f_prefix = {k: v for k, v in prefix_idx.items() if len(v) <= min(post_cap, 2000)}

        hit_cand_pairs = 0
        tot_cands_cnt = 0
        tp_tot, fp_tot = 0, 0
        r_at_1_hits, mrr_sum = 0, 0.0
        total_preds_cnt = 0
        peak_rss = 0

        sub_chunk_size = 1000
        s1_ids_list = list(gt_map.keys())
        num_sub = math.ceil(len(s1_ids_list) / sub_chunk_size)

        for sc_i in range(num_sub):
            sub_s1 = s1_ids_list[sc_i * sub_chunk_size : (sc_i + 1) * sub_chunk_size]
            rows, meta = [], []

            for s1_id in sub_s1:
                if s1_id not in s1_feats:
                    continue
                f1_tup = s1_feats[s1_id]
                f1 = parse_entity_features(f1_tup)
                country = f1['country']
                norm_name = f1['name']
                sig_toks = f1['sig_tokens']
                addr_toks = f1['addr_tokens']
                addr_nums = f1['num_tokens']

                # Candidate Retrieval & Evidence Scoring
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

                cands_keys = list(cand_hits.keys())
                scored_cands = []
                for cid in cands_keys:
                    sc = cand_hits[cid]
                    c_country, c_sig, c_addr, c_nums, c_name, _ = s23_meta[cid]
                    if norm_name and norm_name == c_name:
                        sc += 10.0
                    scored_cands.append((sc, cid))

                # Deterministic Evidence Sort & Truncation Cap
                scored_cands.sort(key=lambda x: (-x[0], x[1]))
                top_cands = [x[1] for x in scored_cands[:cand_cap]]

                true_set = gt_map[s1_id]
                tot_cands_cnt += len(top_cands)
                hit_cand_pairs += len(true_set.intersection(top_cands))

                total_cands_len = len(top_cands)
                for rank, cand_id in enumerate(top_cands, start=1):
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

        prec = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) > 0 else 0.0
        rec = tp_tot / total_gt_pairs if total_gt_pairs > 0 else 0.0
        f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (prec + rec) > 0 else 0.0
        cand_rec = hit_cand_pairs / total_gt_pairs if total_gt_pairs > 0 else 0.0
        r_at_1 = r_at_1_hits / len(gt_map) if len(gt_map) > 0 else 0.0
        mrr = mrr_sum / len(gt_map) if len(gt_map) > 0 else 0.0
        runtime_sec = time.time() - t0

        delta_f05 = f05 - baseline_f05
        delta_cand_rec = cand_rec - baseline_cand_rec

        print(f"Results for {exp_id}:")
        print(f"  Candidate Blocking Recall: {cand_rec * 100:.2f}% (Delta: {delta_cand_rec * 100:+.2f}%)")
        print(f"  End-to-End Precision:      {prec * 100:.2f}%")
        print(f"  End-to-End Recall:         {rec * 100:.2f}%")
        print(f"  End-to-End Macro F0.5:     {f05:.4f} (Delta: {delta_f05:+.4f})")
        print(f"  Recall@1:                  {r_at_1 * 100:.2f}%")
        print(f"  MRR:                       {mrr:.4f}")
        print(f"  Average Candidates/Query:  {tot_cands_cnt / len(gt_map):.1f}")
        print(f"  Runtime:                   {runtime_sec:.2f}s | Peak RAM: {peak_rss:.1f} MB")

        res_dict = {
            'exp_id': exp_id,
            'posting_cap': post_cap,
            'candidate_cap': cand_cap,
            'candidate_blocking_recall': cand_rec,
            'delta_cand_recall': delta_cand_rec,
            'precision': prec,
            'recall': rec,
            'macro_f05': f05,
            'delta_macro_f05': delta_f05,
            'recall_at_1': r_at_1,
            'mrr': mrr,
            'predicted_pairs': total_preds_cnt,
            'avg_candidates_per_query': tot_cands_cnt / len(gt_map),
            'runtime_sec': runtime_sec,
            'peak_ram_mb': peak_rss
        }
        all_results.append(res_dict)

    res_df = pd.DataFrame(all_results)
    out_csv = os.path.join(blocking_dir, 'blocking_experiments_results.csv')
    res_df.to_csv(out_csv, index=False)
    print(f"\nAll blocking optimization experiments saved to: {out_csv}")

    verify_protected_artifacts()

if __name__ == '__main__':
    main()
