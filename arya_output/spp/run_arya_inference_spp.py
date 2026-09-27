#!/usr/bin/env python3
"""
ARYA — S++ 20-Feature Streaming Inference Pipeline
Amazon ML Challenge 2026
Outputs to arya_output/spp/matching_results_spp.tsv.
Never modifies original matching_results.tsv or matching_results_v2.tsv!
"""

import os
import sys
import time
import math
import joblib
import numpy as np
import pandas as pd

BASE_DIR = '/Users/swapnil/Documents/ML'
CLEAN_DIR = os.path.join(BASE_DIR, 'manthan_output/phase_2_cleaning/cleaned')
CAND_DIR = os.path.join(BASE_DIR, 'manthan_output/phase_3_blocking/candidates')
SPP_DIR = os.path.join(BASE_DIR, 'arya_output/spp')
os.makedirs(SPP_DIR, exist_ok=True)

TEST_CAND_PATH = os.path.join(CAND_DIR, 'candidate_pairs_test_v2.tsv')
MODEL_PATH = os.path.join(SPP_DIR, 'arya_model_spp.joblib')
MATCHING_RESULTS_SPP_PATH = os.path.join(SPP_DIR, 'matching_results_spp.tsv')

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

def levenshtein_sim(str1, str2):
    if not str1 and not str2:
        return 1.0
    if not str1 or not str2:
        return 0.0
    if str1 == str2:
        return 1.0
    l1, l2 = len(str1), len(str2)
    if abs(l1 - l2) > 10:
        return 0.0
    set1, set2 = set(str1), set(str2)
    return len(set1 & set2) / max(len(set1 | set2), 1)

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

def read_cand_chunk(file_obj, chunk_size):
    chunk = {}
    for line in file_obj:
        parts = line.rstrip('\n').split('\t')
        if not parts or not parts[0]:
            continue
        s1_id = parts[0]
        cands = parts[1].split(',') if len(parts) > 1 and parts[1].strip() else []
        chunk[s1_id] = cands
        if len(chunk) >= chunk_size:
            break
    return chunk

def main():
    start_time = time.time()
    print("==================================================")
    print("  ARYA PHASE 4: S++ 20-FEATURE STREAMING INFERENCE")
    print("==================================================")
    print(f"Loading trained model from: {MODEL_PATH}")
    clf = joblib.load(MODEL_PATH)

    test_s1_path = os.path.join(CLEAN_DIR, 'clean_test_source1.tsv')
    print("Reading test S1 country mapping...")
    test_s1_df = pd.read_csv(test_s1_path, sep='\t', usecols=['entity_id', 'country_clean'], dtype=str).fillna('')
    s1_country_map = dict(zip(test_s1_df['entity_id'], test_s1_df['country_clean']))

    out_file = open(MATCHING_RESULTS_SPP_PATH, 'w', encoding='utf-8')
    out_file.write("source1_entity_id\tmatched_entity_ids\n")
    out_file.flush()

    print("\n--- FAST 1-PASS FEATURE TUPLE INDEXING ---")
    s1_test_feats = load_feature_tuples(test_s1_path)
    s2_test_feats = load_feature_tuples(os.path.join(CLEAN_DIR, 'clean_test_source2.tsv'))
    s3_test_feats = load_feature_tuples(os.path.join(CLEAN_DIR, 'clean_test_source3.tsv'))
    s2_s3_feats = {**s2_test_feats, **s3_test_feats}
    del s2_test_feats, s3_test_feats

    print("\n--- STREAMING INFERENCE ---")
    chunk_size = 50000
    sub_chunk_size = 5000

    cand_file = open(TEST_CAND_PATH, 'r', encoding='utf-8')
    next(cand_file)

    processed_cnt = 0
    while True:
        chunk = read_cand_chunk(cand_file, chunk_size)
        if not chunk:
            break

        s1_ids_in_chunk = list(chunk.keys())
        num_sub_chunks = math.ceil(len(s1_ids_in_chunk) / sub_chunk_size)
        
        for sc_i in range(num_sub_chunks):
            sub_s1_ids = s1_ids_in_chunk[sc_i * sub_chunk_size : (sc_i + 1) * sub_chunk_size]
            rows, meta = [], []
            
            for s1_id in sub_s1_ids:
                if s1_id not in s1_test_feats:
                    continue
                f1 = parse_entity_features_from_tuple(s1_test_feats[s1_id])
                cands = chunk[s1_id]
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
                        float(total_cands),
                        levenshtein_sim(f1['name'], f2['name']),
                        min(f1['name_len'], f2['name_len']) / max(f1['name_len'], f2['name_len'], 1),
                        len(f1['name_tokens'] & f2['name_tokens']) / max(len(f1['name_tokens']), 1),
                        1.0 if (f1['num_tokens'] & f2['num_tokens']) else 0.0
                    ]
                    rows.append(row)
                    meta.append((s1_id, cand_id))

            if rows:
                X_sub = np.array(rows, dtype=np.float32)
                probs = clf.predict_proba(X_sub)[:, 1]

                s1_probs = {}
                for (s1_id, cand_id), prob in zip(meta, probs):
                    s1_probs.setdefault(s1_id, []).append((cand_id, float(prob)))

                for s1_id in sub_s1_ids:
                    country = s1_country_map.get(s1_id, 'US')
                    
                    if s1_id in s1_probs:
                        c_probs = s1_probs[s1_id]
                        c_probs.sort(key=lambda x: x[1], reverse=True)
                        
                        if country == 'US':
                            selected = [c_id for c_id, p in c_probs if p >= 0.60]
                        elif country == 'India':
                            selected = [c_id for c_id, p in c_probs if p >= 0.50]
                        else:
                            selected = [c_id for c_id, p in c_probs if p >= 0.35]
                            if not selected and c_probs[0][1] >= 0.30:
                                selected = [c_probs[0][0]]
                        
                        out_file.write(f"{s1_id}\t{','.join(selected)}\n")
                    else:
                        out_file.write(f"{s1_id}\t\n")
            else:
                for s1_id in sub_s1_ids:
                    out_file.write(f"{s1_id}\t\n")
                    
            processed_cnt += len(sub_s1_ids)

        out_file.flush()
        print(f"Processed {processed_cnt:,} / {len(test_s1_ordered):,} test queries...")

    out_file.close()
    cand_file.close()
    print(f"COMPLETE! Output generated at: {MATCHING_RESULTS_SPP_PATH}")
    print(f"Total Inference Time: {time.time() - start_time:.2f}s")

if __name__ == '__main__':
    main()
