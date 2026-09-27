#!/usr/bin/env python3
"""
S++ Challenger 2 & 3 Optimization Experiments — ML Challenge 2026
Evaluates Zero-Candidate Fallback Retrieval & Adaptive Candidate Cap (400) on 20,000 S1 validation split.
"""

import os
import sys
import time
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc

print("=== S++ CHALLENGER EXPERIMENTS 2 & 3 (VALIDATION SPLIT) ===")
t0 = time.time()

BASE_DIR = '/Users/swapnil/Documents/ML'
GT_PATH = os.path.join(BASE_DIR, 'student_resource/dataset/train/train_ground_truth.tsv')
TRAIN_S1_PATH = os.path.join(BASE_DIR, 'manthan_output/phase_2_cleaning/cleaned/clean_train_source1.tsv')
TRAIN_S2_PATH = os.path.join(BASE_DIR, 'manthan_output/phase_2_cleaning/cleaned/clean_train_source2.tsv')
TRAIN_S3_PATH = os.path.join(BASE_DIR, 'manthan_output/phase_2_cleaning/cleaned/clean_train_source3.tsv')
CAND_TRAIN_PATH = os.path.join(BASE_DIR, 'manthan_output/phase_3_blocking/candidates/candidate_pairs_train_v2.tsv')

# Load Ground Truth validation split
gt_df = pd.read_csv(GT_PATH, sep='\t', dtype=str, keep_default_na=False)
gt_map = {}
val_s1_set = set(gt_df['source1_entity_id'].values[:20000])

for r in gt_df.itertuples():
    s1_id = r.source1_entity_id
    if s1_id not in val_s1_set:
        continue
    m_str = r.matched_entity_ids
    matches = set(x.strip() for x in m_str.split(',') if x.strip()) if m_str else set()
    if matches:
        gt_map[s1_id] = matches

total_val_gt_pairs = sum(len(v) for v in gt_map.values())
print(f"Loaded GT for {len(val_s1_set):,} validation S1 queries ({total_val_gt_pairs:,} total true pairs).")

# Helper string functions
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
    feats = {}
    for r in df.itertuples():
        feats[r.entity_id] = (r.business_name_clean, r.name_significant_tokens, r.business_address_clean, r.country_clean, r.address_numbers)
    return feats

print("Loading clean feature tuples into memory...")
s1_feats = load_feature_tuples(TRAIN_S1_PATH)
s2_feats = load_feature_tuples(TRAIN_S2_PATH)
s3_feats = load_feature_tuples(TRAIN_S3_PATH)
s23_feats = {**s2_feats, **s3_feats}
del s2_feats, s3_feats

def parse_tuple(tup):
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

# Build validation feature matrix for Challenger 1 (20 Features)
print("Extracting 20-feature matrix for Challenger 1...")
rows, y_val, meta_val = [], [], []

cand_file = open(CAND_TRAIN_PATH, 'r', encoding='utf-8')
next(cand_file)

cands_per_query = []
zero_cands_count = 0
reachable_hits = 0

for line in cand_file:
    parts = line.rstrip('\n').split('\t')
    if not parts or not parts[0]:
        continue
    s1_id = parts[0]
    if s1_id not in val_s1_set:
        continue
    cands = parts[1].split(',') if len(parts) > 1 and parts[1].strip() else []
    c_len = len(cands)
    cands_per_query.append(c_len)
    if c_len == 0:
        zero_cands_count += 1
    
    true_set = gt_map.get(s1_id, set())
    if true_set:
        reachable_hits += len(true_set.intersection(cands))
        
    if s1_id not in s1_feats:
        continue
    f1 = parse_tuple(s1_feats[s1_id])
    
    for rank, cand_id in enumerate(cands, start=1):
        if cand_id not in s23_feats:
            continue
        f2 = parse_tuple(s23_feats[cand_id])
        label = 1 if cand_id in true_set else 0
        
        base_f = [
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
            float(c_len)
        ]
        
        l_sim = levenshtein_sim(f1['name'], f2['name'])
        len_ratio = min(f1['name_len'], f2['name_len']) / max(f1['name_len'], f2['name_len'], 1)
        tok_overlap_ratio = len(f1['name_tokens'] & f2['name_tokens']) / max(len(f1['name_tokens']), 1)
        num_overlap_bool = 1.0 if (f1['num_tokens'] & f2['num_tokens']) else 0.0
        
        chal_f = base_f + [l_sim, len_ratio, tok_overlap_ratio, num_overlap_bool]
        rows.append(chal_f)
        y_val.append(label)
        meta_val.append((s1_id, cand_id))

print(f"\n--- CHALLENGER 1 SUMMARY ---")
print(f"Validation S1 Queries Evaluated: {len(cands_per_query):,}")
print(f"Validation Blocking Recall: {reachable_hits/total_val_gt_pairs*100:.4f}% ({reachable_hits:,} / {total_val_gt_pairs:,})")
print(f"Average Candidates/Query: {np.mean(cands_per_query):.2f}")
print(f"Zero Candidates Count: {zero_cands_count:,} ({zero_cands_count/len(cands_per_query)*100:.2f}%)")

X_mat = np.array(rows, dtype=np.float32)
y_arr = np.array(y_val, dtype=np.int32)

split_idx = int(len(y_arr) * 0.7)
X_tr, X_te = X_mat[:split_idx], X_mat[split_idx:]
y_tr, y_te = y_arr[:split_idx], y_arr[split_idx:]
meta_te = meta_val[split_idx:]

print("\nFitting Challenger 20-Feature GBDT Classifier...")
clf = HistGradientBoostingClassifier(max_iter=100, random_state=42)
clf.fit(X_tr, y_tr)
probs = clf.predict_proba(X_te)[:, 1]

# Macro F0.5 evaluation at optimal p=0.55
s1_map = {}
for (s1_id, cand_id), pr in zip(meta_te, probs):
    s1_map.setdefault(s1_id, []).append((cand_id, pr))

scores = []
for s1_id, cands in s1_map.items():
    true_set = gt_map.get(s1_id, set())
    pred_set = set(c_id for c_id, pr in cands if pr >= 0.55)
    
    if not true_set and not pred_set:
        scores.append(1.0)
        continue
    if not pred_set:
        scores.append(0.0)
        continue
        
    tp = len(true_set & pred_set)
    prec = tp / len(pred_set)
    rec = tp / len(true_set) if true_set else 0.0
    
    if prec + rec == 0:
        scores.append(0.0)
    else:
        f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
        scores.append(f05)

macro_f05 = np.mean(scores)

y_pred = (probs >= 0.55).astype(int)
prec = precision_score(y_te, y_pred, zero_division=0)
rec = recall_score(y_te, y_pred, zero_division=0)
f1 = f1_score(y_te, y_pred, zero_division=0)
roc = roc_auc_score(y_te, probs)
precision_curve, recall_curve, _ = precision_recall_curve(y_te, probs)
pr_auc = auc(recall_curve, precision_curve)

print("\n==================================================")
print("   S++ CHALLENGER 1 (20 FEAT, p=0.55) RESULTS     ")
print("==================================================")
print(f"Candidate Pair Precision: {prec*100:.2f}%")
print(f"Candidate Pair Recall:    {rec*100:.2f}%")
print(f"Candidate Pair F1:        {f1*100:.2f}%")
print(f"ROC-AUC:                  {roc:.4f}")
print(f"PR-AUC:                   {pr_auc:.4f}")
print(f"Validation Macro F0.5:    {macro_f05:.4f}")
print(f"Total Experiment Time:    {time.time() - t0:.2f}s")
