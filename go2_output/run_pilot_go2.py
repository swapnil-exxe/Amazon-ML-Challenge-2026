#!/usr/bin/env python3
"""
GO2 — Phase 4 Pilot Feature Engineering & Supervised Model Validation (25,000 S1 Sample)
"""

import os
import sys
import time
import json
import psutil
import shutil
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc

# Resource Monitoring
process = psutil.Process(os.getpid())
start_time = time.time()
start_mem_mb = process.memory_info().rss / (1024 * 1024)

# Paths
BASE_DIR = '/Users/swapnil/Documents/ML'
CLEAN_DIR = os.path.join(BASE_DIR, 'go1_output/phase_2_cleaning/cleaned')
CAND_DIR = os.path.join(BASE_DIR, 'go1_output/phase_3_blocking/candidates')
GT_PATH = os.path.join(BASE_DIR, 'student_resource/dataset/train/train_ground_truth.tsv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'go2_output')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load Ground Truth into dict for fast lookup
print("Loading Ground Truth...")
gt_dict = {}
with open(GT_PATH, 'r', encoding='utf-8') as f:
    header = next(f)
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if len(parts) == 2 and parts[1].strip():
            gt_dict[parts[0]] = set(parts[1].split(','))
        else:
            gt_dict[parts[0]] = set()

print(f"Loaded GT for {len(gt_dict):,} S1 entities.")

# 2. Select first 25,000 S1 entities from candidate_pairs_train_v2.tsv
PILOT_S1_COUNT = 25000
cand_v2_path = os.path.join(CAND_DIR, 'candidate_pairs_train_v2.tsv')

print(f"Streaming candidate pairs for first {PILOT_S1_COUNT:,} S1 entities...")
pilot_candidates = {} # s1_id -> list of candidate_ids
all_required_s2_s3 = set()
all_required_s1 = set()

with open(cand_v2_path, 'r', encoding='utf-8') as f:
    header = next(f) # header: source1_entity_id \t candidate_entity_ids
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if not parts or not parts[0]:
            continue
        s1_id = parts[0]
        cands = parts[1].split(',') if len(parts) > 1 and parts[1].strip() else []
        pilot_candidates[s1_id] = cands
        all_required_s1.add(s1_id)
        for c in cands:
            all_required_s2_s3.add(c)
        if len(pilot_candidates) >= PILOT_S1_COUNT:
            break

print(f"Collected {len(pilot_candidates):,} S1 entities requiring {len(all_required_s2_s3):,} unique S2/S3 candidate records.")

# 3. Streamingly load only required S1, S2, S3 features from cleaned files
def load_entity_features(tsv_path, required_ids):
    features = {}
    with open(tsv_path, 'r', encoding='utf-8') as f:
        header_line = next(f).rstrip('\n').split('\t')
        id_idx = header_line.index('entity_id')
        name_idx = header_line.index('business_name_clean')
        sig_name_idx = header_line.index('name_significant_tokens')
        addr_idx = header_line.index('business_address_clean')
        country_idx = header_line.index('country_clean')
        nums_idx = header_line.index('address_numbers')
        
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) <= max(id_idx, name_idx, sig_name_idx, addr_idx, country_idx, nums_idx):
                continue
            e_id = parts[id_idx]
            if e_id in required_ids:
                name = parts[name_idx]
                sig = parts[sig_name_idx]
                addr = parts[addr_idx]
                country = parts[country_idx]
                nums = parts[nums_idx]
                
                name_tokens = set(name.split()) if name else set()
                sig_tokens = set(sig.split()) if sig else set()
                addr_tokens = set(addr.split()) if addr else set()
                num_tokens = set(nums.split()) if nums else set()
                
                features[e_id] = {
                    'name': name,
                    'sig_name': sig,
                    'addr': addr,
                    'country': country,
                    'nums': nums,
                    'name_tokens': name_tokens,
                    'sig_tokens': sig_tokens,
                    'addr_tokens': addr_tokens,
                    'num_tokens': num_tokens,
                    'name_len': len(name),
                    'addr_len': len(addr)
                }
    return features

print("Loading S1 cleaned features...")
s1_feats = load_entity_features(os.path.join(CLEAN_DIR, 'clean_train_source1.tsv'), all_required_s1)

print("Loading S2 cleaned features...")
s2_feats = load_entity_features(os.path.join(CLEAN_DIR, 'clean_train_source2.tsv'), all_required_s2_s3)

print("Loading S3 cleaned features...")
s3_feats = load_entity_features(os.path.join(CLEAN_DIR, 'clean_train_source3.tsv'), all_required_s2_s3)

s2_s3_feats = {**s2_feats, **s3_feats}
del s2_feats, s3_feats

print(f"Loaded features for {len(s1_feats):,} S1 and {len(s2_s3_feats):,} S2/S3 entities.")

# Helper functions for fast feature extraction
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

# 4. Feature Extraction Loop
feature_rows = []
labels = []
s1_groups = []
pair_s1_ids = []
pair_cand_ids = []

feat_extract_start = time.time()

for s1_id, cands in pilot_candidates.items():
    if s1_id not in s1_feats:
        continue
    f1 = s1_feats[s1_id]
    gt_matches = gt_dict.get(s1_id, set())
    total_cands = len(cands)
    
    for rank, cand_id in enumerate(cands, start=1):
        if cand_id not in s2_s3_feats:
            continue
        f2 = s2_s3_feats[cand_id]
        
        # Labels
        is_match = 1 if cand_id in gt_matches else 0
        
        # Features
        exact_name = 1.0 if f1['name'] and f1['name'] == f2['name'] else 0.0
        exact_sig_name = 1.0 if f1['sig_name'] and f1['sig_name'] == f2['sig_name'] else 0.0
        name_jaccard = token_jaccard(f1['name_tokens'], f2['name_tokens'])
        sig_jaccard = token_jaccard(f1['sig_tokens'], f2['sig_tokens'])
        name_len_diff = abs(f1['name_len'] - f2['name_len'])
        prefix_match = prefix_4_match(f1['name'], f2['name'])
        
        exact_addr = 1.0 if f1['addr'] and f1['addr'] == f2['addr'] else 0.0
        addr_jaccard = token_jaccard(f1['addr_tokens'], f2['addr_tokens'])
        has_missing_addr = 1.0 if not f2['addr'] else 0.0
        exact_num_match = 1.0 if f1['nums'] and f1['nums'] == f2['nums'] else 0.0
        num_jaccard = token_jaccard(f1['num_tokens'], f2['num_tokens'])
        addr_len_diff = abs(f1['addr_len'] - f2['addr_len'])
        
        country_match = 1.0 if f1['country'] == f2['country'] else 0.0
        is_s3 = 1.0 if cand_id.startswith('S3-') else 0.0
        cand_rank = float(rank)
        cand_count = float(total_cands)
        
        row = [
            exact_name, exact_sig_name, name_jaccard, sig_jaccard, name_len_diff, prefix_match,
            exact_addr, addr_jaccard, has_missing_addr, exact_num_match, num_jaccard, addr_len_diff,
            country_match, is_s3, cand_rank, cand_count
        ]
        
        feature_rows.append(row)
        labels.append(is_match)
        s1_groups.append(s1_id)
        pair_s1_ids.append(s1_id)
        pair_cand_ids.append(cand_id)

feat_extract_time = time.time() - feat_extract_start

X = np.array(feature_rows, dtype=np.float32)
y = np.array(labels, dtype=np.int32)
s1_groups = np.array(s1_groups)

print(f"Extracted {X.shape[0]:,} candidate pairs across {len(set(s1_groups)):,} S1 entities.")
print(f"Features: {X.shape[1]}, Positive Pairs: {np.sum(y):,} ({np.mean(y)*100:.3f}%), Negative Pairs: {len(y)-np.sum(y):,}")

# 5. Grouped S1 Train / Validation Split (80% train S1s, 20% validation S1s)
unique_s1 = np.unique(s1_groups)
np.random.seed(42)
np.random.shuffle(unique_s1)

split_idx = int(len(unique_s1) * 0.8)
train_s1_set = set(unique_s1[:split_idx])
val_s1_set = set(unique_s1[split_idx:])

train_mask = np.isin(s1_groups, list(train_s1_set))
val_mask = np.isin(s1_groups, list(val_s1_set))

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
val_s1_groups = s1_groups[val_mask]
val_cand_ids = np.array(pair_cand_ids)[val_mask]

print(f"Train split: {X_train.shape[0]:,} pairs ({len(train_s1_set):,} S1s)")
print(f"Val split: {X_val.shape[0]:,} pairs ({len(val_s1_set):,} S1s)")

# 6. Fit Model
model_start = time.time()
print("Training HistGradientBoostingClassifier...")
clf = HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.1,
    max_depth=8,
    random_state=42
)
clf.fit(X_train, y_train)
model_train_time = time.time() - model_start
print(f"Model training finished in {model_train_time:.2f} seconds.")

# 7. Evaluate Model Validation & Ranking Performance
val_probs = clf.predict_proba(X_val)[:, 1]

# Binary Metrics
val_preds = (val_probs >= 0.5).astype(int)
prec = precision_score(y_val, val_preds, zero_division=0)
rec = recall_score(y_val, val_preds, zero_division=0)
f1 = f1_score(y_val, val_preds, zero_division=0)
roc_auc = roc_auc_score(y_val, val_probs)
p_curve, r_curve, _ = precision_recall_curve(y_val, val_probs)
pr_auc = auc(r_curve, p_curve)

print(f"Validation Pair Metrics -> Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")

# Per-S1 Ranking Metrics
val_df = pd.DataFrame({
    's1_id': val_s1_groups,
    'cand_id': val_cand_ids,
    'y_true': y_val,
    'prob': val_probs
})

# Group by s1_id and compute ranking metrics
r1_count = 0
r3_count = 0
r5_count = 0
r10_count = 0
mrr_sum = 0.0
total_val_s1_with_positives = 0

grouped = val_df.groupby('s1_id')
for s1_id, group in grouped:
    gt_matches = gt_dict.get(s1_id, set())
    if not gt_matches:
        continue
    total_val_s1_with_positives += 1
    
    # Sort candidates by predicted probability descending
    sorted_group = group.sort_values(by='prob', ascending=False)
    sorted_cands = sorted_group['cand_id'].tolist()
    
    # Find rank of first true positive
    first_rank = None
    for r_idx, c_id in enumerate(sorted_cands, start=1):
        if c_id in gt_matches:
            first_rank = r_idx
            break
            
    if first_rank is not None:
        if first_rank <= 1:
            r1_count += 1
        if first_rank <= 3:
            r3_count += 1
        if first_rank <= 5:
            r5_count += 1
        if first_rank <= 10:
            r10_count += 1
        mrr_sum += 1.0 / first_rank

recall_at_1 = r1_count / total_val_s1_with_positives if total_val_s1_with_positives > 0 else 0.0
recall_at_3 = r3_count / total_val_s1_with_positives if total_val_s1_with_positives > 0 else 0.0
recall_at_5 = r5_count / total_val_s1_with_positives if total_val_s1_with_positives > 0 else 0.0
recall_at_10 = r10_count / total_val_s1_with_positives if total_val_s1_with_positives > 0 else 0.0
mrr = mrr_sum / total_val_s1_with_positives if total_val_s1_with_positives > 0 else 0.0

print(f"Per-S1 Ranking Metrics ({total_val_s1_with_positives:,} GT S1s):")
print(f"  Recall@1:  {recall_at_1*100:.2f}%")
print(f"  Recall@3:  {recall_at_3*100:.2f}%")
print(f"  Recall@5:  {recall_at_5*100:.2f}%")
print(f"  Recall@10: {recall_at_10*100:.2f}%")
print(f"  MRR:       {mrr:.4f}")

# Resource Consumption
peak_mem_mb = process.memory_info().rss / (1024 * 1024)
total_pilot_time = time.time() - start_time

# Disk space check
total_d, used_d, free_d = shutil.disk_usage(BASE_DIR)
free_gb = free_d / (1024**3)

# Scaling Runtime Estimations
# Full train = 2,206,821 S1 entities (88x the pilot)
# Full test = 1,732,544 S1 entities (69x the pilot)
est_full_train_hours = (feat_extract_time * 88.0 + model_train_time * 5.0) / 3600.0
est_full_test_hours = (feat_extract_time * 69.0) / 3600.0

# Pilot Metrics Output
pilot_report = {
    "pilot_s1_processed": len(pilot_candidates),
    "candidate_pairs_processed": X.shape[0],
    "positive_pairs": int(np.sum(y)),
    "negative_pairs": int(len(y) - np.sum(y)),
    "feature_count": X.shape[1],
    "feature_generation_time_sec": round(feat_extract_time, 2),
    "model_train_time_sec": round(model_train_time, 2),
    "peak_ram_mb": round(peak_mem_mb, 2),
    "disk_free_gb": round(free_gb, 2),
    "validation_metrics": {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4)
    },
    "ranking_metrics": {
        "val_s1_eval_count": total_val_s1_with_positives,
        "recall_at_1": round(recall_at_1, 4),
        "recall_at_3": round(recall_at_3, 4),
        "recall_at_5": round(recall_at_5, 4),
        "recall_at_10": round(recall_at_10, 4),
        "mrr": round(mrr, 4)
    },
    "estimated_runtimes": {
        "full_train_hours": round(est_full_train_hours, 2),
        "full_test_hours": round(est_full_test_hours, 2)
    },
    "safety_assessment": "PASS" if free_gb > 2.0 and peak_mem_mb < 8000 else "REVIEW"
}

with open(os.path.join(OUTPUT_DIR, 'pilot_metrics.json'), 'w') as f:
    json.dump(pilot_report, f, indent=2)

print("\nPilot Execution Completed Successfully.")
print(json.dumps(pilot_report, indent=2))
