#!/usr/bin/env python3
"""
S++ Threshold Grid Search on Challenger 20-Feature Model
Determines the optimal macro F0.5 thresholds on the validation split.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

print("=== S++ THRESHOLD GRID SEARCH ON CHALLENGER (20 FEATURES) ===")
from run_spp_challenger_experiments import test_meta, p_chal, gt_map

thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

# s1_id -> list of (cand_id, prob)
s1_map = {}
for (s1_id, cand_id), prob in zip(test_meta, p_chal):
    s1_map.setdefault(s1_id, []).append((cand_id, prob))

print(f"Evaluating {len(s1_map):,} test validation queries across {len(thresholds)} threshold levels...\n")

results = []
for t in thresholds:
    scores = []
    for s1_id, cands in s1_map.items():
        true_set = gt_map.get(s1_id, set())
        pred_set = set(c_id for c_id, pr in cands if pr >= t)
        
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
            
    avg_f05 = np.mean(scores)
    print(f"  Threshold = {t:.2f} -> Validation Macro F0.5 = {avg_f05:.4f}")
    results.append((t, avg_f05))

best_t, best_f05 = max(results, key=lambda x: x[1])
print(f"\nOptimal Threshold: {best_t:.2f} with Macro F0.5 = {best_f05:.4f}")
