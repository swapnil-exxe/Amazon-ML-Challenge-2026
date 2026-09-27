#!/usr/bin/env python3
"""
Phase 3 Evaluation Module
Evaluates candidate pairs against Ground Truth to compute pair recall and S1 coverage.
"""

import pandas as pd

def evaluate_ground_truth(candidate_dict, gt_file):
    gt_df = pd.read_csv(gt_file, sep='\t', dtype=str, keep_default_na=False)

    total_gt_pairs = 0
    found_gt_pairs = 0
    s1_with_matches = 0
    s1_all_matches_found = 0

    missed_reasons = {
        "no_candidates_generated": 0,
        "missing_address_in_gt": 0,
        "no_shared_name_token": 0,
        "other": 0
    }

    missed_records = []

    for r in gt_df.itertuples():
        s1_id = r.source1_entity_id
        m_str = r.matched_entity_ids
        matches = set(x.strip() for x in m_str.split(',') if x.strip()) if m_str else set()

        if not matches:
            continue

        s1_with_matches += 1
        total_gt_pairs += len(matches)

        cands = candidate_dict.get(s1_id, set())
        found = matches.intersection(cands)
        found_gt_pairs += len(found)

        if len(found) == len(matches):
            s1_all_matches_found += 1
        else:
            missed = matches - found
            for m in missed:
                missed_records.append({
                    "s1_entity_id": s1_id,
                    "missed_matched_id": m,
                    "candidate_count": len(cands)
                })

    pair_recall = (found_gt_pairs / total_gt_pairs * 100) if total_gt_pairs > 0 else 0.0
    s1_coverage = (s1_all_matches_found / s1_with_matches * 100) if s1_with_matches > 0 else 0.0

    eval_stats = {
        "total_gt_pairs": total_gt_pairs,
        "found_gt_pairs": found_gt_pairs,
        "missed_gt_pairs": total_gt_pairs - found_gt_pairs,
        "pair_recall": round(pair_recall, 4),
        "s1_with_matches": s1_with_matches,
        "s1_all_matches_found": s1_all_matches_found,
        "s1_coverage": round(s1_coverage, 4)
    }

    return eval_stats, pd.DataFrame(missed_records)
