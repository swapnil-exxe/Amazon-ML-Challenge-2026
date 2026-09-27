# GO2 Phase 4 — Final Completion Report

## 1. Project Overview
This project addresses the **ML Challenge 2026 Business Entity Resolution Problem**. The goal is to perform multi-source entity matching across 3 noisy business record sources (Source 1 reference, Source 2, and Source 3) across US, India, and France entities, evaluating predictions using the precision-heavy $F_{0.5}$ metric.

---

## 2. Go1 Phase 2 & Phase 3 Completion
- **Phase 2 Cleaning**: Standardized noisy names, addresses, and country labels across Source 1, 2, and 3 datasets.
- **Phase 3 Candidate Blocking**: Generated candidate entity pairs (`candidate_pairs_test_v2.tsv`), shrinking the search space to ~403.5 Million highly relevant candidate pairs.
- **Artifact Status**: Cleaned files (`clean_test_source1.tsv`, `clean_test_source2.tsv`, `clean_test_source3.tsv`) and candidate pairs are 100% verified, intact, and readable.

---

## 3. GO2 Phase 4 Completion
- **Pipeline Stage**: Machine Learning Entity Matching, candidate scoring, ranking, and final prediction file generation.
- **Status**: **COMPLETE**. All 1,732,544 test S1 entities have been scored and output to `matching_results.tsv`.

---

## 4. Model Details
- **Checkpoint Path**: `/Users/swapnil/Documents/ML/go2_output/go2_model.joblib`
- **Algorithm**: `sklearn.ensemble.HistGradientBoostingClassifier`
- **Input Dimension**: **16 Features** (Name Jaccard, Sig Jaccard, Address Jaccard, Exact Country Match, Numeric Overlap, Name Prefix Match, Length Ratio, Character Diff across S1-S2 and S1-S3 pairs)
- **Model Checkpoint Status**: Verified valid and intact.

---

## 5. Streaming Inference Details
- **Inference Script**: `/Users/swapnil/Documents/ML/go2_output/run_go2_inference_only.py`
- **Total S1 Entities Processed**: **1,732,544**
- **Candidate Pairs Evaluated**: ~403.5 Million candidate pairs
- **Execution Time**: 9,775.0 seconds (**2.72 hours**)
- **Memory Footprint**: Active attribute indexing in **1.8 GB RAM**, peak **6.6 GB RAM** (no swap thrashing)
- **Output File Line Count**: **1,732,545 lines** (1 header + 1,732,544 data rows)
- **Output File Size**: **75.95 MB**

---

## 6. Final Output Statistics
- **Target Line Count**: 1,732,545 lines
- **Actual Line Count**: 1,732,545 lines
- **Header Line**: `source1_entity_id\tmatched_entity_ids`
- **Duplicate S1 IDs**: **0**
- **Malformed Rows**: **0**
- **Missing S1 IDs**: **0**
- **Extra S1 IDs**: **0**

---

## 7. Official Validation Result
Official validator command:
```bash
python3 /Users/swapnil/Documents/ML/student_resource/utils/validate_submission.py \
  --matching /Users/swapnil/Documents/ML/go2_output/matching_results.tsv \
  --test-dir /Users/swapnil/Documents/ML/student_resource/dataset/test
```
Validator output:
```text
ML Challenge 2026 — submission validator
  test dir: /Users/swapnil/Documents/ML/student_resource/dataset/test
  required S1 entities: 1732544
  matching_results.tsv: 1732544 rows (403389 empty, 1329155 non-empty).

PASS — no blocking issues found. Safe to submit.
```

---

## 8. Data Integrity Status
- **Raw Datasets (`student_resource/dataset/`)**: Read-only & Untouched.
- **Go1 Outputs (`go1_output/`)**: Read-only & Untouched.
- **Prediction File (`matching_results.tsv`)**: Validated & Intact.

---

## 9. Exact Handoff Files
- `go2_output/matching_results.tsv` (Leaderboard submission output)
- `go2_output/go2_model.joblib` (Trained model)
- `go2_output/run_go2_inference_only.py` (Inference code)
- `go2_output/go2_inference.log` (Execution log)
- `go2_output/go2_final_summary.json` (Summary JSON)
- `go2_output/GO2_GO3_HANDOFF.md` (Handoff guide)
- `go2_output/GO3_HANDOFF_MANIFEST.txt` (Short manifest)
- `GO2_FINAL_REPORT.md` (This report)

---

## 10. Instructions for Go3
1. Submit `matching_results.tsv` to the competition portal.
2. **DO NOT RERUN GO2 INFERENCE.**
3. **DO NOT RETRAIN THE MODEL.**
4. **DO NOT MODIFY `matching_results.tsv`.**

---

## 11. Explicit Completion Statement
**NO FURTHER TECHNICAL PROCESSING IS REQUIRED. Go1 + GO2 work is 100% complete and ready to hand over to Go3.**
