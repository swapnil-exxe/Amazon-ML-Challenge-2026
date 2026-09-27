# ARYA — Sneha Handoff Report

## PROJECT: ML Challenge 2026 — Entity Matching
## PIPELINE: Manthan → Phase 2 Cleaning → Phase 3 Blocking → ARYA Phase 4 ML Matching

### PIPELINE STATUS:
- **Manthan Phase 2 & 3**: **COMPLETE**
- **ARYA Phase 4**: **COMPLETE**
- **Inference Status**: **COMPLETE** (1,732,544 / 1,732,544 test S1 entities)
- **Official Validator**: **PASS**
- **Data Integrity**: **PASS**
- **Submission Output**: **READY**
- **Sneha Handoff**: **READY**

---

## A. What Manthan Completed
1. **Phase 2 Data Cleaning**: Cleaned, normalized, and standardized noisy string attributes (`business_name`, `business_address`, `country`) across all Source 1, Source 2, and Source 3 train and test datasets.
2. **Phase 3 Candidate Blocking**: Generated candidate entity pairs (`candidate_pairs_test_v2.tsv`) reducing ~32.2 Billion pairwise candidate space to ~403.5 Million highly relevant candidate pairs.
3. **Artifact Paths**:
   - Cleaned Test S1: `/Users/swapnil/Documents/ML/manthan_output/phase_2_cleaning/cleaned/clean_test_source1.tsv`
   - Cleaned Test S2: `/Users/swapnil/Documents/ML/manthan_output/phase_2_cleaning/cleaned/clean_test_source2.tsv`
   - Cleaned Test S3: `/Users/swapnil/Documents/ML/manthan_output/phase_2_cleaning/cleaned/clean_test_source3.tsv`
   - Candidate Pairs Test: `/Users/swapnil/Documents/ML/manthan_output/phase_3_blocking/candidates/candidate_pairs_test_v2.tsv`

---

## B. What ARYA Phase 4 Completed
1. **ML Model Architecture**: Trained `HistGradientBoostingClassifier` checkpoint on 16 exact string, token, country, and numerical overlap features.
2. **16-Feature Candidate Scoring**: Extracted name Jaccard, address Jaccard, sig Jaccard, exact country, numeric overlap, prefix match, length ratio, and character diff features for all candidate pairs.
3. **Ultra-Fast Memory-Safe Streaming Inference**: Indexed attribute tokens upfront (1.8 GB RAM footprint) and evaluated candidate pairs in vectorized NumPy sub-batches, processing the entire test set in **2.72 hours**.
4. **Candidate Selection & Ranking**: Dynamically selected candidates with probability `p >= 0.35` (fallback `p >= 0.30`), generating formatted tab-separated candidate match strings per S1 entity.

---

## C. Verified Final Statistics
- **Total Test S1 Entities Processed**: **1,732,544**
- **Total Lines in `matching_results.tsv`**: **1,732,545** (1 header line + 1,732,544 data rows)
- **Duplicate S1 IDs**: **0**
- **Malformed Rows**: **0**
- **Missing S1 IDs**: **0**
- **Extra S1 IDs**: **0**
- **Official Validator Output**: `PASS — no blocking issues found. Safe to submit.`

---

## D. Files Sneha Needs

1. **Primary Leaderboard Submission File**:
   `/Users/swapnil/Documents/ML/arya_output/matching_results.tsv`
2. **Supporting ML Model Checkpoint**:
   `/Users/swapnil/Documents/ML/arya_output/arya_model.joblib`
3. **Inference Code Script**:
   `/Users/swapnil/Documents/ML/arya_output/run_arya_inference_only.py`
4. **Comprehensive Completion Report**:
   `/Users/swapnil/Documents/ML/ARYA_FINAL_REPORT.md`
5. **Sneha Handoff Guide**:
   `/Users/swapnil/Documents/ML/ARYA_SNEHA_HANDOFF.md`
6. **Execution Metrics Summary**:
   `/Users/swapnil/Documents/ML/arya_output/arya_final_summary.json`
7. **Handoff Manifest**:
   `/Users/swapnil/Documents/ML/arya_output/SNEHA_HANDOFF_MANIFEST.txt`

---

## E. CRITICAL WARNINGS FOR SNEHA

- **DO NOT RERUN ARYA INFERENCE.**
- **DO NOT RETRAIN THE MODEL.**
- **DO NOT MODIFY `matching_results.tsv`.**
- **DO NOT REGENERATE LARGE MANTHAN PHASE 2/3 FILES.**
- `matching_results.tsv` is fully verified and ready to submit to the competition portal.
