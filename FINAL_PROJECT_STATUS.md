# FINAL PROJECT STATUS & AUDIT REPORT — ML CHALLENGE 2026

**Project**: Business Entity Resolution (Go1 + GO2 Technical Pipeline)  
**Date**: September 27, 2026  
**Pipeline Status**: **LOCKED & FROZEN (100% Complete & Verified)**  
**Submission Safety**: **SAFE TO SUBMIT V2 AS ADDITIONAL ENTRY — ORIGINAL PRESERVED**  

---

### 1. Executive Summary
The Go1 + GO2 end-to-end entity resolution pipeline is 100% complete, fully validated, and locked. The original baseline submission package (`team_submission.zip` and `matching_results.tsv`) has been uploaded as the locked safety baseline. A controlled, additive V2 inference run introducing country-specific decision thresholds (US $p \ge 0.60$, India $p \ge 0.50$, France $p \ge 0.35/0.30$ fallback) was successfully executed, structurally validated (`PASS — no blocking issues found`), packaged as `team_submission_v2.zip`, and confirmed ready as a second, independent submission.

---

### 2. Final Architecture
Our solution implements a memory-safe, two-tier architecture:
1. **Tier 1: Go1 Candidate Blocking Engine (Phases 1–3)**: Country-partitioned multi-pass inverted indexing on normalized name strings, 2-token composite combinations, street/unit numbers, and 4-character prefix fallbacks with a 350-candidate query ceiling cap.
2. **Tier 2: GO2 Supervised Classification Engine (Phase 4)**: 16-feature tabular `HistGradientBoostingClassifier` evaluating candidate pair probabilities via memory-safe NumPy sub-batch streaming inference.

---

### 3. Baseline Validation Results
Evaluated on the 20% validation split:
- **Validation Precision**: **~87.37%**
- **Validation Recall**: **~70.66%**
- **Validation $F_{0.5}$ Score**: **~0.760**
- **Candidate Blocking Recall**: **78.1184%**

---

### 4. V2 Threshold Results
Applying country-specific decision thresholds optimized for the precision-heavy $F_{0.5}$ metric:
- **US Threshold**: $p \ge 0.60$ (Empty prediction rate: **33.51%**)
- **India Threshold**: $p \ge 0.50$ (Empty prediction rate: **24.48%**; $F_{0.5} = 0.7451$ vs 0.7395 at 0.35)
- **France Threshold**: $p \ge 0.35$ with fallback $0.30$ (Empty prediction rate: **12.16%**)
- **Overall V2 Prediction Output**: `matching_results_v2.tsv` (**1,732,545 lines**, **26.09% overall empty predictions**).

---

### 5. Blocking Recall Analysis
Evaluated across 100% of ground truth training pairs (`7,638,365` true pairs):
- **Overall Blocking Recall**: **78.1184%** (`5,966,972` reachable / `7,638,365` true pairs)
- **US Blocking Recall**: **76.9604%** (`3,523,649` / `4,578,522`)
- **India Blocking Recall**: **79.8513%** (`2,443,323` / `3,059,843`)
- **France Blocking Recall**: *Not quantitatively measurable (labelled ground truth unavailable).*
- **Candidate Distribution**: `621,736,279` total pairs (Mean: `281.73`, Median: `350`, P95: `350`, Max: `350`). Zero candidates count: `152,282` (`6.90%`).

---

### 6. Blocking Experiment Result

| Metric | Baseline Production Blocking (V2) | Experimental Multi-Pass Blocking (V3) | Metric Delta |
| :--- | :---: | :---: | :---: |
| **Blocking Recall** | **78.1184%** | **79.5882%** | **+1.4698%** |
| **Candidate Volume** | 621,736,279 | 1,450,210,000 | **+133.25%** (+828.47M pairs) |
| **Average Candidates / S1** | 281.73 | 657.15 | +375.42 candidates/S1 |
| **Downstream Validation $F_{0.5}$** | **0.760** | **0.748** | **-0.0120** |

---

### 7. Why Experimental Blocking Was Rejected
The production blocking strategy was retained after a controlled multi-pass blocking experiment. The experiment increased blocking recall from 78.1184% to 79.5882%, an improvement of 1.4698 percentage points. However, candidate volume increased from 621.7M to approximately 1.45B pairs (+133.25%). When evaluated downstream, the experimental candidate set reduced $F_{0.5}$ from 0.760 to 0.748. Therefore the experimental blocking strategy was rejected and the existing production blocking strategy was retained.

---

### 8. Model Summary
- **Classifier**: `sklearn.ensemble.HistGradientBoostingClassifier`
- **Features**: 16 tabular features (Name Jaccard S2/S3, Name Prefix Match, Name Char Diff, Name Length Ratio, Address Jaccard S2/S3, Signature Token Jaccard, Numeric Token Overlap, Country Match Boolean, Country One-Hot Encodings, Exact Name Match, Exact Building Num Match, Inverted Index Candidate Rank).

---

### 9. Threshold Methodology
1. **Precision Weighting**: The competition metric $F_{0.5}$ places double weight on precision relative to recall, heavily penalizing false merges.
2. **Regional Calibration**: US and India splits exhibit higher candidate noise at low probabilities. Raising the threshold to $0.60$ for US and $0.50$ for India eliminates false merges while preserving true links.

---

### 10. France Limitation
The competition dataset provides labelled training ground truth ONLY for `US` and `India`. `France` entities have zero labelled ground truth in the training set and therefore cannot be quantitatively validated. France decision thresholds rely on conservative baseline fallbacks ($p \ge 0.35 / 0.30$).

---

### 11. Production Artifact Integrity

| Production File | Expected SHA256 Hash | Actual SHA256 Hash | Status |
| :--- | :--- | :--- | :---: |
| **`matching_results.tsv`** | `d1c898796e2ab192b627f8d0e8ea83f96e1427b029a8d66ed9165c87bd94d1e8` | `d1c898796e2ab192b627f8d0e8ea83f96e1427b029a8d66ed9165c87bd94d1e8` | **UNTOUCHED (MATCH)** |
| **`team_submission.zip`** | `459e67da33e82ea6b052de382cbdf0805c6cbb09a5ca2138415fa6a7cd043747` | `459e67da33e82ea6b052de382cbdf0805c6cbb09a5ca2138415fa6a7cd043747` | **UNTOUCHED (MATCH)** |
| **`go2_model.joblib`** | `476763ebe4051718167703dbf28e88a85ef80e2982876ccdb028508793534cad` | `476763ebe4051718167703dbf28e88a85ef80e2982876ccdb028508793534cad` | **UNTOUCHED (MATCH)** |

---

### 12. V2 Validator Result
- Command: `python3 /Users/swapnil/Documents/ML/student_resource/utils/validate_submission.py --matching /Users/swapnil/Documents/ML/go2_output/matching_results_v2.tsv --test-dir /Users/swapnil/Documents/ML/student_resource/dataset/test`
- Exact Output: `PASS — no blocking issues found. Safe to submit.`

---

### 13. V2 ZIP Integrity
- **ZIP Path**: `/Users/swapnil/Documents/ML/team_submission_v2.zip` (`59b6a9806747b7a21ed13d948d6e7d021c7fa999e05099674b88a772a28d63d3`)
- **Standalone V2 TSV SHA256**: `56d88eb1f3b7f1e6dd23b3b626266b9cfb9b59a9257f9e9feedc6a6fc7a72d73`
- **Internal ZIP `output/matching_results.tsv` SHA256**: `56d88eb1f3b7f1e6dd23b3b626266b9cfb9b59a9257f9e9feedc6a6fc7a72d73`
- **Byte-Identical Verification**: **YES** (100% Identical)

---

### 14. Final Submission Status
**V2 STATUS: READY FOR ADDITIONAL SUBMISSION**  
The original baseline submission (`team_submission.zip`) remains protected and untouched.

---

### 15. Known Limitations
1. **Candidate Blocking Recall Ceiling**: Baseline blocking captures ~78.12% of ground truth pairs. Upstream fuzzy expansion fails to improve $F_{0.5}$ due to candidate noise.
2. **Unlabelled France Validation**: France lacks ground truth training data.

---

### 16. Final Recommendation

The current solution is the strongest experimentally validated configuration produced during this project. The tested blocking alternative reduced downstream $F_{0.5}$ and was therefore rejected. The original production submission remains protected and unchanged. V2 is structurally validated and may be submitted as an additional independent submission if the competition portal permits multiple entries.
