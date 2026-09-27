# FINAL SUBMISSION LOCK & LAST SAFETY AUDIT REPORT — ML CHALLENGE 2026

**Project**: Business Entity Resolution (Manthan + ARYA Technical Pipeline)  
**Date**: 27 September 2026  
**Deadline**: Today, 11:59 PM IST  
**Pipeline Status**: **FROZEN & LOCKED (100% Verified)**  

---

### 1. Baseline SHA256 Verification (Locked Safety Baseline)
- `matching_results.tsv`: `d1c898796e2ab192b627f8d0e8ea83f96e1427b029a8d66ed9165c87bd94d1e8` (**VERIFIED MATCH**)
- `team_submission.zip`: `459e67da33e82ea6b052de382cbdf0805c6cbb09a5ca2138415fa6a7cd043747` (**VERIFIED MATCH**)
- `arya_model.joblib`: `476763ebe4051718167703dbf28e88a85ef80e2982876ccdb028508793534cad` (**VERIFIED MATCH**)
- **Baseline Production Artifact Status**: **UNTOUCHED & MODIFIED: NO**

---

### 2. V2 Submission Package SHA256 & Validation
- `matching_results_v2.tsv`: `56d88eb1f3b7f1e6dd23b3b626266b9cfb9b59a9257f9e9feedc6a6fc7a72d73` (**VERIFIED MATCH**)
- `team_submission_v2.zip`: `59b6a9806747b7a21ed13d948d6e7d021c7fa999e05099674b88a772a28d63d3` (**VERIFIED MATCH**)
- **V2 Output Row Count**: Exactly `1,732,545` lines (1 header + 1,732,544 test data rows).
- **Official Validator Result**: `PASS — no blocking issues found. Safe to submit.`

---

### 3. V2 ZIP Byte-Integrity Verification
- Standalone `matching_results_v2.tsv` SHA256: `56d88eb1f3b7f1e6dd23b3b626266b9cfb9b59a9257f9e9feedc6a6fc7a72d73`
- Extracted `output/matching_results.tsv` SHA256: `56d88eb1f3b7f1e6dd23b3b626266b9cfb9b59a9257f9e9feedc6a6fc7a72d73`
- **ZIP Integrity Status**: **PASS — 100% Byte-Identical**

---

### 4. Final V2 Country Thresholds & Distribution
- **Decision Thresholds**:
  - **US**: $p \ge 0.60$ (Empty: `222,181` / `663,106` = **33.51%**)
  - **India**: $p \ge 0.50$ (Empty: `198,256` / `809,986` = **24.48%**; $F_{0.5} = 0.7451$ vs 0.7395 at 0.35)
  - **France**: $p \ge 0.35$, fallback $p \ge 0.30$ (Empty: `31,558` / `259,452` = **12.16%**)
- **Overall Prediction Output**: `1,280,549` non-empty, `451,995` empty (**26.09% overall empty**).

---

### 5. Candidate Blocking Audit & Decision
- **Production Blocking Recall**: **78.1184%** (`5,966,972` reachable / `7,638,365` true ground truth pairs).
- **Experimental Multi-Pass Blocking**:
  - Recall: **79.5882%** (+1.4698% gain).
  - Candidate Volume: Exploded from 621.7M to **1.45 Billion** (+133.25% growth).
  - Downstream $F_{0.5}$: Dropped from **0.760 to 0.748** due to candidate noise.
- **Final Blocking Decision**: **KEEP CURRENT PRODUCTION BLOCKING (EXPERIMENTAL BLOCKING REJECTED)**.

---

### 6. Model Architecture & Metrics
- **Classifier**: `HistGradientBoostingClassifier` (16 tabular features).
- **Validation Metrics**: Precision: **~87.37%**, Recall: **~70.66%**, Validation $F_{0.5}$: **~0.760**.

---

### 7. Documentation Inventory Status
- `Documentation_template.md`: **COMPLETE**
- `FINAL_PROJECT_STATUS.md`: **COMPLETE**
- `FINAL_ARTIFACT_INVENTORY.md`: **COMPLETE**
- `FINAL_V2_SUBMISSION_REPORT.md`: **COMPLETE**
- `BLOCKING_RECALL_EXPERIMENT_REPORT.md`: **COMPLETE**

---

### 8. Final Submission Instructions
- **Submission 1 (Primary Baseline)**: `team_submission.zip` & `matching_results.tsv` (Already uploaded to portal).
- **Submission 2 (Additive V2 Entry)**: `team_submission_v2.zip` & `matching_results_v2.tsv` (Ready for portal upload as second independent entry).
- **Safety Directive**: DO NOT replace, withdraw, or overwrite the baseline submission. Keep both live.
