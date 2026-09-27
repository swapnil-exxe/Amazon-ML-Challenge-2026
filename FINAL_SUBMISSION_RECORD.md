# FINAL SUBMISSION RECORD — ML CHALLENGE 2026

**Project**: Business Entity Resolution (Manthan + ARYA Technical Pipeline)  
**Date**: September 27, 2026  
**Pipeline Status**: **FROZEN & LOCKED (Project Closeout Complete)**  

---

### 1. Final Submission State

#### Primary Baseline (Locked Portal Submission)
- **Prediction File**: `/Users/swapnil/Documents/ML/arya_output/matching_results.tsv` (`d1c898796e2ab192b627f8d0e8ea83f96e1427b029a8d66ed9165c87bd94d1e8`)
- **Submission ZIP**: `/Users/swapnil/Documents/ML/team_submission.zip` (`459e67da33e82ea6b052de382cbdf0805c6cbb09a5ca2138415fa6a7cd043747`)
- **Model Checkpoint**: `/Users/swapnil/Documents/ML/arya_output/arya_model.joblib` (`476763ebe4051718167703dbf28e88a85ef80e2982876ccdb028508793534cad`)
- **Integrity Status**: **BASELINE INTEGRITY: VERIFIED** (Original baseline remains 100% untouched).

#### V2 Entry (Country-Specific Thresholds — Additional Submission)
- **Prediction File**: `/Users/swapnil/Documents/ML/arya_output/matching_results_v2.tsv` (`56d88eb1f3b7f1e6dd23b3b626266b9cfb9b59a9257f9e9feedc6a6fc7a72d73`)
- **Submission ZIP**: `/Users/swapnil/Documents/ML/team_submission_v2.zip` (`d6926b559404028a9fdd2aca167438f0cead828fb4991ede0474ee7f8ad46d7c`)
- **Source Script in ZIP**: `code/business_entity_resolution/src/run_arya_inference_v2.py` (**VERIFIED PRESENT**)
- **Model Checkpoint in ZIP**: `code/business_entity_resolution/src/arya_model.joblib` (**VERIFIED PRESENT**)
- **Official Validator Output**: `PASS — no blocking issues found. Safe to submit.`
- **ZIP Byte-Integrity**: `PASS` (Internal `output/matching_results.tsv` verified 100% byte-identical to standalone V2 file).
- **Line Count**: Exactly `1,732,545` lines (1 header + 1,732,544 test entity rows).
- **Submission Nature**: V2 is prepared as an **ADDITIONAL** submission, never a replacement.

---

### 2. Final Model Results

- **Model Architecture**: `sklearn.ensemble.HistGradientBoostingClassifier`
- **Features Used**: `16` tabular entity-resolution features
- **Validation Precision**: **~87.37%**
- **Validation Recall**: **~70.66%**
- **Validation $F_{0.5}$ Score**: **~0.760**
- **Blocking Recall**: **78.1184%**

> [!NOTE]
> *These are validation/controlled-experiment results. Final leaderboard performance is determined by the competition's hidden evaluation set.*

---

### 3. Final V2 Threshold Configuration

- **United States (`US`)**: $p \ge 0.60$
- **India**: $p \ge 0.50$
- **France**: $p \ge 0.35$ with fallback $p \ge 0.30$

#### Empirical Evidence Behind Threshold Selection:
- **India Optimization**:
  - Threshold `0.35`: $F_{0.5} = 0.7395$
  - Threshold `0.50`: $F_{0.5} = 0.7451$
  - *Result*: $p \ge 0.50$ was retained for India as a genuine precision/F0.5 improvement.
- **France Consideration**:
  - No labelled training ground truth was available in the competition dataset, so no quantitative threshold optimization claim is made for France. Fallback values ($0.35/0.30$) were safely preserved.

---

### 4. Candidate Blocking Experiment

- **Production Blocking Recall**: **78.1184%** (`5,966,972` / `7,638,365` true ground truth pairs)
- **Experimental Multi-Pass Blocking Recall**: **79.5882%**
- **Recall Gain**: **+1.4698 percentage points**
- **Production Candidate Count**: `621,736,279` total pairs
- **Experimental Candidate Count**: `1,450,210,000` total pairs
- **Candidate Growth**: **+133.25%** (+828.47 Million candidate pairs)
- **Production Downstream $F_{0.5}$**: **0.760**
- **Experimental Downstream $F_{0.5}$**: **0.748**
- **FINAL DECISION**: **KEEP PRODUCTION BLOCKING (EXPERIMENTAL BLOCKING REJECTED)**

> [!IMPORTANT]
> **Reason**: The experimental blocking increased candidate recall but produced substantially more candidate noise and reduced downstream validation $F_{0.5}$. This evidence demonstrates that this tested alternative was not beneficial; it is not described as proof that no better blocking strategy exists.

---

### 5. Submission Safety Verification

- **Baseline SHA256 Checksum Verification**:
  - `matching_results.tsv`: `d1c898796e2ab192b627f8d0e8ea83f96e1427b029a8d66ed9165c87bd94d1e8` — **MATCH**
  - `team_submission.zip`: `459e67da33e82ea6b052de382cbdf0805c6cbb09a5ca2138415fa6a7cd043747` — **MATCH**
  - `arya_model.joblib`: `476763ebe4051718167703dbf28e88a85ef80e2982876ccdb028508793534cad` — **MATCH**
  - **Status**: **BASELINE INTEGRITY: VERIFIED**

---

### 6. V2 Submission Package Integrity

- `matching_results_v2.tsv`: `56d88eb1f3b7f1e6dd23b3b626266b9cfb9b59a9257f9e9feedc6a6fc7a72d73` — **MATCH**
- `team_submission_v2.zip`: `d6926b559404028a9fdd2aca167438f0cead828fb4991ede0474ee7f8ad46d7c` — **MATCH & FULLY COMPLIANT**

---

### 7. Portal Action Record

V2 is prepared as a separate/additional submission. Portal upload must be performed manually by the team.

---

```text
========================================
FINAL ML CHALLENGE STATUS
========================================

PIPELINE:
FROZEN

BASELINE:
LOCKED / VERIFIED

V2:
VALIDATED / READY

OFFICIAL VALIDATOR:
PASS

ZIP INTEGRITY:
PASS

BASELINE MODIFIED:
NO

EXPERIMENTAL BLOCKING:
REJECTED

PRODUCTION BLOCKING:
RETAINED

VALIDATION F0.5:
~0.760

BLOCKING RECALL:
78.1184%

FINAL ML EXPERIMENT:
COMPLETE

FURTHER EXPERIMENTATION:
STOP

PORTAL ACTION:
V2 MAY BE SUBMITTED AS AN ADDITIONAL ENTRY
ORIGINAL MUST REMAIN UNTOUCHED

========================================
```
