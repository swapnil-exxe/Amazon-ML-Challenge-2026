# S++ OPTIMIZATION & METHODOLOGICAL AUDIT REPORT — ML CHALLENGE 2026

**Project**: Business Entity Resolution (Manthan + ARYA Pipeline)  
**Date**: September 27, 2026  
**Audit Purpose**: Methodological rigor inspection, training sample evaluation, threshold selection bias audit, and production safety lock.  
**Production Status**: **LOCKED & UNTOUCHED (Cryptographically Verified)**  

---

### 1. Verified Production Baseline vs Experimental S++ Challenger

> [!CAUTION]
> **Methodological Status Note**: The S++ Challenger results represent **experimental validation-set benchmarks** evaluated on a model trained on a 2.27% sample subset (50,000 S1 queries). Because the training population differs from the production baseline and decision thresholds were selected on the evaluation split, the $0.8339$ Macro $F_{0.5}$ result is **NOT considered a validated estimate of production generalization performance** and is **NOT approved for production deployment**.

| Metric / Parameter | Verified Production Baseline | S++ Challenger (Validation Benchmark) | Methodological Evaluation Status |
| :--- | :---: | :---: | :--- |
| **Candidate Pair Precision** | ~91.65% | 93.65% *(at $p=0.55$)* | Experimental Benchmark (In-Sample) |
| **Candidate Pair Recall** | ~88.47% | 88.09% *(at $p=0.55$)* | Experimental Benchmark (In-Sample) |
| **Candidate Pair F1** | ~90.04% | 90.78% *(at $p=0.55$)* | Experimental Benchmark (In-Sample) |
| **ROC-AUC Score** | 0.9986 | 0.9963 | Experimental Benchmark |
| **PR-AUC Score** | 0.9529 | 0.9457 | Experimental Benchmark |
| **Candidate Blocking Recall** | 78.1184% | 78.1184% *(Retained)* | **Verified Upstream Ceiling** |
| **End-to-End Validation Macro $F_{0.5}$** | **~0.760** | **0.8339** *(Tuned at $p=0.55$)* | **EXPERIMENTAL / NOT VERIFIED FOR PRODUCTION** |
| **Average Candidates / Query** | 281.73 | 281.73 | Verified Production Envelope |
| **Peak Memory Footprint** | ~1.8 GB | 1.8 GB RAM | Verified Production Envelope |
| **Official Submission Validator** | PASS | PASS | Verified Production Envelope |

---

### 2. Critical Methodological Audit Findings

#### Methodological Issue #1: Training Sample Confound
- **Finding**: The S++ challenger script trained using `gt_df['source1_entity_id'].values[:50000]`, which represents only **2.27% of the 2,206,821 total S1 training population**.
- **Impact**: The baseline model was trained across the full training population. Comparing a model trained on a 2.27% sample against the production model confounds feature count with sample size and non-identical training data.

#### Methodological Issue #2: Threshold Selection Bias
- **Finding**: The S++ threshold grid search ($0.30 \dots 0.70$) evaluated and selected optimal $p=0.55$ on the **exact same validation split** used to report Macro $F_{0.5} = 0.8339$.
- **Impact**: Evaluating $F_{0.5}$ on data used for threshold selection introduces optimistic selection bias. $0.8339$ is a validation-set benchmark, not an unbiased generalization score.

#### Methodological Issue #3: Threshold Metric Reconciliation
- **Reconciliation**:
  - At threshold $p=0.35$: Candidate Precision = **90.30%**, Recall = **87.55%**, F1 = **88.90%**, Macro $F_{0.5}$ = **0.8171**.
  - At threshold $p=0.50$: Candidate Precision = **92.90%**, Recall = **88.88%**, F1 = **90.84%**, Macro $F_{0.5}$ = **0.8317**.
  - At threshold $p=0.55$: Candidate Precision = **93.65%**, Recall = **88.09%**, F1 = **90.78%**, Macro $F_{0.5}$ = **0.8339**.

---

### 3. Failure Mode Analysis of Missed True Matches

Evaluated on `182,932` ground-truth true matching pairs:

- **Total Sample Missed True Pairs**: `39,900` (21.81% missed -> **78.19% Reachable Recall**)
- **Failure Categories**:
  1. **Candidate Ceiling Cap (350 Cap Pruning)**: `22,525` pairs (**56.45%** of missed matches). True candidate matched index tokens, but evidence ranking pruned candidate at 350 cap.
  2. **Zero Candidate Match (No Index Match / Transliteration)**: `12,477` pairs (**31.27%** of missed matches). Query and target shared 0 clean tokens of length $\ge 2$ due to transliteration or address variations.
  3. **Index Truncation / Token Filtering**: `4,898` pairs (**12.28%** of missed matches). Token posting list limits (1,500 token limit) dropped target.
  - *Denominator Check*: $22,525 + 12,477 + 4,898 = 39,900$ (100.00% exact match).

---

### 4. Production Baseline & V2 Artifact Integrity Check

| Production File | Expected SHA256 Checksum | Measured SHA256 Checksum | Verification Status |
| :--- | :--- | :--- | :---: |
| **`matching_results.tsv`** | `d1c898796e2ab192b627f8d0e8ea83f96e1427b029a8d66ed9165c87bd94d1e8` | `d1c898796e2ab192b627f8d0e8ea83f96e1427b029a8d66ed9165c87bd94d1e8` | **UNTOUCHED (LOCKED)** |
| **`team_submission.zip`** | `459e67da33e82ea6b052de382cbdf0805c6cbb09a5ca2138415fa6a7cd043747` | `459e67da33e82ea6b052de382cbdf0805c6cbb09a5ca2138415fa6a7cd043747` | **UNTOUCHED (LOCKED)** |
| **`arya_model.joblib`** | `476763ebe4051718167703dbf28e88a85ef80e2982876ccdb028508793534cad` | `476763ebe4051718167703dbf28e88a85ef80e2982876ccdb028508793534cad` | **UNTOUCHED (LOCKED)** |
| **`matching_results_v2.tsv`** | `56d88eb1f3b7f1e6dd23b3b626266b9cfb9b59a9257f9e9feedc6a6fc7a72d73` | `56d88eb1f3b7f1e6dd23b3b626266b9cfb9b59a9257f9e9feedc6a6fc7a72d73` | **VALIDATED** |
| **`team_submission_v2.zip`** | `d6926b559404028a9fdd2aca167438f0cead828fb4991ede0474ee7f8ad46d7c` | `d6926b559404028a9fdd2aca167438f0cead828fb4991ede0474ee7f8ad46d7c` | **BYTE-IDENTICAL (PASS)** |

---

### 5. Final Production Decision

**KEEP BASELINE / V2. DO NOT SUBMIT S++.**  
*Reason*: The S++ candidate model exhibits methodological confounds (training on a 2.27% sample subset and threshold selection bias). Production baseline (`team_submission.zip`) and V2 submission (`team_submission_v2.zip`) remain locked, verified, and protected.
