# SUPERSEDED — EXPERIMENTAL RESULTS NOT VALIDATED FOR PRODUCTION

> [!CAUTION]
> **SUPERSEDED NOTICE**: The contents of this report represent historical experimental observations from an evaluation track with known methodological confounds (training on a 2.27% sample subset, in-sample threshold selection bias, and split non-equivalence). **These results MUST NOT be interpreted as validated production improvements or estimates of unseen generalization performance.** The production pipeline remains LOCKED on Baseline / V2.

---

# S++ OPTIMIZATION REPORT (HISTORICAL EXPERIMENTAL ARCHIVE)

**Project**: Business Entity Resolution (Manthan + ARYA Pipeline)  
**Date**: September 27, 2026  
**Methodological Status**: **SUPERSEDED / EXPERIMENTAL ONLY — NOT APPROVED FOR PRODUCTION**  

---

### 1. Production Baseline vs Experimental S++ Challenger (Historical Observation)

| Metric / Parameter | Production Baseline | Experimental Challenger Benchmark | Methodological Status |
| :--- | :---: | :---: | :--- |
| **Candidate Pair Precision** | ~91.65% | 93.65% *(at $p=0.55$)* | Experimental Benchmark (In-Sample) |
| **Candidate Pair Recall** | ~88.47% | 88.09% *(at $p=0.55$)* | Experimental Benchmark (In-Sample) |
| **Candidate Pair F1** | ~90.04% | 90.78% *(at $p=0.55$)* | Experimental Benchmark (In-Sample) |
| **ROC-AUC Score** | 0.9969 | 0.9963 | Experimental Benchmark |
| **PR-AUC Score** | 0.9529 | 0.9457 | Experimental Benchmark |
| **Candidate Blocking Recall** | 78.1184% | 78.1184% *(Retained)* | **Verified Upstream Ceiling** |
| **End-to-End Validation Macro $F_{0.5}$** | **~0.760** | **0.8339** *(Tuned at $p=0.55$)* | **EXPERIMENTAL / NOT VERIFIED FOR PRODUCTION** |

---

### 2. Methodological Issues Documented

1. **Training Population Confound**: Trained on `train_s1_set = set(gt_df['source1_entity_id'].values[:50000])`, representing only 2.27% of the total 2,206,821 S1 training population.
2. **Threshold Selection Bias**: Evaluated and selected optimal threshold $p=0.55$ on the same validation split used to report $0.8339$.
3. **Metric Reconciliation**:
   - At $p=0.35$: Precision = 90.30%, Recall = 87.55%, F1 = 88.90%, Macro $F_{0.5}$ = 0.8171.
   - At $p=0.50$: Precision = 92.90%, Recall = 88.88%, F1 = 90.84%, Macro $F_{0.5}$ = 0.8317.
   - At $p=0.55$: Precision = 93.65%, Recall = 88.09%, F1 = 90.78%, Macro $F_{0.5}$ = 0.8339.

---

### 3. Failure Mode Breakdown of Missed True Matches

Evaluated on `182,932` ground-truth true matching pairs:
- **Sample Missed True Pairs**: `39,900` (21.81% missed -> **78.19% Reachable Recall**)
  1. **Candidate Cap (350 Cap Pruning)**: `22,525` pairs (**56.45%** of missed matches).
  2. **Zero Candidates (No Index Match / Transliteration)**: `12,477` pairs (**31.27%** of missed matches).
  3. **Index Truncation / Token Filtering**: `4,898` pairs (**12.28%** of missed matches).

---

### 4. Production Safety & Final Decision

**KEEP BASELINE / V2. DO NOT SUBMIT S++.**  
Original production baseline artifacts (`matching_results.tsv`, `team_submission.zip`, `arya_model.joblib`) and V2 artifacts (`matching_results_v2.tsv`, `team_submission_v2.zip`) remain locked, verified, and protected.
