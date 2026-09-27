# ML Challenge 2026 — Final Optimization & Decision Gate Report

## Executive Summary
This report summarizes the time-boxed, evidence-gated diagnostic audit conducted prior to the 11:59 PM IST deadline. A complete, verified, and officially validated submission (`matching_results.tsv` / `team_submission.zip`) is already completed and uploaded as a safety baseline.

---

## Phase 1 — Diagnostic Audit Findings

- **Elapsed Time**: 77.49 seconds (well within the 30-minute budget).

### 1. Candidate Blocking Recall
- **Overall Recall**: **78.16%** (134,999 / 172,731 true ground-truth pairs in sample)
- **US Candidate Recall**: **77.02%** (79,355 / 103,037 true pairs)
- **India Candidate Recall**: **79.84%** (55,644 / 69,694 true pairs)

### 2. Cached Probabilities on Disk
- **Finding**: **NO REUSABLE PRODUCTION PROBABILITY CACHE FOUND ON DISK.**
- **Impact**: Any threshold modification cannot be executed as a "cheap" multi-minute operation. It requires a full 2.72-hour re-inference run over all 403.5 Million candidate pairs.

### 3. Model Feature Definitions (16 Features)
- **Classifier**: `sklearn.ensemble.HistGradientBoostingClassifier`
- **Features (16)**: Exact Name Match, Sig Name Exact Match, Name Token Jaccard, Sig Name Token Jaccard, Name Length Diff, Prefix-4 Match, Address Exact Match, Address Token Jaccard, S2 Address Empty Flag, Number Exact Match, Number Token Jaccard, Address Length Diff, Country Exact Match, Is-S3 Candidate Flag, Candidate Blocking Rank, Total Candidates for S1.
- **Missing Features**: Model-probability-based rank within S1 and score-gap-to-next-best candidate are absent in the feature vector.

### 4. Country-Specific Threshold Sweeps
- **US Optimal Threshold**: $p = 0.60$ (Macro $F_{0.5} = 0.7968$)
- **India Optimal Threshold**: $p = 0.50$ (Macro $F_{0.5} = 0.7451$)
- **France Status**: **0 France entities exist in labelled training data.** France thresholds are unvalidated.

### 5. Probability Calibration
- Probabilities are monotonic and well-calibrated (Bin $[0.8, 1.0]$ has empirical match rate of $97.05\%$).

---

## Phase 2 — Decision Gate Evaluation

1. **Evidence-Backed Room for Improvement**: Validation sweeps indicate potential $F_{0.5}$ gains on US/India at higher thresholds ($0.50 - 0.60$).
2. **Execution Cost**: **EXPENSIVE (2.72 hours)**. No cached per-pair test probabilities exist on disk.
3. **Risk Analysis**:
   - Re-running 2.72-hour inference on deadline day carries unnecessary execution and system risk.
   - France test entities (259,452 entities) are unvalidated in training data; increasing thresholds to $0.50-0.60$ risks dropping valid France matches and causing severe recall loss.
   - The current production pipeline ($p \ge 0.35 / p \ge 0.30$) achieves high Precision (**87.37%**) and solid Macro $F_{0.5}$ (**0.7599**).

### **DECISION GATE CONCLUSION: STOP HERE AND KEEP EXISTING SUBMISSION AS FINAL.**

---

## Final Recommendation & Instruction

**RECOMMENDATION**: **Submit the original (already done) as final — no changes justified/safe today.**

- **Primary Submission File**: `/Users/swapnil/Documents/ML/go2_output/matching_results.tsv`
- **Final Package Archive**: `/Users/swapnil/Documents/ML/team_submission.zip`
- **Official Validator Status**: **PASS — no blocking issues found. Safe to submit.**
- **Deadline Buffer**: ~12 hours remaining before 11:59 PM IST.

Keeping the validated original baseline is the safe, professional, and correct choice.
