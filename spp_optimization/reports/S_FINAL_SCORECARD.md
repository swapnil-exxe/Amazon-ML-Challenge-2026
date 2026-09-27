# OFFICIAL S++ FINAL SCORECARD — ML CHALLENGE 2026

**Project:** Business Entity Resolution — ML Challenge 2026  
**Date:** September 27, 2026  
**Evaluation:** V2 Blocking Recovery & Pre-Ranking Optimization  
**Final Status:** **S++ NOT REACHED** (Macro F0.5 = 0.8240 vs S++ Target 0.85)

---

## 1. Official S++ Metric Scorecard

| Metric | Protected V2 Baseline | Measured Optimized System | Required S++ Target | Mechanical Status | Measurement Method |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Blocking Recall** | 78.1184% | 91.50% | $\ge 95\%$ | **FAIL** | Full Ground Truth Evaluation |
| **Candidate Precision** | 96.00% | 96.50% | $\ge 95\%$ | **PASS** | Candidate Pool Analysis |
| **Candidate Recall** | 78.1184% | 91.50% | $\ge 92\%$ | **FAIL** | Full Ground Truth Evaluation |
| **Candidate F1** | 86.13% | 93.90% | $\ge 93\%$ | **PASS** | Full Ground Truth Evaluation |
| **ROC-AUC** | 0.988 | 0.992 | $\ge 0.999$ | **FAIL** | Classification Metric Run |
| **PR-AUC** | 0.956 | 0.971 | $\ge 0.98$ | **FAIL** | Classification Metric Run |
| **Recall@1** | 86.20% | 93.40% | $\ge 97\%$ | **FAIL** | Candidate Ranking Metric |
| **MRR** | 0.912 | 0.954 | $\ge 0.98$ | **FAIL** | Candidate Ranking Metric |
| **End-to-End Precision** | 86.15% | 88.40% | $\ge 92\%$ | **FAIL** | Test Validation Run |
| **End-to-End Recall** | 57.30% | 71.20% | $\ge 80\%$ | **FAIL** | Test Validation Run |
| **Macro $F_{0.5}$** | **0.7854** | **0.8240** | $\ge 0.85$ | **FAIL** | Test Validation Run |
| **Candidates / Query** | 281.73 | 312.40 | $\le 350\text{--}400$ | **PASS** | Measured Average / Query |
| **RAM** | ~1.3 GB | ~1.4 GB | $\le 1.8\text{ GB}$ | **PASS** | Peak Memory Profiling |
| **Runtime** | ~0.52 h | ~0.58 h | $\le 2.5\text{ h}$ | **PASS** | Total Test Suite Runtime |
| **Throughput** | ~980/s | ~910/s | $\ge 700\text{--}800\text{/s}$ | **PASS** | Entities Processed / Sec |
| **Data Loss** | 0% | 0% | $0\%$ | **PASS** | 1,732,544 rows matched |
| **Official Validator** | **PASS** | **PASS** | **PASS** | **PASS** | Official `validate_submission.py` |

---

## 2. Final Authorization & Directive
Because 7 metric requirements (including Macro $F_{0.5} = 0.8240$ vs $\ge 0.85$ target) fall short of the required S++ threshold, the status is mechanically set to **`S++ NOT REACHED`**.

Per non-negotiable safety rules:
1. No S++ ZIP package (`team_submission_spp.zip`) is generated.
2. Protected production artifacts (`team_submission_v2.zip`, `team_submission.zip`, `arya_model.joblib`, `matching_results_v2.tsv`) remain 100% untouched and cryptographically verified.
3. Recommended submission file remains **`/Users/swapnil/Documents/ML/team_submission_v2.zip`**.
