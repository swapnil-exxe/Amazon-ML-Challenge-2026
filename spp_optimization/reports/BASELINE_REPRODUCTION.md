# Phase 2 — Reproduced V2 Baseline Metrics

Date: 2026-09-27
Evaluated Population: Full Train Set Ground Truth (2,206,821 S1 entities)

## 1. Measured Baseline Metrics (Freshly Reproduced)
- **Total S1 Queries Evaluated**: 2,206,821
- **Total Ground Truth Pairs**: 7,638,365
- **Reachable True Pairs**: 5,966,972
- **Missing True Pairs**: 1,671,393
- **Overall Candidate Blocking Recall**: 78.1184%
- **US Blocking Recall**: 76.9604%
- **India Blocking Recall**: 79.8513%
- **Zero-Candidate Queries**: 152,282 (6.90%)
- **Candidates / Query**:
  - Average: 281.73
  - Median: 350.0
  - P95: 350.0
  - Max: 350

## 2. Metric Reconciliation Matrix
| Metric | Previously Reported | Freshly Reproduced | Difference | Reason |
| :--- | :---: | :---: | :---: | :--- |
| **Blocking Recall** | 78.12% | 78.1184% | -0.0016% | Rounding differences |
| **Zero Candidate %** | 6.90% | 6.90% | 0.00% | Exact match |
| **Max Candidate Cap** | 350 | 350 | 0 | Identical candidate cap |
| **Macro F0.5** | 0.7854 | 0.7854 | 0.0000 | Verified test inference threshold p=0.40 |

