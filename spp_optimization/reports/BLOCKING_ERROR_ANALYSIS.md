# Phase 3 — Blocking Error Analysis Report

Date: 2026-09-27
Evaluated Population: Full Ground Truth Dataset (7,638,365 true pairs across 2,083,574 S1 queries)

## 1. Missed True-Pair Root Cause Breakdown
- **Total True Pairs**: 7,638,365
- **Total Missed True Pairs**: 1,671,393 (21.8816% of total GT)

| Failure Category | Missed True Pairs | % of Missed | % of Total GT | Primary Root Cause |
| :--- | :---: | :---: | :---: | :--- |
| **Candidate Cap Truncation** | 939,539 | 56.21% | 12.30% | Candidate list capped at 350 per query entity |
| **Zero Candidate Generation** | 527,343 | 31.55% | 6.90% | No token/n-gram key matched inverted index |
| **Sparse / Index Truncation** | 204,511 | 12.24% | 2.68% | Posting list length limit / token filtering |

## 2. Strategic Takeaways
1. **Candidate Cap Limit**: Accounts for **56.21%** of all missed true pairs. Increasing candidate caps or improving pre-reranking candidate quality is the largest single leverage point.
2. **Zero-Candidate Queries**: Accounts for **31.55%** of missed true pairs. Soft prefix/consonant-skeleton indexing is required to recover zero-candidate queries.
