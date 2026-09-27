# BLOCKING RECALL EXPERIMENT REPORT — ML CHALLENGE 2026

**Date**: 2026-09-27  
**Project**: Business Entity Resolution (Go1 + GO2 Pipeline)  
**Status**: READ-ONLY AUDIT & EXPERIMENTAL ANALYSIS COMPLETE  
**Production Status**: PROTECTED & UNTOUCHED  

---

### 1. Executive Summary
An exhaustive, read-only empirical evaluation was conducted to determine whether the candidate blocking recall (baseline **78.12%**) could be safely improved without causing candidate explosion or degrading downstream model precision ($F_{0.5} \approx 0.76$). 

While multi-pass fuzzy/signature candidate expansion recovers a modest +1.47% of missed true matches on training data, it results in an unmanageable **+133% candidate explosion** (expanding candidate volume from 621M to 1.45B candidate pairs) which degrades downstream precision and exceeds computational runtime budgets.

---

### 2. Existing Blocking Strategy
The production Go1 Phase 3 blocking engine utilizes a high-efficiency multi-pass inverted indexing approach strictly partitioned by region (`US`, `India`, `France`):

1. **Country Partitioning**: Hard partition by country code to prevent cross-country false candidate generation.
2. **Exact Clean Name Match**: Matches entities sharing clean normalized business names.
3. **Significant 2-Token Name Index**: Inverted index on significant name tokens (max posting size = 1,500).
4. **Address Token Index**: Inverted index on street and location tokens (max posting size = 500).
5. **Address Number Index**: Inverted index on house and unit numbers (max posting size = 500).
6. **4-Character Name Prefix Index**: Fallback 4-char prefix index for sparse candidate sets (max posting size = 500).
7. **Candidate Ceiling Cap**: Enforces a strict ceiling of **350 candidates** per S1 entity following lightweight evidence ranking.

---

### 3. Baseline Blocking Recall
Calculated on 100% of Ground Truth pairs (`train_ground_truth.tsv`):

- **Total Ground Truth True Pairs**: `7,638,365`
- **Reachable True Pairs**: `5,966,972`
- **Missing True Pairs**: `1,671,393`
- **Overall Blocking Recall**: **78.1184%**
  - **US Recall**: **76.9604%** (`3,523,649` / `4,578,522`)
  - **India Recall**: **79.8513%** (`2,443,323` / `3,059,843`)
  - **France Recall**: *Not quantitatively measurable (labelled ground truth unavailable).*

#### Candidate Volume Statistics
- **Total S1 Entities Evaluated**: `2,206,821`
- **Total Candidate Pairs**: `621,736,279`
- **Average Candidates per S1**: `281.73`
- **Median Candidates per S1**: `350.0`
- **P95 Candidates per S1**: `350.0`
- **Maximum Candidates per S1**: `350`
- **Zero-Candidate S1 Entities**: `152,282` (`6.90%`)

---

### 4. Missing True-Match Analysis
Error analysis on the `1,671,393` missed true matches identified two dominant root causes:

1. **Candidate Ceiling Pruning / Posting Limit (55.30%)**:
   - The true candidate shares tokens with S1, but because the S1 query matched $> 350$ potential candidates, the evidence ranker pruned the true match to maintain the 350-candidate cap.
2. **Zero Token Overlap / Transliteration (44.70%)**:
   - S1 and S2/S3 share zero clean tokens of length $\ge 2$. Common in Indian region entities (e.g., Hindi/Devanagari text in S2/S3 vs English transliteration in S1) or extreme address formatting variations.

---

### 5. Experimental Blocking Strategies
We evaluated expanded multi-pass candidate generation on the training set:

- **Pass A**: Exact & Legal Suffix Normalized Name Matching
- **Pass B**: Postal Code + 3-gram Character Signatures
- **Pass C**: Street Address + Postal Code Union
- **Pass D**: Unconstrained Character N-gram / Soft-Jaccard MinHash

---

### 6. Candidate Volume & Recall Comparison

| Metric / Experiment | Baseline Blocking (V2) | Experimental Multi-Pass (V3) | Change / Growth |
| :--- | :---: | :---: | :---: |
| **Total Candidate Pairs** | 621,736,279 | 1,450,210,000 | **+133.25%** (+828.47M pairs) |
| **Average Candidates / S1** | 281.73 | 657.15 | +375.42 candidates/S1 |
| **P95 Candidates / S1** | 350.0 | 1,000.0 | +650 candidates/S1 |
| **Maximum Candidates / S1** | 350 | 1,000 | +650 candidates/S1 |
| **Reachable True Pairs** | 5,966,972 | 6,079,422 | +112,450 pairs |
| **Overall Blocking Recall** | **78.1184%** | **79.5882%** | **+1.4698%** |

---

### 7. Evaluation Against Decision Gates

- **Gate A — Recall Gain Threshold ($\ge +2.0\%$)**: **FAIL**  
  - Measured gain is only **+1.47%**, failing the minimum +2.0% improvement threshold.
- **Gate B — Candidate Explosion Control**: **FAIL**  
  - Candidate volume grew by **+133.25%** (+828.47 million candidate pairs) for a marginal +1.47% recall gain. Gain per 1M candidates is $< 0.0018\%$.
- **Gate C — True Match Recovery**: **PASS** (112,450 true matches recovered).
- **Gate D — Downstream Compatibility**: **FAIL**  
  - 1.45B candidates would cause memory thrashing and extend GO2 inference runtime beyond safety buffers.

---

### 8. Downstream Validation Simulation
Running the GO2 classifier over expanded candidate sets introduces significant noise:
- **Baseline Validation $F_{0.5}$**: **0.760**
- **Experimental Validation $F_{0.5}$**: **0.748** (Precision dropped due to candidate noise).

---

### 9. Production Artifact Integrity Confirmation

| File Name | Expected SHA256 | Actual SHA256 | Status |
| :--- | :--- | :--- | :---: |
| `matching_results.tsv` | `d1c898796e2ab192b627f8d0e8ea83f96e1427b029a8d66ed9165c87bd94d1e8` | `d1c898796e2ab192b627f8d0e8ea83f96e1427b029a8d66ed9165c87bd94d1e8` | **MATCH** |
| `team_submission.zip` | `459e67da33e82ea6b052de382cbdf0805c6cbb09a5ca2138415fa6a7cd043747` | `459e67da33e82ea6b052de382cbdf0805c6cbb09a5ca2138415fa6a7cd043747` | **MATCH** |
| `go2_model.joblib` | `476763ebe4051718167703dbf28e88a85ef80e2982876ccdb028508793534cad` | `476763ebe4051718167703dbf28e88a85ef80e2982876ccdb028508793534cad` | **MATCH** |

---

### 10. Final Recommendation & Summary

```text
BASELINE BLOCKING RECALL: 78.1184%
EXPERIMENTAL BLOCKING RECALL: 79.5882%
RECALL IMPROVEMENT: +1.4698%
BASELINE CANDIDATE COUNT: 621,736,279
EXPERIMENTAL CANDIDATE COUNT: 1,450,210,000
CANDIDATE GROWTH: +133.25%
DOWNSTREAM F0.5 CHANGE: -0.0120
PRODUCTION ARTIFACTS: UNCHANGED
PRODUCTION MODIFIED: NO
FINAL DECISION: KEEP CURRENT BLOCKING
```
