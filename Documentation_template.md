# ML Challenge 2026: Business Entity Resolution Solution Documentation

**Team Name:** Team Swapnil / Go1-GO2  
**Team Members:** Swapnil Patil (Lead ML Engineer), Go3 (Submission Lead)  
**Submission Date:** September 27, 2026  

---

## 1. Executive Summary
We present a high-precision, memory-safe, end-to-end Entity Resolution pipeline combining **Go1** (Phases 1-3: String Cleaning & Candidate Blocking) and **GO2** (Phase 4: ML Candidate Scoring & Dynamic Threshold Ranking). Our solution evaluates candidate pairs across 1.73 Million Source 1 test queries using a 16-feature `HistGradientBoostingClassifier`, achieving full streaming inference while maintaining a strict 1.8 GB RAM footprint and passing all official submission validation rules.

---

## 2. Dataset & Entity Structure

### 2.1 Entity Relationship Setup
- **Source 1 (`S1`)**: Query entity dataset (`train_source1.tsv`: 2,206,821 rows; `test_source1.tsv`: 1,732,544 rows).
- **Source 2 (`S2`) & Source 3 (`S3`)**: Reference target repositories against which `S1` entities are matched.
- **Ground Truth (`train_ground_truth.tsv`)**: Contains ground truth multi-target maps (`S1` → `S2`/`S3`) for model training and validation.
- **Geographic Coverage**: Three distinct geographic regions — **US**, **India**, and **France**.
- **France Dataset Limitation**: The competition dataset provides labelled training ground truth ONLY for `US` and `India`. `France` entities have zero labelled ground truth in the training set and therefore cannot be quantitatively validated.

---

## 3. Candidate Generation (Blocking)

### 3.1 Current Production Blocking Strategy
Our production blocking engine uses multi-pass inverted indexing strictly partitioned by country:
1. **Country Partitioning**: Hard constraint enforcing strict country matching (`US`, `India`, `France`).
2. **Exact Clean Name Index**: Matches entities sharing clean normalized business names.
3. **Significant 2-Token Name Index**: Inverted index on significant name tokens (max posting list = 1,500).
4. **Address Token Index**: Inverted index on street and city location tokens (max posting list = 500).
5. **Address Number Index**: Inverted index on street/unit numbers (max posting list = 500).
6. **4-Character Name Prefix Index**: Fallback prefix index for sparse candidate sets (max posting list = 500).
7. **Candidate Ceiling**: Strict ceiling of **350 candidates** per `S1` entity after lightweight evidence ranking.

### 3.2 Baseline Blocking Recall
Evaluated on 100% of Ground Truth training pairs (`7,638,365` true pairs across `2,206,821` `S1` entities):
- **Measured Blocking Recall**: **78.1184%** (`5,966,972` reachable / `7,638,365` true pairs).
  - **US Blocking Recall**: **76.9604%** (`3,523,649` / `4,578,522`)
  - **India Blocking Recall**: **79.8513%** (`2,443,323` / `3,059,843`)
  - **France Blocking Recall**: *Not quantitatively measurable (labelled ground truth unavailable).*
- **Candidate Volume**: `621,736,279` total pairs (Average: `281.73` candidates per `S1`, P95: `350`, Max: `350`).

### 3.3 Experimental Multi-Pass Blocking Audit
We conducted a controlled multi-pass blocking experiment to test whether recall could be expanded:
- **Experimental Blocking Recall**: **79.5882%** (Net recall gain of **+1.4698 percentage points**).
- **Candidate Volume Explosion**: Total candidate pairs expanded from 621.7M to **1.45 Billion** (**+133.25% candidate growth**).
- **Downstream Validation Performance**: When evaluated downstream using the trained classifier, the experimental candidate noise reduced validation $F_{0.5}$ from **0.760 to 0.748** (-0.0120 drop).
- **Conclusion**: **Experimental blocking REJECTED. Production blocking RETAINED.**

---

## 4. Machine Learning Model Architecture

### 4.1 Feature Engineering (16 Features)
The GO2 model consumes 16 tabular similarity and structural features computed between candidate pairs:
1. `name_jaccard_s2`: S1-S2 Token Jaccard Similarity on clean business names.
2. `name_jaccard_s3`: S1-S3 Token Jaccard Similarity on clean business names.
3. `name_prefix_match`: Boolean match of 4-character name prefixes.
4. `name_char_diff`: Absolute character length difference between name strings.
5. `name_length_ratio`: Ratio of shorter name string length to longer name string length.
6. `addr_jaccard_s2`: S1-S2 Token Jaccard Similarity on business addresses.
7. `addr_jaccard_s3`: S1-S3 Token Jaccard Similarity on business addresses.
8. `sig_tok_jaccard`: Jaccard similarity on significant name tokens.
9. `num_tok_overlap`: Count of exact matching numeric street/unit tokens.
10. `exact_country_match`: Boolean flag for country alignment.
11. `country_us`: One-hot flag for US region.
12. `country_india`: One-hot flag for India region.
13. `country_france`: One-hot flag for France region.
14. `name_exact_clean_match`: Exact equality of cleaned name strings.
15. `addr_num_exact_match`: Exact equality of extracted house/building numbers.
16. `candidate_rank`: Rank position of the candidate in the inverted index evidence ranking.

### 4.2 Supervised Model
- **Model**: `sklearn.ensemble.HistGradientBoostingClassifier`
- **Validation Metrics**: Precision: **~87.37%**, Recall: **~70.66%**, Validation $F_{0.5}$: **~0.760**.

---

## 5. Decision Threshold Policy

### 5.1 Evidence-Backed V2 Country Thresholds
Because $F_{0.5}$ penalizes false positives twice as heavily as false negatives, regional score thresholding was optimized:
- **United States (`US`)**: $p \ge 0.60$ (Quantitatively validated on US Ground Truth split).
- **India**: $p \ge 0.50$ (Quantitatively validated on India Ground Truth split; $F_{0.5} = 0.7451$ vs $0.7395$ at 0.35).
- **France**: $p \ge 0.35$ with fallback $p \ge 0.30$ (Preserved baseline fallback; not quantitatively validated due to zero training ground truth).

---

## 6. Official Submission Validation Results
- **Official Validator Output**: `PASS — no blocking issues found. Safe to submit.`
- **Required S1 Test Entities**: `1,732,544`
- **Output Rows**: `1,732,545` (1 header + 1,732,544 test data rows)
- **Duplicate S1 IDs**: 0
- **Malformed Rows**: 0

---

## 7. Known System Limitations
1. **Upstream Blocking Recall Limit**: Current candidate blocking captures ~78.12% of true matches. Increasing blocking recall via fuzzy passes proved counter-productive because candidate noise degraded downstream classifier precision and $F_{0.5}$.
2. **France Ground Truth Absence**: Lacking labelled ground truth for France in the training data, France thresholds could not be quantitatively optimized and rely on conservative baseline fallbacks.
