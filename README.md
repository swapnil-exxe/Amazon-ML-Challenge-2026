<div align="center">

# 🛒 Amazon ML Challenge 2026: Business Entity Resolution
### High-Precision Multi-Source Entity Matching Pipeline for Open-World Catalogs

[![Competition](https://img.shields.io/badge/Platform-Unstop-blue?style=for-the-badge&logo=target)](https://unstop.com/competitions/1743604/round/1593683/play/code)
[![Score](https://img.shields.io/badge/Best%20Macro%20F0.5-0.864691-success?style=for-the-badge&logo=checkmarx)]()
[![Model](https://img.shields.io/badge/Model-LightGBM%20(16%20Features)-orange?style=for-the-badge&logo=lightgbm)]()
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-green?style=for-the-badge)]()

</div>

---

## 👥 Authorship & Team Attribution

| | |
|---|---|
| 👑 **Author & Lead Developer** | **Swapnil Patil** |
| 🏆 **Role** | Solo Developer — end-to-end design, data auditing, blocking architecture, ML modeling & submission pipeline |
| 👥 **Team Members** | **Swapnil Patil** · **Manthan Palkar** · **Sneha More** · **Arya Kadam** |

> 🎓 Participated in the **Amazon ML Challenge 2026** hosted on [Unstop](https://unstop.com/competitions/1743604/round/1593683/play/code) (25th September 2026 – 27th September 2026). Every phase of the engineering solution—from exploratory data analysis and token-based inverted index blocking to feature extraction, LightGBM classifier training, country threshold calibration, and validation—was architected and executed by **Swapnil Patil**.

---

## 📌 Problem Overview & Challenge Formulation

In commercial e-commerce platforms, business identity data originates from multiple disparate, uncoordinated third-party sources. Each source contributes partial, noisy, and conflicting fragments of information regarding real-world commercial entities with **no shared universal primary keys**:

- **Source 1 ($S_1$)**: The deduplicated, reference catalog of entities.
- **Source 2 ($S_2$) & Source 3 ($S_3$)**: Secondary partner and seller catalogs containing noisy, duplicate, or fragmented listings.

### Objective
For each record in Reference Source 1, identify all matching records from Source 2 and Source 3. An entity in Source 1 can match **zero (singleton)**, **one**, or **multiple** records across Sources 2 and 3.

```
       Source 1 Entity (S1-00042)
             /           \
            ▼             ▼
   Source 2 (S2-00193)    Source 3 (S3-00812)
   [Matches Real-World Business]
```

### Key Real-World Challenges & Noise Patterns
1. **Name Permutations & Legal Suffixes**: Abbreviations (`Corp.` vs `Corporation`, `Pvt` vs `Private`, `Ltd` vs `Limited`), DBA ("doing business as") trade names, ampersand substitutions (`&` vs `and`), word reordering, and OCR typos.
2. **Address Inconsistencies**: Street abbreviations (`Rd.` vs `Road`, `St.` vs `Street`), missing ZIP/PIN codes, landmark-based descriptions (*"Near SBI ATM, MG Road"*), transliteration variations, and municipal zone numbering.
3. **Open-World Country Distribution**:
   - **Training Set**: Businesses located in **United States (`US`)** and **India (`IN`)**.
   - **Test Set**: Contains an unseen third country label: **France (`FR`)**. Models must treat country as an open categorical set without hardcoding or naive binary filtering.
4. **Scale & Search Space Explosion**: Pairwise Cartesian comparison between millions of $S_1$ and $S_2 \cup S_3$ records exceeds $\mathcal{O}(10^{12})$ combinations, necessitating an ultra-scalable multi-stage blocking mechanism.

---

## 🎯 Evaluation Metric: Macro-Averaged $F_{0.5}$

Submissions are evaluated on **Macro-Averaged $F_{0.5}$**, a precision-weighted harmonic mean where precision is weighted **$2\times$ heavier** than recall:

$$\beta = 0.5 \implies F_{0.5} = \frac{(1 + 0.5^2) \times \text{Precision} \times \text{Recall}}{0.5^2 \times \text{Precision} + \text{Recall}} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

### Why $F_{0.5}$ (Precision-Heavy)?
In enterprise entity resolution, **false merges** (linking two distinct, unrelated businesses) pollute catalog integrity and cause catastrophic transaction errors. A false negative (missing a match) is vastly less damaging than an aggressive false positive.

### Singleton Dynamics
Singletons are reference entities ($S_1$) that have **zero matching records** in $S_2$ or $S_3$:
- Predicting an empty list `""` for a true singleton awards **$1.0$**.
- Predicting any incorrect match for a singleton immediately collapses that entity's score to **$0.0$**.
- Our calibrated thresholds protect singletons from false-positive over-matching.

---

## 🏗️ End-to-End System Architecture

```
                                  DATA INGESTION (Unstop Dataset)
                   ┌───────────────────────────────────────────────────────────┐
                   │  train_source1.tsv, train_source2.tsv, train_source3.tsv  │
                   │  test_source1.tsv,  test_source2.tsv,  test_source3.tsv   │
                   └─────────────────────────────┬─────────────────────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: DATA AUDIT & STRUCTURAL INTEGRITY (`go1_output/phase_1_audit/`)                   │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│ • Strict TSV formatting validation (tab separators with commas inside addresses/names)       │
│ • Null value profiling, prefix verification (S1-, S2-, S3-), and duplicate ID audit         │
│ • Country label frequency extraction (US, India, open-set France)                           │
│ • Artifact hashing & SHA256 integrity baseline verification                                 │
└────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: CANONICAL NORMALIZATION & CLEANING (`go1_output/phase_2_cleaning/`)                │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│ • Legal Entity Suffix Stripping: regex dictionary for LLC, Ltd, Corp, Inc, Pvt, GmbH, SA, AG │
│ • Unicode Normalization: NFKD decomposition, accented character stripping, case flattening   │
│ • Signature Name Derivation: standardized canonical token sequences                         │
│ • Address Standardization: abbreviation mapping (St -> street, Rd -> road, Ave -> avenue)    │
│ • Token Frequency Map: corpus-wide inverse token indexing for selective stopword pruning    │
└────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: CANDIDATE GENERATION & INVERTED INDEX BLOCKING (`go1_output/phase_3_blocking/`)   │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│ • Country-partitioned inverted token index built over Target Catalogs ($S_2 \cup S_3$)       │
│ • Evidence Scoring Heuristic:                                                               │
│     Score(u, v) = 3.0 * (Name_Tokens) + 1.5 * (Addr_Tokens) + 2.0 * (Num_Tokens) + 10.0 * (Exact) │
│ • Multi-tier Pruning: Top-350 candidates retained per Source 1 reference entity             │
│ • Candidate Pairs Output: `candidate_pairs.tsv` (~78.20% blocking recall ceiling)           │
└────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: 16-FEATURE EXTRACTION & LIGHTGBM SCORING (`go2_output/`)                           │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│ Vectorized extraction of 16 granular pairwise signals:                                     │
│ [Name Jaccard, Signature Jaccard, Prefix-4, Address Jaccard, Exact Match Flags, Rank, etc.] │
│                                                                                             │
│                         ▼ LightGBM Classifier (`go2_model.joblib`)                          │
│                                                                                             │
│ Dynamic Threshold Calibration Engine:                                                       │
│   • United States:  P(Match) >= 0.60                                                        │
│   • India:          P(Match) >= 0.50                                                        │
│   • France / Other: P(Match) >= 0.35                                                        │
│   • Unknown:        P(Match) >= 0.30                                                        │
│   • Exact Canonical Override: Identical normalized name -> Guaranteed Link                  │
└────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                         │
                                         ▼
                               FINAL OUTPUT ARTIFACT
                     ┌────────────────────────────────────────┐
                     │   go2_output/matching_results_v2.tsv   │
                     │    Macro F0.5 = 0.864691 (VERIFIED)    │
                     └────────────────────────────────────────┘
```

---

## 📊 Performance Benchmark & Experiment Matrix

All experiments were evaluated on a controlled 20,000-entity stratified validation split representing identical distributions of singletons, US, and India records:

| Run ID | Blocking Strategy | Classifier | Threshold Rules | Candidate Recall | Precision | Recall | Macro $F_{0.5}$ | Status |
|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | Inverted Index (Cap 200) | LightGBM | Global 0.50 | 74.10% | 89.20% | 63.40% | 0.864000 | Reference |
| **EXP-B2** | Pre-ranked Index (Cap 350) | LightGBM | Global 0.50 | 78.20% | 90.15% | 66.10% | 0.864210 | Candidate boost |
| **EXP-V2 (Ours)** | **Pre-ranked Index (Cap 350)** | **LightGBM** | **Country Grid + Exact Override** | **78.20%** | **92.83%** | **67.67%** | **0.864691** | 🏆 **Best Submission** |
| **EXP-EMG** | Fast Index (Cap 300) | LightGBM | Aggressive Low 0.35 | 76.50% | 84.10% | 71.20% | 0.858811 | Rejected (Over-merging) |

---

## 🧬 Feature Engineering (16 Discriminative Signals)

Each pairwise candidate $(S_1, S_k)$ generated in Phase 3 is mapped to a 16-dimensional float vector:

| # | Feature Name | Computation Formula / Semantics | Value Range |
|---|---|---|:---:|
| 1 | `exact_name_match` | $\mathbb{I}[\text{raw\_name}_1 = \text{raw\_name}_2]$ | $\{0, 1\}$ |
| 2 | `exact_sig_name_match` | $\mathbb{I}[\text{sig\_name}_1 = \text{sig\_name}_2]$ (Suffix-stripped) | $\{0, 1\}$ |
| 3 | `name_jaccard` | $\frac{\|T(name_1) \cap T(name_2)\|}{\|T(name_1) \cup T(name_2)\|}$ | $[0.0, 1.0]$ |
| 4 | `sig_name_jaccard` | Jaccard similarity over canonical legal-stripped tokens | $[0.0, 1.0]$ |
| 5 | `abs_name_len_diff` | $\| \text{len}(name_1) - \text{len}(name_2) \|$ | $[0, \infty)$ |
| 6 | `prefix_4_match` | $\mathbb{I}[name_1[:4] = name_2[:4]]$ | $\{0, 1\}$ |
| 7 | `exact_addr_match` | $\mathbb{I}[\text{clean\_addr}_1 = \text{clean\_addr}_2]$ | $\{0, 1\}$ |
| 8 | `addr_jaccard` | Token Jaccard overlap of address strings | $[0.0, 1.0]$ |
| 9 | `addr_missing` | $\mathbb{I}[\text{addr}_1 \text{ is null} \lor \text{addr}_2 \text{ is null}]$ | $\{0, 1\}$ |
| 10 | `exact_num_match` | Exact equality of extracted address/phone number tokens | $\{0, 1\}$ |
| 11 | `num_jaccard` | Jaccard overlap of isolated numeric sequence tokens | $[0.0, 1.0]$ |
| 12 | `abs_addr_len_diff`| Absolute character count discrepancy between addresses | $[0, \infty)$ |
| 13 | `country_match` | $\mathbb{I}[\text{country}_1 = \text{country}_2]$ | $\{0, 1\}$ |
| 14 | `is_source3` | Indicator flag whether target candidate originates from $S_3$ | $\{0, 1\}$ |
| 15 | `candidate_rank` | Rank position assigned during Phase 3 evidence sorting | $[1, 350]$ |
| 16 | `total_candidates` | Cardinality of candidate pool produced for query $S_1$ | $[1, 350]$ |

---

## 🗃️ Complete Repository & File Map

```
Amazon-ML-Challenge-2026/
├── README.md                                  # ⭐ Primary comprehensive documentation & guide
├── requirements.txt                           # Pinned Python dependencies
├── DATASET_SETUP.md                           # Dataset download & verification instructions
├── blocking_recall_audit_v2.py                # Standalone blocking recall metric evaluator
├── blocking_recall_experiment.py              # Candidate cap exploration script
├── run_spp_blocking_failure_analysis.py       # Systematic false-negative diagnostic tool
├── run_spp_challenger_experiments.py          # Alternative model testing benchmark
├── run_spp_threshold_grid.py                  # Grid search optimization for country cuts
│
├── go1_output/                                # PHASE 1, 2, & 3 PIPELINES (Audit, Clean, Block)
│   ├── GO1_FINAL_AUDIT.md                     # Comprehensive audit record of data & recall
│   ├── phase_1_audit/
│   │   ├── audit_dataset.py                   # Automated missingness and format validator
│   │   └── reports/                           # Profiling tables (countries, nulls, duplicates)
│   ├── phase_2_cleaning/
│   │   ├── src/normalize.py                   # Canonical regex cleaning & legal suffix removal
│   │   └── reports/                           # Transformation stats & normalization diffs
│   └── phase_3_blocking/
│       ├── src/
│       │   ├── blocking_keys.py               # Token extraction & index key generator
│       │   ├── build_indexes.py               # High-throughput inverted index builder
│       │   ├── generate_candidates.py         # Multi-field evidence ranking & candidate pruning
│       │   ├── run_go1_phase3.py              # Main blocking workflow execution script
│       │   └── run_v2_blocking_pipeline.py    # Production blocking pipeline (V2)
│       ├── reports/                           # Token frequency & reduction ratio logs
│       └── logs/                              # Execution logs for blocking runs
│
├── go2_output/                                # PHASE 4: INFERENCE ENGINE & SUBMISSION
│   ├── go2_model.joblib                       # ⭐ TRAINED LIGHTGBM CLASSIFIER (0.23 MB)
│   ├── run_go2_inference_v2.py                # ⭐ PRODUCTION INFERENCE ENGINE (16 feats + thresholds)
│   ├── matching_results_v2.tsv                # ⭐ BEST SUBMISSION FILE (Macro F0.5 = 0.864691)
│   ├── matching_results.tsv                   # Baseline reference submission file
│   ├── go2_final_summary.json                 # Checkpoint parameters & execution telemetry
│   ├── go2_inference_v2.log                   # Full log transcript of test set inference
│   └── spp/                                   # S++ 20-feature experimental variants
│
├── spp_optimization/                          # REPRODUCIBILITY & VALIDATION SUITE
│   ├── experiments/
│   │   ├── dev_ground_truth.tsv               # Stratified validation ground truth (97 MB)
│   │   ├── val_ground_truth.tsv               # Compact validation ground truth (24 MB)
│   │   ├── emergency_fast_optimizer.py        # Rapid grid explorer
│   │   ├── final_submission_mode.py           # Submission generator with safety assertions
│   │   └── baseline_streaming/
│   │       ├── evaluate_baseline_streaming.py # Chunked streaming validator (O(1) RAM)
│   │       └── phase2_error_ceiling.py        # Attribution analysis of precision/recall errors
│   ├── metrics/                               # Experiment CSV tracking matrices
│   └── reports/                               # Detailed technical scorecards
│
└── student_resource/                          # COMPETITION SPECS & VALIDATOR
    ├── Documentation_template.md              # Official methodology report template
    ├── README.md                              # Official competition overview
    └── utils/
        └── validate_submission.py             # Official format verification script
```

---

## 📦 Dataset Acquisition & Setup

Competition dataset files are governed by Amazon/Unstop competition rules and can be downloaded from the portal:

🔗 **[Amazon ML Challenge Portal — Dataset Download](https://unstop.com/competitions/1743604/round/1593683/play/code)**

### Placement Directory Layout
Extract raw TSVs into `student_resource/dataset/`:

```
student_resource/dataset/
├── train/
│   ├── train_source1.tsv           # S1 reference records (Deduplicated)
│   ├── train_source2.tsv           # S2 partner catalog records
│   ├── train_source3.tsv           # S3 partner catalog records
│   └── train_ground_truth.tsv      # Gold-standard entity links
└── test/
    ├── test_source1.tsv            # S1 query records (US, India, France)
    ├── test_source2.tsv            # S2 test catalog
    └── test_source3.tsv            # S3 test catalog
```

---

## 🚀 Step-by-Step Execution Guide

### 1. Environment Setup
```bash
git clone https://github.com/swapnil-exxe/Amazon-ML-Challenge-2026.git
cd Amazon-ML-Challenge-2026
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Phase 1 — Data Audit
Inspect data schema, detect formatting anomalies, verify country labels:
```bash
python go1_output/phase_1_audit/audit_dataset.py
```

### 3. Phase 2 — Data Normalization & Cleaning
Generate canonical stripped and normalized text tables:
```bash
python go1_output/phase_2_cleaning/src/normalize.py
```

### 4. Phase 3 — Candidate Blocking & Pre-Ranking
Build country-partitioned inverted index and retrieve Top-350 candidates:
```bash
python go1_output/phase_3_blocking/src/run_v2_blocking_pipeline.py
```

### 5. Phase 4 — Pairwise Feature Extraction & Inference
Run LightGBM scoring, apply country thresholds, and generate final TSV matches:
```bash
python go2_output/run_go2_inference_v2.py
```

### 6. Verification Against Official Submission Criteria
Execute the Unstop official validator to guarantee zero formatting violations:
```bash
python student_resource/utils/validate_submission.py \
  --matching go2_output/matching_results_v2.tsv \
  --candidate go1_output/phase_3_blocking/candidates/candidate_pairs_test.tsv \
  --test-dir student_resource/dataset/test
```
*(Expected output: `PASS (exit 0) - All formatting rules satisfied`)*

---

## 🔒 Artifact Integrity & SHA-256 Checksums

| Target Artifact | Rel Path | SHA-256 Checksum |
|---|---|---|
| **Best Model** | `go2_output/go2_model.joblib` | `476763ebe4051718167703dbf28e88a85ef80e2982876ccdb028508793534cad` |
| **Best Submission** | `go2_output/matching_results_v2.tsv` | `56d88eb1f3b7f1e6dd23b3b626266b9cfb9b59a9257f9e9feedc6a6fc7a72d73` |
| **Baseline Matches** | `go2_output/matching_results.tsv` | `d1c898796e2ab192b627f8d0e8ea83f96e1427b029a8d66ed9165c87bd94d1e8` |

---

## ⚖️ Academic Integrity & Competition Compliance

- **No External Lookups**: In compliance with competition rules, zero external web scraping, external phone/address APIs, or proprietary databases were employed.
- **Model Size Limits**: The LightGBM classifier contains fewer than **500 trees** (0.23 MB on disk), well below the competition ceiling of 8 Billion parameters.
- **Open Source Licensing**: All libraries utilized (LightGBM, scikit-learn, NumPy, Pandas) operate under permissive BSD/MIT/Apache 2.0 licenses.

---

<div align="center">

**Built with ❤️ by Swapnil Patil**  
*Team Members: Swapnil Patil · Manthan Palkar · Sneha More · Arya Kadam*  
*Amazon ML Challenge 2026 — Business Entity Resolution*

</div>
