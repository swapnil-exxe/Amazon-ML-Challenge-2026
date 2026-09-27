<div align="center">

# 🛒 Amazon ML Challenge 2026
## Business Entity Resolution — Team Submission

[![Competition](https://img.shields.io/badge/Platform-Unstop-blue?style=for-the-badge)](https://unstop.com/competitions/1743604/round/1593683/play/code)
[![Score](https://img.shields.io/badge/Best%20Score-0.8647%20Macro%20F0.5-brightgreen?style=for-the-badge)]()
[![Model](https://img.shields.io/badge/Model-LightGBM-orange?style=for-the-badge)]()
[![Python](https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge)]()

> **Task**: Given two large business entity catalogs (Source 1 & Source 2), identify which records refer to the same real-world business — even when names, addresses, and other fields are noisy, abbreviated, or missing.

</div>

---

## 👤 Author

| | |
|---|---|
| 👑 **Built by** | **Swapnil Patil** |
| 🏆 **Role** | Solo Developer — end-to-end design, engineering & ML |

> 🎓 Participated in the **Amazon ML Challenge 2026** hosted on [Unstop](https://unstop.com/competitions/1743604/round/1593683/play/code) — a 72-hour national-level hackathon. Every phase of the pipeline — data audit, cleaning, blocking, feature engineering, model training, threshold tuning, and submission — was designed and implemented by **Swapnil Patil**.

**Team Members:** Swapnil Patil · Manthan Palkar · Sneha More · Arya Kadam

---

## 🏆 Results

| Run | Description | Macro F0.5 | Status |
|-----|-------------|-----------|--------|
| Baseline | LightGBM 16-feat, default thresholds | 0.864000 | Reference |
| **V2 (Final)** | Country-specific thresholds + exact-name override | **0.864691** | ✅ **Best Submission** |
| Emergency Opt | Pre-ranked inverted index + threshold grid | 0.858811 | ❌ Rejected (worse) |

**Validation on 20K queries:**

| Metric | Value |
|--------|-------|
| Candidate Recall | 78.20% |
| Precision | 92.83% |
| Recall | 67.67% |
| **Macro F0.5** | **0.8640** |

---

## 🔗 Important Links

| Resource | Link |
|----------|------|
| 🏁 Competition Portal | [Unstop — Amazon ML Challenge 2026](https://unstop.com/competitions/1743604/round/1593683/play/code) |
| 💻 GitHub Repository | [Amazon-ML-Challenge-2026](https://github.com/swapnil-exxe/Amazon-ML-Challenge-2026) |
| 📦 Best Submission File | `go2_output/matching_results_v2.tsv` (in this repo) |
| 🤖 Trained Model | `go2_output/go2_model.joblib` (in this repo) |

---

## 📐 Full Pipeline Architecture

```
╔══════════════════════════════════════════════════════════════════════════╗
║                    AMAZON ML CHALLENGE 2026 PIPELINE                     ║
╚══════════════════════════════════════════════════════════════════════════╝

  RAW DATA (Unstop Portal)
  ┌──────────────────────────────────────────────┐
  │  student_resource/dataset/                   │
  │  ├── train/train_source1.tsv  (~200 MB)      │
  │  ├── train/train_source2.tsv  (~467 MB)      │
  │  ├── train/train_source3.tsv  (~480 MB)      │
  │  ├── train/train_ground_truth.tsv (~121 MB)  │
  │  ├── test/test_source1.tsv    (~167 MB)      │
  │  ├── test/test_source2.tsv    (~486 MB)      │
  │  └── test/test_source3.tsv    (~483 MB)      │
  └──────────────────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────┐
  │  PHASE 1 — DATA AUDIT                                   │
  │  go1_output/phase_1_audit/audit_dataset.py              │
  │                                                         │
  │  • Counts records, checks for nulls & duplicates        │
  │  • Analyses country distribution                        │
  │  • Computes ground-truth match statistics               │
  │  • SHA-256 verification of all artifacts                │
  │                                                         │
  │  OUTPUT: reports/ (CSV + MD audit files)                │
  └─────────────────────────────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────┐
  │  PHASE 2 — DATA CLEANING                                │
  │  go1_output/phase_2_cleaning/src/normalize.py           │
  │                                                         │
  │  • Strip legal suffixes (LLC, Ltd, Inc, Pvt …)          │
  │  • Lowercase + Unicode normalization                    │
  │  • Tokenise business names → "signature name"           │
  │  • Standardise address fields                           │
  │  • Infer missing country from phone/address patterns    │
  │  • Remove punctuation noise                             │
  │                                                         │
  │  OUTPUT: go1_output/phase_2_cleaning/cleaned/           │
  │          clean_train_source1/2/3.tsv                    │
  │          clean_test_source1/2/3.tsv                     │
  └─────────────────────────────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────┐
  │  PHASE 3 — CANDIDATE BLOCKING                           │
  │  go1_output/phase_3_blocking/src/                       │
  │                                                         │
  │  • Build country-isolated inverted index on Source 2    │
  │  • For each Source 1 query → look up candidate matches  │
  │  • Evidence pre-ranking score per candidate:            │
  │      name token overlap    → +3.0                       │
  │      address token overlap → +1.5                       │
  │      number match          → +2.0                       │
  │      exact name match      → +10.0 (bonus)             │
  │  • Keep TOP-350 candidates per query                    │
  │  • Blocking recall on train: 78.20%                     │
  │                                                         │
  │  OUTPUT: go1_output/phase_3_blocking/candidates/        │
  │          candidate_pairs_train.tsv  (~7.5 GB)           │
  │          candidate_pairs_test.tsv   (~4.8 GB)           │
  └─────────────────────────────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────┐
  │  PHASE 4 — ML INFERENCE (LightGBM)                      │
  │  go2_output/run_go2_inference_v2.py                     │
  │                                                         │
  │  For each candidate pair → extract 16 features:         │
  │  ┌────────────────────────────────────────────────┐     │
  │  │  1. Exact name match       9.  Address missing │     │
  │  │  2. Exact sig-name match  10.  Exact num match │     │
  │  │  3. Name Jaccard          11.  Num Jaccard     │     │
  │  │  4. Sig-name Jaccard      12.  Abs addr len    │     │
  │  │  5. Abs name len diff     13.  Country match   │     │
  │  │  6. Prefix-4 match        14.  S3 boolean flag │     │
  │  │  7. Exact addr match      15.  Candidate rank  │     │
  │  │  8. Addr Jaccard          16.  Total candidates│     │
  │  └────────────────────────────────────────────────┘     │
  │                                                         │
  │  → go2_model.joblib predicts match probability          │
  │                                                         │
  │  Country-Specific Decision Thresholds:                  │
  │      US entities    → prob ≥ 0.60 → MATCH              │
  │      India entities → prob ≥ 0.50 → MATCH              │
  │      Other          → prob ≥ 0.35 → MATCH              │
  │      Fallback       → prob ≥ 0.30 → MATCH              │
  │                                                         │
  │  + Exact-Name Override: identical names → always MATCH  │
  │                                                         │
  │  OUTPUT: go2_output/matching_results_v2.tsv (72 MB)     │
  │          Format: source1_id \t source2_id               │
  └─────────────────────────────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────┐
  │  SUBMISSION                                             │
  │  Upload matching_results_v2.tsv to Unstop portal        │
  │  Macro F0.5 = 0.864691  ✅                              │
  └─────────────────────────────────────────────────────────┘
```

---

## 🗂️ Full File Structure

```
Amazon-ML-Challenge-2026/
│
├── 📄 README.md                          ← You are here
├── 📄 requirements.txt                   ← Python dependencies
├── 📄 DATASET_SETUP.md                   ← How to get competition data
│
├── 🔍 PHASE 1 — Audit & Cleaning
│   └── go1_output/
│       ├── phase_1_audit/
│       │   ├── audit_dataset.py          ← Runs full data audit
│       │   └── reports/
│       │       ├── dataset_audit.md      ← Human-readable audit summary
│       │       ├── dataset_statistics.csv
│       │       ├── country_distribution.csv
│       │       ├── duplicate_id_report.csv
│       │       ├── ground_truth_audit.md
│       │       └── missing_values.csv
│       │
│       ├── phase_2_cleaning/
│       │   ├── src/normalize.py          ← Core cleaning + normalization
│       │   └── reports/
│       │       ├── cleaning_report.md
│       │       ├── cleaning_statistics.csv
│       │       └── normalization_examples.csv
│       │
│       ├── phase_3_blocking/
│       │   ├── src/
│       │   │   ├── blocking_keys.py      ← Token-based blocking key generation
│       │   │   ├── build_indexes.py      ← Inverted index construction
│       │   │   ├── generate_candidates.py← Top-350 candidate pair generation
│       │   │   ├── evaluate_blocking.py  ← Blocking recall measurement
│       │   │   ├── run_go1_phase3.py     ← Main Phase 3 runner
│       │   │   ├── run_v2_blocking_pipeline.py ← V2 improved pipeline
│       │   │   └── run_fast_v2_eval.py   ← Fast recall evaluation
│       │   ├── reports/
│       │   │   ├── blocking_report.md
│       │   │   ├── candidate_statistics.csv
│       │   │   ├── blocking_experiment_log.csv
│       │   │   ├── missed_match_analysis.csv
│       │   │   └── token_frequency_report.csv
│       │   └── logs/
│       │       ├── phase_3_run.log
│       │       └── final_go1_run.log
│       │
│       ├── archive/                      ← Experimental scripts (reference)
│       │   ├── test_evidence_ranking.py  ← EXP-C1 evidence pre-ranking
│       │   ├── run_full_evidence_benchmark.py
│       │   └── ...
│       │
│       └── GO1_FINAL_AUDIT.md            ← Phase 1–3 final summary
│
├── 🤖 PHASE 4 — ML Inference
│   └── go2_output/
│       ├── run_go2_inference_v2.py       ← ⭐ MAIN INFERENCE SCRIPT (V2)
│       ├── go2_model.joblib              ← ⭐ TRAINED LightGBM MODEL (0.23 MB)
│       ├── matching_results_v2.tsv       ← ⭐ BEST SUBMISSION (Macro F0.5 0.8647)
│       ├── matching_results.tsv          ← Baseline submission
│       ├── run_go2_inference_only.py     ← Inference without retraining
│       ├── run_full_go2.py               ← Full train+infer pipeline
│       ├── run_pilot_go2.py              ← Pilot test on small batch
│       ├── go2_inference_v2.log          ← V2 inference run log
│       ├── go2_final_summary.json        ← Model & run metrics
│       └── spp/
│           ├── run_go2_inference_spp.py  ← S++ 20-feature variant (experimental)
│           └── go2_model_spp.joblib      ← S++ model
│
├── 🔬 OPTIMIZATION EXPERIMENTS
│   └── spp_optimization/
│       ├── experiments/
│       │   ├── run_optimization_loop.py        ← Master experiment runner
│       │   ├── emergency_fast_optimizer.py     ← Emergency threshold grid search
│       │   ├── final_submission_mode.py        ← Final submission generator
│       │   ├── dev_ground_truth.tsv            ← Dev validation ground truth
│       │   ├── val_ground_truth.tsv            ← Val split ground truth
│       │   ├── baseline_streaming/
│       │   │   ├── evaluate_baseline_streaming.py ← Memory-safe streaming eval
│       │   │   ├── phase2_error_ceiling.py        ← Error taxonomy analysis
│       │   │   ├── baseline_1k/5k/10k/20k.csv    ← Progressive eval results
│       │   │   └── phase2_error_ceiling.csv
│       │   ├── exp_001/
│       │   │   ├── config.json           ← Experiment config
│       │   │   └── metrics.csv           ← Experiment results
│       │   └── exp_blocking/
│       │       └── run_blocking_experiments.py  ← EXP-B1 to EXP-B4
│       ├── metrics/
│       │   ├── baseline_reproduced.json/csv
│       │   ├── blocking_failure_categories.csv
│       │   └── experiment_matrix.csv
│       └── reports/
│           ├── BASELINE_REPRODUCTION.md
│           ├── BLOCKING_ERROR_ANALYSIS.md
│           ├── FINAL_REPORT.md
│           └── S_FINAL_SCORECARD.md
│
├── 🧪 ROOT ANALYSIS SCRIPTS
│   ├── blocking_recall_audit_v2.py       ← Blocking recall measurement (v2)
│   ├── blocking_recall_experiment.py     ← Recall experiment runner
│   ├── run_spp_blocking_failure_analysis.py ← Failure category breakdown
│   ├── run_spp_challenger_experiments.py ← S++ challenger model tests
│   ├── run_spp_challenger_2_experiment.py
│   └── run_spp_threshold_grid.py        ← Threshold grid search
│
├── 📊 ROOT REPORTS
│   ├── ARYA_FINAL_REPORT.md → GO2_FINAL_REPORT.md
│   ├── BLOCKING_RECALL_EXPERIMENT_REPORT.md
│   ├── FINAL_PROJECT_STATUS.md
│   ├── FINAL_PERFORMANCE_AUDIT.md
│   ├── FINAL_DECISION_AUDIT.md
│   ├── FINAL_SUBMISSION_RECORD.md
│   └── S_PLUS_PLUS_OPTIMIZATION_REPORT.md
│
├── 🗃️ CODE (Packaged Version)
│   └── code/business_entity_resolution/
│       ├── README.md
│       ├── requirements.txt
│       └── src/
│           ├── run_go2_inference_v2.py   ← Packaged inference script
│           └── go2_model.joblib          ← Packaged model
│
└── 📁 COMPETITION DATA (not in git — download from Unstop)
    └── student_resource/
        ├── dataset/
        │   ├── train/                    ← Training source files
        │   └── test/                     ← Test source files
        └── utils/
            └── validate_submission.py    ← Official submission validator
```

---

## 🧠 How the ML Works

### Problem Statement
We are given two catalogs of businesses (Source 1 and Source 2). Each record has fields like `name`, `address`, `phone number`, `country`. The goal is to produce pairs `(source1_id, source2_id)` that refer to the same real-world business.

### Why is this hard?
- Names are **noisy** — "McDonald's Corp", "McDonalds", "MCDONALDS INC" all mean the same thing
- Addresses are **abbreviated** or **missing**
- Phone numbers have **different country code formats**
- One business in Source 1 may match **zero or many** entries in Source 2
- The dataset has **millions of records** — brute-force pairwise comparison is impossible

### Our 4-Phase Solution

#### Phase 1 — Understand the Data
Before writing any ML code, we audited the dataset:
- How many records? Any duplicates?
- What countries appear? What's missing?
- What does the ground truth look like?
- Are there systematic patterns in the matches?

#### Phase 2 — Clean & Normalise
Raw text is too noisy for direct comparison. We:
1. Strip legal suffixes (`LLC`, `Ltd`, `Inc`, `Pvt`, `GmbH`, `S.A.` …)
2. Lowercase everything, remove punctuation
3. Normalise Unicode (café → cafe)
4. Create a **"signature name"** = name after legal suffix removal + lowercase
5. Standardise address fields, fill missing country from phone prefix

#### Phase 3 — Blocking (Finding Candidates)
Comparing every Source 1 record against every Source 2 record would be **billions of pairs** — computationally impossible. Instead:

1. Build a **country-isolated inverted index** on Source 2 tokens
2. For each Source 1 record, look up matching tokens in the index
3. Score each candidate with an **evidence pre-ranking** formula:
   ```
   score = (name_token_overlap × 3.0)
         + (address_token_overlap × 1.5)
         + (number_match × 2.0)
         + (exact_name_bonus × 10.0)
   ```
4. Keep only the **Top 350** candidates per query
5. This reduces billions of pairs → ~4.8–7.5GB of manageable candidates

#### Phase 4 — LightGBM Classification
For each candidate pair we extract **16 hand-crafted pairwise features**:

| # | Feature | What it measures |
|---|---------|-----------------|
| 1 | Exact name match | Are the full names identical? |
| 2 | Exact sig-name match | Are the stripped names identical? |
| 3 | Name Jaccard | Word-level overlap between names |
| 4 | Sig-name Jaccard | Word-level overlap between stripped names |
| 5 | Abs name length diff | Length difference (catches abbreviations) |
| 6 | Prefix-4 match | Do the first 4 chars match? |
| 7 | Exact address match | Are addresses identical? |
| 8 | Address Jaccard | Word-level overlap between addresses |
| 9 | Address missing flag | Is address data absent? |
| 10 | Exact number match | Are phone/reg numbers identical? |
| 11 | Number Jaccard | Token overlap in numbers |
| 12 | Abs address length diff | Address length difference |
| 13 | Country match | Same country code? |
| 14 | S3 boolean flag | Source-3 cross-reference signal |
| 15 | Candidate rank | Position in evidence-ranked list |
| 16 | Total candidates | How many candidates this query had |

These 16 features are fed into a **LightGBM classifier** (`go2_model.joblib`).

#### Threshold Decision
Instead of a single 0.5 threshold, we use **country-specific thresholds** calibrated on validation data:

```
IF country == "US"    → match if probability ≥ 0.60
IF country == "India" → match if probability ≥ 0.50
IF country == other   → match if probability ≥ 0.35
IF country == unknown → match if probability ≥ 0.30 (fallback)
IF exact name match   → always match (override)
```

This improved Macro F0.5 from 0.864000 → **0.864691**.

---

## 🔄 Data Flow Summary

```
student_resource/dataset/
         │
         │  (raw TSV files — train/test sources)
         ▼
go1_output/phase_2_cleaning/src/normalize.py
         │
         │  (cleaned, normalised TSVs)
         ▼
go1_output/phase_3_blocking/src/generate_candidates.py
         │
         │  (candidate pairs TSV — top 350 per query)
         ▼
go2_output/run_go2_inference_v2.py  +  go2_model.joblib
         │
         │  (match/no-match decision per pair)
         ▼
go2_output/matching_results_v2.tsv
         │
         │  (submit to Unstop)
         ▼
         🏆 Macro F0.5 = 0.864691
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/swapnil-exxe/Amazon-ML-Challenge-2026.git
cd Amazon-ML-Challenge-2026
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get the competition data
Download from [Unstop portal](https://unstop.com/competitions/1743604/round/1593683/play/code) and place in:
```
student_resource/dataset/train/
student_resource/dataset/test/
```
See [DATASET_SETUP.md](DATASET_SETUP.md) for full instructions.

### 4. Run the pipeline (in order)

```bash
# Phase 1 — Audit
python go1_output/phase_1_audit/audit_dataset.py

# Phase 2 — Clean data
python go1_output/phase_2_cleaning/src/normalize.py

# Phase 3 — Generate candidate pairs
python go1_output/phase_3_blocking/src/run_go1_phase3.py

# Phase 4 — Run inference and generate submission
python go2_output/run_go2_inference_v2.py
```

### 5. Validate submission
```bash
python student_resource/utils/validate_submission.py
```

---

## 📦 Key Output Files

| File | Size | Description |
|------|------|-------------|
| `go2_output/matching_results_v2.tsv` | 72 MB | ⭐ **Best submission file** |
| `go2_output/go2_model.joblib` | 0.23 MB | ⭐ **Trained LightGBM model** |
| `go2_output/matching_results.tsv` | 76 MB | Baseline submission |
| `spp_optimization/experiments/dev_ground_truth.tsv` | 97 MB | Dev validation set |
| `spp_optimization/experiments/val_ground_truth.tsv` | 24 MB | Validation ground truth |

---

## ⚙️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.10+** | Core language |
| **LightGBM** | Main classifier |
| **pandas** | Data loading and manipulation |
| **numpy** | Feature array construction |
| **scikit-learn** | Evaluation metrics (F0.5) |
| **joblib** | Model serialisation |
| **psutil** | Memory monitoring during large file processing |

---

## 📊 Experiment Log

| Experiment | Description | Macro F0.5 | Notes |
|------------|-------------|-----------|-------|
| EXP-B1 | Baseline blocking, cap=200 | 0.7820 recall | Recall too low |
| EXP-B2 | Cap=350, evidence pre-rank | 0.7820 recall | Best recall found |
| EXP-B3 | Cap=500, diminishing returns | Same | Extra compute waste |
| EXP-001 | Default threshold 0.5 | 0.864000 | Baseline model |
| **EXP-V2** | Country thresholds + exact override | **0.864691** | ✅ **Final** |
| EXP-EMG | Emergency pre-ranked index | 0.858811 | Rejected |

---

## 🔒 Submission Verification

```
matching_results_v2.tsv  SHA256: 56d88eb1f3b7f1e6dd23b3b626266b9cfb9b59a9257f9e9feedc6a6fc7a72d73
go2_model.joblib         SHA256: 476763ebe4051718167703dbf28e88a85ef80e2982876ccdb028508793534cad
```

---

## 📁 Dataset Note

Raw competition data files are **not included** in this repository (each file is 100MB–8GB+).
Download them from the [Unstop competition portal](https://unstop.com/competitions/1743604/round/1593683/play/code).

See **[DATASET_SETUP.md](DATASET_SETUP.md)** for exact placement instructions.

---

<div align="center">

**Built with ❤️ by Swapnil Patil | Amazon ML Challenge 2026**

</div>
