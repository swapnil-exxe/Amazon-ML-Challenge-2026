# 🛒 Amazon ML Challenge 2026 — Business Entity Resolution

> **72-Hour Hackathon** hosted on [Unstop](https://unstop.com)  
> **Task**: Match business entities across two noisy, real-world product catalogs

---

## 🏆 Results

| Run | Description | Macro F0.5 | Status |
|-----|-------------|-----------|--------|
| Baseline | LightGBM 16-feat, default thresholds | 0.864000 | Reference |
| **V2 (Final)** | Country-specific thresholds + exact-name override | **0.864691** | ✅ Best Submission |
| Emergency Opt | Fast pre-ranked inverted index + threshold grid | 0.858811 | ❌ Rejected (worse) |

**Best Public Score**: **0.864691** (Macro F0.5) — submitted as `matching_results.tsv`

---

## 📐 Pipeline Architecture

```
Source1 (query catalog)
        │
        ▼
Phase 1: Data Audit & SHA256 Verification
        │
        ▼
Phase 2: Cleaning (go1_output/phase_2_cleaning/)
  - Legal suffix normalization
  - Name / address standardization
  - Country inference
        │
        ▼
Phase 3: Blocking (go1_output/phase_3_blocking/)
  - Country-isolated inverted index
  - Evidence pre-ranking (name +3.0, addr +1.5, num +2.0, exact +10.0)
  - Top-350 candidates per query
        │
        ▼
Phase 4: Inference (go2_output/)
  - 16 pairwise LightGBM features
  - go2_model.joblib (LightGBM classifier)
  - Country-specific thresholds:
      US ≥ 0.60 | India ≥ 0.50 | Other ≥ 0.35 (fallback 0.30)
  - Exact-name override
        │
        ▼
matching_results.tsv  ← Unstop submission
```

---

## 🔢 16 LightGBM Features

| # | Feature |
|---|---------|
| 1 | Exact name match |
| 2 | Exact sig-name match |
| 3 | Name Jaccard similarity |
| 4 | Sig-name Jaccard similarity |
| 5 | Abs name length difference |
| 6 | Prefix-4 match |
| 7 | Exact address match |
| 8 | Address Jaccard similarity |
| 9 | Address missing flag |
| 10 | Exact number match |
| 11 | Number Jaccard similarity |
| 12 | Abs address length difference |
| 13 | Country match |
| 14 | S3 boolean flag |
| 15 | Candidate rank |
| 16 | Total candidates |

---

## 📂 Project Structure

```
ML/
├── go2_output/              # Phase 4: Inference engine
│   ├── run_go2_inference_v2.py   # Production V2 pipeline
│   ├── go2_model.joblib          # Trained LightGBM model (0.23 MB)
│   └── spp/                       # S++ 20-feature variant (experimental)
│
├── go1_output/           # Phases 2–3 outputs
│   ├── phase_2_cleaning/     # Cleaning scripts & configs
│   └── phase_3_blocking/     # Blocking scripts & configs
│       └── candidates/       # ⚠️ Large TSVs excluded from git
│
├── spp_optimization/         # Optimization experiments
│   └── experiments/
│       ├── baseline_streaming/    # 1K/5K/10K/20K streaming evaluator
│       ├── exp_blocking/          # Blocking optimization (EXP-B1–B4)
│       ├── emergency_optimization/ # Legal-suffix + rescue rules
│       └── ...
│
├── student_resource/         # ⚠️ Raw competition data (excluded from git)
│   └── dataset/              #    Download from Unstop portal
│
├── code/                     # Utility code
├── blocking_recall_audit_v2.py
├── run_spp_*.py              # S++ challenger experiments
├── *.md                      # Reports & audits
├── requirements.txt
└── DATASET_SETUP.md
```

---

## ⚙️ Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get competition data

See [DATASET_SETUP.md](DATASET_SETUP.md) — raw data files must be downloaded from Unstop and placed in `student_resource/dataset/`.

### 3. Run the pipeline

```bash
# Phase 2: Cleaning
python go1_output/phase_2_cleaning/run_cleaning.py

# Phase 3: Blocking
python go1_output/phase_3_blocking/run_blocking.py

# Phase 4: Inference (V2)
python go2_output/run_go2_inference_v2.py
```

---

## 🔒 Protected Artifacts (SHA256)

| Artifact | SHA256 |
|----------|--------|
| `go2_model.joblib` | `476763eb...793534cad` |
| `matching_results_v2.tsv` | `56d88eb1...c7a72d73` |
| `team_submission_v2.zip` | `d6926b55...d7d7c` |

---

## 📊 Validation Results (20K queries)

| Metric | Value |
|--------|-------|
| Candidate Recall | 78.20% |
| Precision | 92.83% |
| Recall | 67.67% |
| **Macro F0.5** | **0.8640** |
| Blocking failures | 21.80% (15,004 pairs) |
| Threshold rejections | 10.54% (7,253 pairs) |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **LightGBM** — main classifier
- **pandas / numpy** — data processing
- **scikit-learn** — evaluation metrics
- **joblib** — model serialization
- **psutil** — memory monitoring

---

## 📜 License

Academic / competition use. Competition data © Amazon / Unstop.
