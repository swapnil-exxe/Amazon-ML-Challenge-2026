# SAFE ML CLEANUP PLAN — AMAZON ML CHALLENGE

Generated At: 2026-09-26 22:06:00
Target Folder: `/Users/swapnil/Documents/ML/`

## Executive Summary

| Category | File Count | Size | Description |
| :--- | :---: | :---: | :--- |
| **Current ML Folder Size** | `89 files` | **`21,857.64 MB`** (~`21.35 GB`) | Full baseline inventory |
| **SAFE TO DELETE** | `20 files` | **`0.24 MB`** (`246,572 B`) | Junk `.DS_Store`, bytecode caches, duplicate directory |
| **SAFE TO ARCHIVE** | `10 files` | **`0.05 MB`** (`52,949 B`) | One-off diagnostic scripts moved to `go1_output/archive/` |
| **MUST KEEP** | `59 files` | **`21,857.35 MB`** (~`21.34 GB`) | Protected raw datasets, cleaned datasets, candidate TSVs, core scripts, reports |
| **Expected Size After Cleanup** | `69 files` | **`21,857.40 MB`** (~`21.34 GB`) | Post-cleanup total folder footprint |
| **Expected Space Recovered** | `20 files` | **`0.24 MB`** (`246.57 KB`) | Recovered from disk |

---

## 1. SAFE TO DELETE

The following 20 files are confirmed disposable artifacts (OS junk files, temporary bytecode caches, and exact byte-for-byte duplicate directory files):

| Path | Size (Bytes) | Reason |
| :--- | :---: | :--- |
| `/Users/swapnil/Documents/ML/.DS_Store` | `12,292` | macOS Finder metadata junk file |
| `go1_output/.DS_Store` | `8,196` | macOS Finder metadata junk file |
| `go1_output/phase_2_cleaning/.DS_Store` | `6,148` | macOS Finder metadata junk file |
| `student_resource/.DS_Store` | `8,196` | macOS Finder metadata junk file |
| `student_resource/dataset/.DS_Store` | `8,196` | macOS Finder metadata junk file |
| `go1_output/phase_3_blocking/src/__pycache__/run_master_go1_audit.cpython-313.pyc` | `36,477` | Python compiled bytecode cache |
| `go1_output/phase_3_blocking/src/__pycache__/run_final_go1.cpython-313.pyc` | `34,258` | Python compiled bytecode cache |
| `go1_output/phase_3_blocking/src/__pycache__/run_go1_phase3.cpython-313.pyc` | `32,452` | Python compiled bytecode cache |
| `go1_output/phase_3_blocking/src/__pycache__/generate_candidates.cpython-313.pyc` | `29,210` | Python compiled bytecode cache |
| `go1_output/phase_3_blocking/src/__pycache__/run_v2_blocking_pipeline.cpython-313.pyc` | `28,858` | Python compiled bytecode cache |
| `go1_output/phase_3_blocking/src/__pycache__/build_indexes.cpython-313.pyc` | `3,498` | Python compiled bytecode cache |
| `go1_output/phase_3_blocking/src/__pycache__/evaluate_blocking.cpython-313.pyc` | `2,540` | Python compiled bytecode cache |
| `go1_output/phase_3_blocking/src/__pycache__/blocking_keys.cpython-313.pyc` | `1,765` | Python compiled bytecode cache |
| `student_resource/go1_output/phase_1_audit/audit_dataset.py` | `24,272` | Exact byte-for-byte duplicate of `go1_output/phase_1_audit/audit_dataset.py` |
| `student_resource/go1_output/phase_1_audit/reports/dataset_audit.md` | `10,147` | Exact byte-for-byte duplicate of `go1_output/phase_1_audit/reports/dataset_audit.md` |
| `student_resource/go1_output/phase_1_audit/reports/dataset_statistics.csv` | `1,173` | Exact byte-for-byte duplicate of `go1_output/phase_1_audit/reports/dataset_statistics.csv` |
| `student_resource/go1_output/phase_1_audit/reports/missing_values.csv` | `1,156` | Exact byte-for-byte duplicate of `go1_output/phase_1_audit/reports/missing_values.csv` |
| `student_resource/go1_output/phase_1_audit/reports/ground_truth_audit.md` | `1,001` | Exact byte-for-byte duplicate of `go1_output/phase_1_audit/reports/ground_truth_audit.md` |
| `student_resource/go1_output/phase_1_audit/reports/country_distribution.csv` | `542` | Exact byte-for-byte duplicate of `go1_output/phase_1_audit/reports/country_distribution.csv` |
| `student_resource/go1_output/phase_1_audit/reports/duplicate_id_report.csv` | `350` | Exact byte-for-byte duplicate of `go1_output/phase_1_audit/reports/duplicate_id_report.csv` |

---

## 2. SAFE TO ARCHIVE

The following 10 one-off diagnostic and experimental scripts are no longer actively invoked by the pipeline, but are retained in `go1_output/archive/` for full historical reproducibility:

| Path | Size (Bytes) | Reason | Target Location |
| :--- | :---: | :--- | :--- |
| `go1_output/phase_3_blocking/src/run_full_evidence_benchmark.py` | `7,609` | Benchmark experiment script | `go1_output/archive/run_full_evidence_benchmark.py` |
| `go1_output/phase_3_blocking/src/test_evidence_ranking.py` | `6,525` | Diagnostic ranking test script | `go1_output/archive/test_evidence_ranking.py` |
| `go1_output/phase_3_blocking/src/analyze_blocking_errors.py` | `5,886` | Blocking error breakdown script | `go1_output/archive/analyze_blocking_errors.py` |
| `go1_output/phase_3_blocking/src/test_pipeline_opt.py` | `5,503` | Diagnostic optimization script | `go1_output/archive/test_pipeline_opt.py` |
| `go1_output/phase_3_blocking/src/verify_final_audit.py` | `5,124` | Diagnostic audit script | `go1_output/archive/verify_final_audit.py` |
| `go1_output/phase_3_blocking/src/complete_train_cands.py` | `4,939` | Experimental candidate generator script | `go1_output/archive/complete_train_cands.py` |
| `go1_output/phase_3_blocking/src/test_capping.py` | `4,789` | Capping experiment script | `go1_output/archive/test_capping.py` |
| `go1_output/phase_3_blocking/src/test_caps.py` | `4,311` | Capping test script | `go1_output/archive/test_caps.py` |
| `go1_output/phase_3_blocking/src/eval_disk_recall.py` | `4,270` | Disk recall evaluator script | `go1_output/archive/eval_disk_recall.py` |
| `go1_output/phase_3_blocking/src/fast_bench.py` | `3,937` | Benchmark helper script | `go1_output/archive/fast_bench.py` |

---

## 3. MUST KEEP

The following 59 files (~`21,857.35 MB`) are strictly protected and MUST remain untouched:

1. **Original Datasets** (`student_resource/dataset/`): All 7 TSV raw data files + `train_ground_truth.tsv` (`2,389.28 MB`).
2. **Cleaned Datasets** (`go1_output/phase_2_cleaning/cleaned/`): All 6 normalized TSV files (`6,818.25 MB`).
3. **Candidate TSV Files** (`go1_output/phase_3_blocking/candidates/`):
   - `candidate_pairs_train.tsv`: `7,669.45 MB` (`8,041,996,198` bytes, 2,206,821 rows)
   - `candidate_pairs_test.tsv`: `4,950.78 MB` (`5,191,272,960` bytes, 1,732,544 rows)
4. **Final Audit & Handoff Artifacts**: `GO1_FINAL_AUDIT.md`, `GO1_FINAL_SUMMARY.json`, `GO2_HANDOFF.md`, `cleanup_duplicate_report.md`.
5. **Core Source Code**:
   - `go1_output/phase_1_audit/audit_dataset.py`
   - `go1_output/phase_2_cleaning/src/normalize.py`
   - `go1_output/phase_3_blocking/src/blocking_keys.py`
   - `go1_output/phase_3_blocking/src/build_indexes.py`
   - `go1_output/phase_3_blocking/src/generate_candidates.py`
   - `go1_output/phase_3_blocking/src/evaluate_blocking.py`
   - `go1_output/phase_3_blocking/src/run_v2_blocking_pipeline.py`
   - `go1_output/phase_3_blocking/src/run_go1_phase3.py`
   - `go1_output/phase_3_blocking/src/run_master_go1_audit.py`
   - `go1_output/phase_3_blocking/src/run_fast_master_audit.py`
   - `go1_output/phase_3_blocking/src/run_fast_v2_eval.py`
   - `go1_output/phase_3_blocking/src/run_final_go1.py`
   - `student_resource/utils/validate_submission.py`
6. **All Phase Reports & Audit Logs**: `dataset_audit.md`, `cleaning_report.md`, `blocking_report.md`, `missed_match_analysis.csv`, `token_frequency_report.csv`, `train_cand_counts.npy`, `test_cand_counts.npy`, execution logs.
