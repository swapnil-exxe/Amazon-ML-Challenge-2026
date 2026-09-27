# FINAL SAFE CLEANUP REPORT — AMAZON ML CHALLENGE

Generated At: 2026-09-26 22:08:00
Target Folder: `/Users/swapnil/Documents/ML/`

---

## 1. Executive Metrics

- **BEFORE ML Folder Size**: `21,857.64 MB` (~`21.35 GB`)
- **AFTER ML Folder Size**: `21,857.40 MB` (~`21.34 GB`)
- **SPACE RECOVERED**: `250.73 KB` (~`0.24 MB`)

---

## 2. File Action Breakdown

### FILES DELETED
- **Count**: `20 files`
- **Total Size**: `250,727 bytes` (`244.85 KB`)
- **Details**:
  - `5` macOS Finder metadata junk files (`.DS_Store`)
  - `8` Python compiled bytecode cache files (`__pycache__/*.pyc`)
  - `7` exact byte-for-byte duplicate directory files (`student_resource/go1_output/phase_1_audit/`)

### FILES ARCHIVED
- **Count**: `10 files`
- **Total Size**: `52,893 bytes` (`51.65 KB`)
- **Target Archive Directory**: `go1_output/archive/`
- **Archived Scripts**:
  1. `run_full_evidence_benchmark.py` (`7,609 B`)
  2. `test_evidence_ranking.py` (`6,525 B`)
  3. `analyze_blocking_errors.py` (`5,886 B`)
  4. `test_pipeline_opt.py` (`5,503 B`)
  5. `verify_final_audit.py` (`5,124 B`)
  6. `complete_train_cands.py` (`4,939 B`)
  7. `test_capping.py` (`4,789 B`)
  8. `test_caps.py` (`4,311 B`)
  9. `eval_disk_recall.py` (`4,270 B`)
  10. `fast_bench.py` (`3,937 B`)

### PROTECTED FILES
- **Verified Count**: `59 files`
- **Total Size**: `21,857.35 MB` (~`21.34 GB`)
- **Verification Summary**:
  - `student_resource/dataset/`: `8 files` (2,389.28 MB) — **UNTOUCHED & VERIFIED**
  - `go1_output/phase_2_cleaning/cleaned/`: `6 files` (6,818.25 MB) — **UNTOUCHED & VERIFIED**
  - `go1_output/phase_3_blocking/candidates/`: `2 files` (12,620.23 MB) — **UNTOUCHED & VERIFIED**
  - Final Audit & Handoff Reports (`GO1_FINAL_AUDIT.md`, `GO1_FINAL_SUMMARY.json`, `GO2_HANDOFF.md`): **VERIFIED**
  - All Core Reproduction Pipeline Scripts & Logs: **VERIFIED**

---

## 3. System Verification Status

```text
ORIGINAL DATA: PASS
GO1 V2: PASS
GO2 INPUTS: PASS
FINAL STATUS: SAFE
```

- **Original Data Integrity**: All 7 raw TSV datasets + `train_ground_truth.tsv` verified line-by-line with exact row count matching.
- **Go1 V2 Integrity**: Candidate files (`candidate_pairs_train.tsv` & `candidate_pairs_test.tsv`) fully verified and protected.
- **Go2 Input Readiness**: All input files required for Phase 4 ML Feature Engineering remain intact and accessible.
