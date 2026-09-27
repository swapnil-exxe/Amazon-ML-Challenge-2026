# FINAL V2 SUBMISSION REPORT — ML CHALLENGE 2026

**Report Generated**: 2026-09-27 13:03:00 IST  
**Deadline**: Today, 2026-09-27 at 11:59 PM IST  
**Time Remaining**: 10 Hours, 56 Minutes  

---

### 1. Completion Status
- **Target Output File**: `/Users/swapnil/Documents/ML/arya_output/matching_results_v2.tsv`
- **Total Line Count**: `1,732,545` lines (Exactly 1 header + 1,732,544 data rows)
- **Inference Process PID**: `54442` (`run_arya_inference_v2.py`)
- **PID Execution Status**: **FULLY EXITED** (Return code 0)
- **Total Elapsed Runtime**: 1 Hour, 37 Minutes, 25 Seconds

---

### 2. Baseline Integrity Verification
Re-calculated SHA256 checksums for the original baseline files:

| File Name | Expected SHA256 | Actual SHA256 | Status |
| :--- | :--- | :--- | :---: |
| **`matching_results.tsv`** | `d1c898796e2ab192b627f8d0e8ea83f96e1427b029a8d66ed9165c87bd94d1e8` | `d1c898796e2ab192b627f8d0e8ea83f96e1427b029a8d66ed9165c87bd94d1e8` | **MATCH** |
| **`team_submission.zip`** | `459e67da33e82ea6b052de382cbdf0805c6cbb09a5ca2138415fa6a7cd043747` | `459e67da33e82ea6b052de382cbdf0805c6cbb09a5ca2138415fa6a7cd043747` | **MATCH** |
| **`arya_model.joblib`** | `476763ebe4051718167703dbf28e88a85ef80e2982876ccdb028508793534cad` | `476763ebe4051718167703dbf28e88a85ef80e2982876ccdb028508793534cad` | **MATCH** |

> [!IMPORTANT]
> All original baseline files remain 100% untouched and byte-for-byte identical to their original state.

---

### 3. Official Validator Result
Command executed:
```bash
python3 /Users/swapnil/Documents/ML/student_resource/utils/validate_submission.py \
  --matching /Users/swapnil/Documents/ML/arya_output/matching_results_v2.tsv \
  --test-dir /Users/swapnil/Documents/ML/student_resource/dataset/test
```

**Exact Output Captured**:
```text
ML Challenge 2026 — submission validator
  test dir: /Users/swapnil/Documents/ML/student_resource/dataset/test
  required S1 entities: 1732544
  matching_results_v2.tsv: 1732544 rows (451995 empty, 1280549 non-empty).

WARNING: ID-existence check is OFF (the default) — not checking that matched/candidate IDs exist in the test set. Every other rule is still checked. Re-run with --check-ids to enable it (needs test_source2/3.tsv; uses more memory). A nonexistent ID only lowers your score, never rejects your submission.
WARNING: output/candidate_pairs.tsv not found — skipping candidate_pairs.tsv checks. It is optional here, but your final submission zip must include output/candidate_pairs.tsv.
PASS — no blocking issues found. Safe to submit.
```

- **Validation Status**: **PASS — no blocking issues found**

---

### 4. Prediction Distribution Sanity Check
Comparison between V1 Baseline (blended threshold `p >= 0.35`) and V2 Output (country-specific thresholds: US `p >= 0.60`, India `p >= 0.50`, France `p >= 0.35/0.30` fallback):

| Metric / Country | V1 Baseline Empty % | V2 Output Empty % | V2 Total Queries | V2 Non-Empty Count | V2 Empty Count | Expected Directional Shift |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Overall** | 23.28% | **26.09%** | 1,732,544 | 1,280,549 | 451,995 | Increased (Higher precision filtering) |
| **US** | 29.60% | **33.51%** | 663,106 | 440,925 | 222,181 | Increased (Threshold `0.35 -> 0.60`) |
| **India** | 21.67% | **24.48%** | 809,986 | 611,730 | 198,256 | Increased (Threshold `0.35 -> 0.50`) |
| **France** | 12.16% | **12.16%** | 259,452 | 227,894 | 31,558 | Unchanged (Threshold `0.35/0.30` preserved) |

> [!NOTE]
> The empty percentage increase for US (+3.91%) and India (+2.81%) is the direct expected result of tightening precision thresholds to eliminate lower-confidence candidate pairs on the high-volume regions.

---

### 5. ZIP Package Integrity Verification
- **Target ZIP Path**: `/Users/swapnil/Documents/ML/team_submission_v2.zip`
- **ZIP File SHA256**: `59b6a9806747b7a21ed13d948d6e7d021c7fa999e05099674b88a772a28d63d3`
- **Standalone `matching_results_v2.tsv` SHA256**: `56d88eb1f3b7f1e6dd23b3b626266b9cfb9b59a9257f9e9feedc6a6fc7a72d73`
- **Extracted `output/matching_results.tsv` SHA256**: `56d88eb1f3b7f1e6dd23b3b626266b9cfb9b59a9257f9e9feedc6a6fc7a72d73`
- **Byte-Identical Confirmation**: **YES** (100% identical)
- **ZIP Contents**:
  - `output/matching_results.tsv`
  - `output/candidate_pairs.tsv`
  - `code/business_entity_resolution/` (`src/`, `README.md`, `requirements.txt`)
  - `Documentation_template.md`

---

### 6. Deadline & Time Buffer
- **Current IST Time**: 1:03 PM IST
- **Deadline**: 11:59 PM IST
- **Buffer Remaining**: 10 Hours, 56 Minutes (> 2 Hours required safety buffer)

---

### 7. FINAL DECISION

**SAFE TO SUBMIT: Upload matching_results_v2.tsv and team_submission_v2.zip as a SECOND, ADDITIONAL leaderboard entry. DO NOT withdraw or replace the original submission — keep both live.**
