# DUPLICATE FILE REPORT — AMAZON ML CHALLENGE

Generated At: 2026-09-26 22:06:00
Target Directory: `/Users/swapnil/Documents/ML/`

## Summary of Duplicate Findings

A complete MD5 hash comparison across all files in `/Users/swapnil/Documents/ML/` identified **7 exact byte-for-byte duplicate files** (totaling 38,698 bytes ~38.7 KB).

All 7 duplicates are located in the redundant subdirectory `student_resource/manthan_output/phase_1_audit/`, which is a full duplicate copy of the canonical audit directory `manthan_output/phase_1_audit/`.

No duplicate TSV datasets, cleaned datasets, or candidate TSV files were found.

---

## Detailed Duplicate Analysis

| Duplicate File | Original Canonical File | Size (Bytes) | MD5 Hash | Recommended Action |
| :--- | :--- | :---: | :---: | :--- |
| `student_resource/manthan_output/phase_1_audit/audit_dataset.py` | `manthan_output/phase_1_audit/audit_dataset.py` | `24,272` | `d36fafe1e82b1284cbe60a93d5f2c58e` | **SAFE TO DELETE** |
| `student_resource/manthan_output/phase_1_audit/reports/dataset_audit.md` | `manthan_output/phase_1_audit/reports/dataset_audit.md` | `10,147` | `2738bfb85517a1eb7c500947f04dfcc3` | **SAFE TO DELETE** |
| `student_resource/manthan_output/phase_1_audit/reports/dataset_statistics.csv` | `manthan_output/phase_1_audit/reports/dataset_statistics.csv` | `1,173` | `1d432d259dfd67735efa21b8c794a445` | **SAFE TO DELETE** |
| `student_resource/manthan_output/phase_1_audit/reports/missing_values.csv` | `manthan_output/phase_1_audit/reports/missing_values.csv` | `1,156` | `bbc0aa2cdc441065c96f078a9eda3033` | **SAFE TO DELETE** |
| `student_resource/manthan_output/phase_1_audit/reports/ground_truth_audit.md` | `manthan_output/phase_1_audit/reports/ground_truth_audit.md` | `1,001` | `bfbe81997d014a09d94935d78ae43f2c` | **SAFE TO DELETE** |
| `student_resource/manthan_output/phase_1_audit/reports/country_distribution.csv` | `manthan_output/phase_1_audit/reports/country_distribution.csv` | `542` | `9e652de7a3a3bcfc9e916d038a87f7e2` | **SAFE TO DELETE** |
| `student_resource/manthan_output/phase_1_audit/reports/duplicate_id_report.csv` | `manthan_output/phase_1_audit/reports/duplicate_id_report.csv` | `350` | `25834f6883586ce11ea780088f16b260` | **SAFE TO DELETE** |

---

## Candidate File Verification
- `manthan_output/phase_3_blocking/candidates/candidate_pairs_train.tsv`: **UNIQUE** (7.5 GB, 2,206,821 rows, modified 2026-09-26 21:46)
- `manthan_output/phase_3_blocking/candidates/candidate_pairs_test.tsv`: **UNIQUE** (4.8 GB, 1,732,544 rows, modified 2026-09-26 04:31)

No old or alternate candidate pair versions exist on disk.
