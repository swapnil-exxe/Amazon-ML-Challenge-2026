# GO1 FINAL AUDIT REPORT — AMAZON ML CHALLENGE 2026

## 1. Environment & Hardware Context
- **Machine**: MacBook Air M4
- **Runtime Environment**: Python 3.11 with Pandas & NumPy
- **Available Disk Storage**: `17.0 GiB` (`/dev/disk3s5`)
- **Audit Execution Engine**: Fast streaming line-by-line scanner (< 200 MB RAM usage, < 20s runtime).

## 2. Dataset Inventory & Safety Audit
All 7 original TSV datasets in `student_resource/dataset/` physically exist, were parsed, and verified completely untouched:

| Dataset Name | File Path | Total Rows | Column Count | File Size | Verification Status |
| --- | --- | ---: | ---: | ---: | :---: |
| `train_source1` | `dataset/train/train_source1.tsv` | `2,206,821` | `4` | `200.34 MB` | **UNTOUCHED / SAFE** |
| `train_source2` | `dataset/train/train_source2.tsv` | `5,034,616` | `4` | `466.63 MB` | **UNTOUCHED / SAFE** |
| `train_source3` | `dataset/train/train_source3.tsv` | `5,285,603` | `4` | `480.37 MB` | **UNTOUCHED / SAFE** |
| `train_ground_truth` | `dataset/train/train_ground_truth.tsv` | `2,206,821` | `2` | `121.13 MB` | **UNTOUCHED / SAFE** |
| `test_source1` | `dataset/test/test_source1.tsv` | `1,732,544` | `4` | `166.91 MB` | **UNTOUCHED / SAFE** |
| `test_source2` | `dataset/test/test_source2.tsv` | `4,887,273` | `4` | `485.86 MB` | **UNTOUCHED / SAFE** |
| `test_source3` | `dataset/test/test_source3.tsv` | `5,082,316` | `4` | `482.56 MB` | **UNTOUCHED / SAFE** |

## 3. Phase 2 Cleaned Datasets Audit
All 6 cleaned TSV datasets in `go1_output/phase_2_cleaning/cleaned/` physically exist and preserve 100% row counts and IDs:
- `clean_train_source1.tsv`: `2,206,821` rows (`614.5 MB`)
- `clean_train_source2.tsv`: `5,034,616` rows (`1.48 GB`)
- `clean_train_source3.tsv`: `5,285,603` rows (`1.51 GB`)
- `clean_test_source1.tsv`: `1,732,544` rows (`508.4 MB`)
- `clean_test_source2.tsv`: `4,887,273` rows (`1.54 GB`)
- `clean_test_source3.tsv`: `5,082,316` rows (`1.51 GB`)

## 4. Phase 3 Candidate Generation & Ground Truth Recall Evaluation
- **Candidate TSV Storage (Train)**: `candidate_pairs_train.tsv` (`7.4 GB`, `2,206,821` rows)
- **Candidate TSV Storage (Test)**: `candidate_pairs_test.tsv` (`4.8 GB`, `1,732,544` rows)
- **Total Candidate Storage**: `12.2 GB` (Preserves `4.8 GiB` free space on disk safely)
- **Ground Truth True Match Pairs**: `7,638,365`
- **True Matches Retrieved**: `4,622,933`
- **Pair-Level Candidate Recall**: **`60.52%`**
- **S1-Level Complete Match Coverage**: **`40.68%`**

## 5. Candidate Distribution Statistics

| Split | Total S1 | Total Candidates | Mean | Median | P90 | P95 | P99 | Max | Zero Cands |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Training` | `2,206,821` | `621,736,279` | `281.73` | `350.0` | `350.0` | `350.0` | `350.0` | `350` | `152,282` |
| `Test` | `1,732,544` | `401,025,269` | `231.47` | `332.0` | `350.0` | `350.0` | `350.0` | `350` | `178,892` |

## 6. Machine-Readable Integrity & Compliance Checks
- Original Datasets Safe: **PASS**
- Phase 1 Verified: **PASS**
- Phase 2 Verified: **PASS**
- Candidate Files Validated: **PASS** (Exact line counts: Train `2,206,821`, Test `1,732,544`)
- ID Integrity: **PASS** (Valid `S1-` entity IDs, no self-matches, valid `S2-`/`S3-` candidates)
- Country Isolation: **PASS** (Enforced across US, India, France)
- Storage Safety: **PASS** (`12.2 GB` consumed, `4.8 GiB` remaining)

## 7. Next Stage Handoff to Go2 (Phase 4)
Go2 should consume:
1. `go1_output/phase_2_cleaning/cleaned/clean_train_source1.tsv`
2. `go1_output/phase_2_cleaning/cleaned/clean_train_source2.tsv`
3. `go1_output/phase_2_cleaning/cleaned/clean_train_source3.tsv`
4. `go1_output/phase_3_blocking/candidates/candidate_pairs_train.tsv`
5. `go1_output/phase_2_cleaning/cleaned/clean_test_source1.tsv`
6. `go1_output/phase_2_cleaning/cleaned/clean_test_source2.tsv`
7. `go1_output/phase_2_cleaning/cleaned/clean_test_source3.tsv`
8. `go1_output/phase_3_blocking/candidates/candidate_pairs_test.tsv`
