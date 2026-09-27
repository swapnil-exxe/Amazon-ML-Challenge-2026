# GO2 HANDOFF SPECIFICATION — PHASE 4 ML FEATURE ENGINEERING

## 1. Input Datasets for Go2
Go2 MUST consume ONLY the following verified final files:

### Training Stage:
- Target S1 Entities: `go1_output/phase_2_cleaning/cleaned/clean_train_source1.tsv`
- Candidate Pool S2: `go1_output/phase_2_cleaning/cleaned/clean_train_source2.tsv`
- Candidate Pool S3: `go1_output/phase_2_cleaning/cleaned/clean_train_source3.tsv`
- Verified Candidate Pairs: `go1_output/phase_3_blocking/candidates/candidate_pairs_train.tsv`
- Ground Truth Labels: `student_resource/dataset/train/train_ground_truth.tsv` (for target label construction)

### Test Stage:
- Target S1 Entities: `go1_output/phase_2_cleaning/cleaned/clean_test_source1.tsv`
- Candidate Pool S2: `go1_output/phase_2_cleaning/cleaned/clean_test_source2.tsv`
- Candidate Pool S3: `go1_output/phase_2_cleaning/cleaned/clean_test_source3.tsv`
- Verified Candidate Pairs: `go1_output/phase_3_blocking/candidates/candidate_pairs_test.tsv`

## 2. Files Go2 MUST NOT Modify
- Do NOT modify raw datasets in `student_resource/dataset/`.
- Do NOT modify cleaned datasets in `go1_output/phase_2_cleaning/cleaned/`.
- Do NOT modify candidate TSVs in `go1_output/phase_3_blocking/candidates/`.

## 3. Storage & Safety Guidelines for Go2
- Candidate TSVs are formatted as `source1_entity_id\tcandidate_entity_ids` (comma-separated).
- Current free disk space: `4.8 GiB`. Feature extraction pipelines should stream or store feature matrices efficiently using compressed Parquet or memmapped NumPy arrays.
