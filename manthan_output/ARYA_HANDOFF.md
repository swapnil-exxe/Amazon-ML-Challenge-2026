# ARYA HANDOFF SPECIFICATION — PHASE 4 ML FEATURE ENGINEERING

## 1. Input Datasets for Arya
Arya MUST consume ONLY the following verified final files:

### Training Stage:
- Target S1 Entities: `manthan_output/phase_2_cleaning/cleaned/clean_train_source1.tsv`
- Candidate Pool S2: `manthan_output/phase_2_cleaning/cleaned/clean_train_source2.tsv`
- Candidate Pool S3: `manthan_output/phase_2_cleaning/cleaned/clean_train_source3.tsv`
- Verified Candidate Pairs: `manthan_output/phase_3_blocking/candidates/candidate_pairs_train.tsv`
- Ground Truth Labels: `student_resource/dataset/train/train_ground_truth.tsv` (for target label construction)

### Test Stage:
- Target S1 Entities: `manthan_output/phase_2_cleaning/cleaned/clean_test_source1.tsv`
- Candidate Pool S2: `manthan_output/phase_2_cleaning/cleaned/clean_test_source2.tsv`
- Candidate Pool S3: `manthan_output/phase_2_cleaning/cleaned/clean_test_source3.tsv`
- Verified Candidate Pairs: `manthan_output/phase_3_blocking/candidates/candidate_pairs_test.tsv`

## 2. Files Arya MUST NOT Modify
- Do NOT modify raw datasets in `student_resource/dataset/`.
- Do NOT modify cleaned datasets in `manthan_output/phase_2_cleaning/cleaned/`.
- Do NOT modify candidate TSVs in `manthan_output/phase_3_blocking/candidates/`.

## 3. Storage & Safety Guidelines for Arya
- Candidate TSVs are formatted as `source1_entity_id\tcandidate_entity_ids` (comma-separated).
- Current free disk space: `4.8 GiB`. Feature extraction pipelines should stream or store feature matrices efficiently using compressed Parquet or memmapped NumPy arrays.
