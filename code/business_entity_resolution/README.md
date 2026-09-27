# Business Entity Resolution Pipeline — Run Instructions

## Overview
This package contains the complete, reproducible end-to-end entity matching pipeline for ML Challenge 2026.

## Directory Structure
```text
code/business_entity_resolution/
├── src/
│   ├── run_arya_inference_only.py    # Main streaming inference script
│   └── arya_model.joblib             # Trained HistGradientBoostingClassifier model
├── README.md                         # This file
└── requirements.txt                  # Python dependencies
```

## Setup & Dependencies
Install dependencies:
```bash
pip install -r requirements.txt
```

## Reproduction Instructions
1. Ensure dataset files exist in `student_resource/dataset/test/`.
2. Ensure candidate pairs file exists in `manthan_output/phase_3_blocking/candidates/candidate_pairs_test_v2.tsv`.
3. Execute the streaming inference pipeline:
```bash
python3 src/run_arya_inference_only.py
```
4. Output will be generated at `matching_results.tsv` and validated using:
```bash
python3 student_resource/utils/validate_submission.py --matching matching_results.tsv --test-dir student_resource/dataset/test
```
