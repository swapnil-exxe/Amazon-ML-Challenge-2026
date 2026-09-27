# Project Manifest: Amazon ML Challenge 2026

## 1. Core Documentation & Configuration
- `README.md`: Primary system architecture, metric definitions, and execution guide.
- `DATASET_SETUP.md`: Dataset layout specifications and acquisition guidelines.
- `PROJECT_MANIFEST.md`: Complete directory inventory of committed & excluded files.
- `LICENSE`: Apache 2.0 open-source software license.
- `requirements.txt`: Python package requirements.
- `.gitignore`: Rules excluding high-volume raw/intermediate datasets and archives.

## 2. Phase 1 — Data Audit
- `go1_output/phase_1_audit/audit_dataset.py`: Comprehensive dataset sanity & format auditor.
- `go1_output/phase_1_audit/reports/`: Statistical reports on missingness, duplicates, and country distributions.

## 3. Phase 2 — Data Cleaning & Normalization
- `go1_output/phase_2_cleaning/src/normalize.py`: Legal suffix stripping, unicode normalization, signature name derivation.
- `go1_output/phase_2_cleaning/reports/`: Cleaning metrics and normalization example tables.

## 4. Phase 3 — Candidate Blocking & Indexing
- `go1_output/phase_3_blocking/src/blocking_keys.py`: Token generation rules.
- `go1_output/phase_3_blocking/src/build_indexes.py`: Inverted index constructor.
- `go1_output/phase_3_blocking/src/generate_candidates.py`: Multi-field evidence ranking & candidate pruning.
- `go1_output/phase_3_blocking/src/run_go1_phase3.py`: Main blocking execution engine.
- `go1_output/phase_3_blocking/src/run_v2_blocking_pipeline.py`: Production candidate generation pipeline.
- `go1_output/phase_3_blocking/reports/`: Candidate count metrics and blocking rule analysis.

## 5. Phase 4 — Pairwise ML Inference & Scoring
- `go2_output/run_go2_inference_v2.py`: Production pairwise inference pipeline (16 features + calibrated thresholds).
- `go2_output/go2_model.joblib`: Trained LightGBM binary classifier (0.23 MB).
- `go2_output/matching_results_v2.tsv`: Best verified submission artifact (Macro F0.5 = 0.864691, 71.92 MB).
- `go2_output/matching_results.tsv`: Baseline reference matches (75.95 MB).
- `go2_output/go2_final_summary.json`: Telemetry metrics from the production inference run.

## 6. Optimization Experiments & Validation
- `spp_optimization/experiments/dev_ground_truth.tsv`: Stratified validation ground truth (96.90 MB).
- `spp_optimization/experiments/val_ground_truth.tsv`: Validation split ground truth (24.0 MB).
- `spp_optimization/experiments/baseline_streaming/evaluate_baseline_streaming.py`: Memory-efficient streaming evaluator.
- `spp_optimization/experiments/baseline_streaming/phase2_error_ceiling.py`: Precision/recall failure taxonomy.
- `spp_optimization/experiments/emergency_fast_optimizer.py`: Rapid grid evaluation script.
- `spp_optimization/experiments/final_submission_mode.py`: Submission pipeline with integrity assertions.
- `spp_optimization/metrics/`: Controlled experiment matrices and baseline reproduction records.
- `spp_optimization/reports/`: Technical analysis scorecards and error breakdowns.

## 7. Standalone Evaluators & Submission Helpers
- `blocking_recall_audit_v2.py`: Standalone recall evaluator.
- `student_resource/utils/validate_submission.py`: Competition submission structure validator.
