# FINAL ARTIFACT INVENTORY — ML CHALLENGE 2026

**Generated Date**: 2026-09-27  
**Project**: Business Entity Resolution (Go1 + GO2 Pipeline)  
**Status**: VERIFIED, COMPLIANT & LOCKED  

---

### 1. Original Baseline Production Artifacts (Locked Baseline)

| Artifact Name | Absolute File Path | File Size | SHA256 Checksum | Purpose | Validation Status |
| :--- | :--- | :---: | :--- | :--- | :---: |
| **`matching_results.tsv`** | `/Users/swapnil/Documents/ML/go2_output/matching_results.tsv` | 75.95 MB (`79,637,369` bytes) | `d1c898796e2ab192b627f8d0e8ea83f96e1427b029a8d66ed9165c87bd94d1e8` | Baseline test predictions (1.73M S1 entities, blended threshold $p \ge 0.35$) | **PASS — Safe to Submit** |
| **`team_submission.zip`** | `/Users/swapnil/Documents/ML/team_submission.zip` | 2.08 GB (`2,180,828,506` bytes) | `459e67da33e82ea6b052de382cbdf0805c6cbb09a5ca2138415fa6a7cd043747` | Baseline team submission package containing prediction TSV, candidate TSV, source code, and docs | **PASS — Safe to Submit** |
| **`go2_model.joblib`** | `/Users/swapnil/Documents/ML/go2_output/go2_model.joblib` | 0.23 MB (`244,392` bytes) | `476763ebe4051718167703dbf28e88a85ef80e2982876ccdb028508793534cad` | Trained 16-feature `HistGradientBoostingClassifier` binary checkpoint | **VERIFIED & LOADABLE** |

---

### 2. V2 Country-Specific Threshold Artifacts (Additive Candidate Submission)

| Artifact Name | Absolute File Path | File Size | SHA256 Checksum | Purpose | Validation Status |
| :--- | :--- | :---: | :--- | :--- | :---: |
| **`matching_results_v2.tsv`** | `/Users/swapnil/Documents/ML/go2_output/matching_results_v2.tsv` | 71.92 MB (`75,409,038` bytes) | `56d88eb1f3b7f1e6dd23b3b626266b9cfb9b59a9257f9e9feedc6a6fc7a72d73` | V2 test predictions with country thresholds (US $0.60$, India $0.50$, France $0.35/0.30$) | **PASS — Safe to Submit** |
| **`team_submission_v2.zip`** | `/Users/swapnil/Documents/ML/team_submission_v2.zip` | 2.08 GB (`2,178,988,963` bytes) | `d6926b559404028a9fdd2aca167438f0cead828fb4991ede0474ee7f8ad46d7c` | V2 submission package (internal prediction file verified byte-identical to standalone V2; source script and model checkpoint included) | **PASS — Fully Compliant & Reproducible** |

---

### 3. Project Reports & Documentation

| Artifact Name | Absolute File Path | File Size | SHA256 Checksum | Purpose | Validation Status |
| :--- | :--- | :---: | :--- | :--- | :---: |
| **`FINAL_V2_SUBMISSION_REPORT.md`** | `/Users/swapnil/Documents/ML/FINAL_V2_SUBMISSION_REPORT.md` | 4.81 KB (`4,805` bytes) | `16a8c2cd8d71a923ce21051897409512ab74146a8a7ec61a2326a0724d7bca8c` | Detailed validation, distribution, and submission decision report for V2 package | **COMPLETE** |
| **`BLOCKING_RECALL_EXPERIMENT_REPORT.md`** | `/Users/swapnil/Documents/ML/BLOCKING_RECALL_EXPERIMENT_REPORT.md` | 6.29 KB (`6,292` bytes) | `68d54b4cdfe66228256e5d11c3dbeb91cf45437627900c1193c66680d6231887` | Empirical audit report detailing why multi-pass blocking expansion was rejected | **COMPLETE** |
| **`FINAL_PROJECT_STATUS.md`** | `/Users/swapnil/Documents/ML/FINAL_PROJECT_STATUS.md` | 7.51 KB (`7,508` bytes) | `efef3ddbdf2dbb095ce82d1dbb851b2e1bf3c9902096ae53d5fa6cfdd0605a0b` | Final comprehensive project status report across architecture, experiments, and submission state | **COMPLETE** |
| **`Documentation_template.md`** | `/Users/swapnil/Documents/ML/Documentation_template.md` | 6.63 KB (`6,626` bytes) | `a676b64df3686ba4ea015d548e7200fee2d5599bffaf058343a91826960fcce0` | Final methodology, blocking, model, threshold, and limitation documentation | **COMPLETE & FULLY UPDATED** |
| **`FINAL_SUBMISSION_RECORD.md`** | `/Users/swapnil/Documents/ML/FINAL_SUBMISSION_RECORD.md` | 6.07 KB (`6,070` bytes) | `63cdbe5ee7f3152778ca2083dbb1d0e19036c61f22e75e92f15ebcd67e411b93` | Final submission lock audit record | **COMPLETE & UPDATED** |

---

### 4. Baseline Candidate File

| Artifact Name | Absolute File Path | File Size | SHA256 Checksum | Purpose | Validation Status |
| :--- | :--- | :---: | :--- | :--- | :---: |
| **`candidate_pairs_test_v2.tsv`** | `/Users/swapnil/Documents/ML/go1_output/phase_3_blocking/candidates/candidate_pairs_test_v2.tsv` | 2.50 GB (`2,683,678,610` bytes) | `5e0913d8c034fe02d0e1119b5a9d87b74d353557941d986c03b8930c635e714c` | Candidate pairs for test set queries | **VERIFIED INTACT** |
