#!/usr/bin/env python3
"""
Phase 1 Audit Script for Amazon ML Challenge 2026 - Business Entity Resolution
This script performs dataset verification, row counts, missing values audit,
entity ID audit, country audit, duplicate business audit, data quality audit,
and ground truth audit on the 7 dataset files without modifying any data.
"""

import sys
import os
import re
import time
import logging
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path("/Users/swapnil/Documents/ML/student_resource")

# Auto-detect dataset directory
if (BASE_DIR / "dataset" / "train").exists():
    DATASET_ROOT = BASE_DIR / "dataset"
elif (BASE_DIR / "train").exists():
    DATASET_ROOT = BASE_DIR
else:
    DATASET_ROOT = BASE_DIR

OUTPUT_DIR = SCRIPT_DIR
REPORTS_DIR = OUTPUT_DIR / "reports"
LOGS_DIR = OUTPUT_DIR / "logs"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Also ensure student_resource/manthan_output/phase_1_audit directory exists
ALT_OUTPUT_DIR = BASE_DIR / "manthan_output" / "phase_1_audit"
ALT_REPORTS_DIR = ALT_OUTPUT_DIR / "reports"
ALT_LOGS_DIR = ALT_OUTPUT_DIR / "logs"
ALT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
ALT_LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Logger setup
log_file = LOGS_DIR / "phase_1_run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Phase1Audit")

# Expected required files
REQUIRED_FILES = {
    "train_source1": DATASET_ROOT / "train" / "train_source1.tsv",
    "train_source2": DATASET_ROOT / "train" / "train_source2.tsv",
    "train_source3": DATASET_ROOT / "train" / "train_source3.tsv",
    "train_ground_truth": DATASET_ROOT / "train" / "train_ground_truth.tsv",
    "test_source1": DATASET_ROOT / "test" / "test_source1.tsv",
    "test_source2": DATASET_ROOT / "test" / "test_source2.tsv",
    "test_source3": DATASET_ROOT / "test" / "test_source3.tsv",
}

EXPECTED_PREFIXES = {
    "train_source1": "S1-",
    "train_source2": "S2-",
    "train_source3": "S3-",
    "test_source1": "S1-",
    "test_source2": "S2-",
    "test_source3": "S3-",
}

def main():
    logger.info("Starting Phase 1 Dataset Setup and Audit...")
    logger.info(f"Dataset root resolved to: {DATASET_ROOT}")

    # Track validation checklist status
    checklist = {
        "All 7 files found": False,
        "All 7 files readable": False,
        "TSV parsing successful": False,
        "Columns inspected": False,
        "Row counts calculated": False,
        "Missing values checked": False,
        "Entity IDs checked": False,
        "Countries checked": False,
        "Duplicate IDs checked": False,
        "Ground truth checked": False,
        "Audit reports created": False
    }

    # -------------------------------------------------------------
    # STEP 1: Verify Files
    # -------------------------------------------------------------
    logger.info("=== STEP 1: VERIFYING FILES ===")
    file_status = {}
    all_found = True
    all_readable = True
    all_tsv_parsable = True

    for name, path in REQUIRED_FILES.items():
        exists = path.exists()
        size_bytes = path.stat().st_size if exists else 0
        size_mb = size_bytes / (1024 * 1024)
        openable = False
        tsv_parsable = False

        if exists:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    f.readline()
                openable = True
            except Exception as e:
                logger.error(f"Failed to open {path}: {e}")
                openable = False

            try:
                df_head = pd.read_csv(path, sep='\t', nrows=5, dtype=str, keep_default_na=False)
                tsv_parsable = True
            except Exception as e:
                logger.error(f"Failed to parse TSV {path}: {e}")
                tsv_parsable = False

        file_status[name] = {
            "path": str(path),
            "size_bytes": size_bytes,
            "size_mb": round(size_mb, 2),
            "exists": exists,
            "openable": openable,
            "tsv_parsable": tsv_parsable
        }

        logger.info(f"File {name}: Exists={exists}, Size={size_mb:.2f} MB, Openable={openable}, Parsable={tsv_parsable}, Path={path}")

        if not exists:
            all_found = False
        if not openable:
            all_readable = False
        if not tsv_parsable:
            all_tsv_parsable = False

    checklist["All 7 files found"] = all_found
    checklist["All 7 files readable"] = all_readable
    checklist["TSV parsing successful"] = all_tsv_parsable

    if not (all_found and all_readable and all_tsv_parsable):
        logger.critical("Critical file verification failed! Stopping audit.")
        sys.exit(1)

    # Data structures for accumulating audit stats
    dataset_stats = []
    missing_value_records = []
    duplicate_id_records = []
    country_distribution_records = []
    
    entity_dfs = {}
    gt_df = None

    # -------------------------------------------------------------
    # LOAD AND AUDIT DATASETS
    # -------------------------------------------------------------
    logger.info("=== LOADING DATASETS AND COMPUTING AUDIT METRICS ===")
    
    non_ascii_pattern = r'[^\x00-\x7F]'
    latin_supp_pattern = r'[\u0080-\u00FF]'
    non_english_script_pattern = r'[\u0400-\u04FF\u0900-\u097F\u0600-\u06FF\u4E00-\u9FFF\u3040-\u30FF\u0E00-\u0E7F]'

    audit_summary = {}

    for name, path in REQUIRED_FILES.items():
        t0 = time.time()
        logger.info(f"Processing {name}...")
        df = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
        t1 = time.time()
        logger.info(f"Loaded {name} in {t1-t0:.2f}s ({len(df):,} rows)")

        columns = list(df.columns)
        num_cols = len(columns)
        num_rows = len(df)
        sample_rows = df.head(3).to_dict(orient="records")

        dataset_stats.append({
            "dataset": name,
            "path": str(path),
            "size_mb": file_status[name]["size_mb"],
            "row_count": num_rows,
            "column_count": num_cols,
            "columns": ", ".join(columns)
        })

        # Missing values audit
        for col in columns:
            s = df[col]
            missing_count = int(((s == '') | (s.str.strip() == '')).sum())
            missing_pct = round((missing_count / num_rows) * 100, 4) if num_rows > 0 else 0.0
            missing_value_records.append({
                "dataset": name,
                "column": col,
                "total_rows": num_rows,
                "missing_count": missing_count,
                "missing_percentage": missing_pct
            })

        if name == "train_ground_truth":
            gt_df = df
            continue

        # Entity files processing (Source 1/2/3)
        entity_dfs[name] = df
        expected_prefix = EXPECTED_PREFIXES[name]
        
        entity_ids = df["entity_id"]
        total_ids = len(entity_ids)
        unique_ids = entity_ids.nunique()
        duplicate_ids = total_ids - unique_ids
        empty_ids = int(((entity_ids == '') | (entity_ids.str.strip() == '')).sum())
        
        # Check prefix & format using fast string methods
        valid_prefix_mask = entity_ids.str.startswith(expected_prefix)
        unexpected_prefix_count = int((~valid_prefix_mask).sum())
        
        malformed_mask = ~entity_ids.str.match(rf"^{expected_prefix}\d+$")
        malformed_ids_count = int(malformed_mask.sum())

        duplicate_id_records.append({
            "dataset": name,
            "total_entity_ids": total_ids,
            "unique_entity_ids": unique_ids,
            "duplicate_entity_ids": duplicate_ids,
            "empty_entity_ids": empty_ids,
            "unexpected_prefix_count": unexpected_prefix_count,
            "malformed_ids_count": malformed_ids_count
        })

        # Country audit
        country_counts = df["country"].value_counts(dropna=False).to_dict()
        for country_val, count in country_counts.items():
            cnt_str = country_val if country_val != '' else '<MISSING>'
            pct = round((count / num_rows) * 100, 4)
            country_distribution_records.append({
                "dataset": name,
                "country": cnt_str,
                "count": count,
                "percentage": pct
            })

        # Duplicate Business Audit
        exact_dup_rows = int(df.duplicated().sum())
        dup_names = int(df["business_name"].duplicated().sum())
        dup_addrs = int(df["business_address"].duplicated().sum())
        dup_name_addr = int(df.duplicated(subset=["business_name", "business_address"]).sum())
        dup_name_addr_country = int(df.duplicated(subset=["business_name", "business_address", "country"]).sum())

        # Fast normalized duplicate check
        norm_name = df["business_name"].str.lower().str.strip()
        norm_addr = df["business_address"].str.lower().str.strip()
        dup_norm_name_addr = int(pd.DataFrame({"n": norm_name, "a": norm_addr}).duplicated().sum())

        # Data Quality & Character Audit
        name_str = df["business_name"]
        addr_str = df["business_address"]

        # Lengths
        name_lens = name_str.str.len()
        addr_lens = addr_str.str.len()

        max_name_len = int(name_lens.max()) if len(name_lens) > 0 else 0
        min_name_len = int(name_lens.min()) if len(name_lens) > 0 else 0
        avg_name_len = round(float(name_lens.mean()), 2) if len(name_lens) > 0 else 0.0

        max_addr_len = int(addr_lens.max()) if len(addr_lens) > 0 else 0
        min_addr_len = int(addr_lens.min()) if len(addr_lens) > 0 else 0
        avg_addr_len = round(float(addr_lens.mean()), 2) if len(addr_lens) > 0 else 0.0

        extremely_long_names = int((name_lens > 150).sum())
        extremely_long_addrs = int((addr_lens > 300).sum())

        # String checks using pandas regex methods
        has_non_ascii = (name_str.str.contains(non_ascii_pattern, regex=True) | addr_str.str.contains(non_ascii_pattern, regex=True))
        has_latin_supp = (name_str.str.contains(latin_supp_pattern, regex=True) | addr_str.str.contains(latin_supp_pattern, regex=True))
        has_non_english_script = (name_str.str.contains(non_english_script_pattern, regex=True) | addr_str.str.contains(non_english_script_pattern, regex=True))
        
        # Whitespace checks
        leading_trailing_ws = (name_str.str.strip() != name_str) | (addr_str.str.strip() != addr_str)
        excessive_ws = name_str.str.contains('  ', regex=False) | addr_str.str.contains('  ', regex=False)
        tabs_or_newlines = (
            name_str.str.contains('\t', regex=False) | name_str.str.contains('\n', regex=False) | name_str.str.contains('\r', regex=False) |
            addr_str.str.contains('\t', regex=False) | addr_str.str.contains('\n', regex=False) | addr_str.str.contains('\r', regex=False)
        )
        unusual_punct = (
            name_str.str.contains('??', regex=False) | name_str.str.contains('!!', regex=False) | name_str.str.contains(',,', regex=False) | name_str.str.contains('--', regex=False) |
            addr_str.str.contains('??', regex=False) | addr_str.str.contains('!!', regex=False) | addr_str.str.contains(',,', regex=False) | addr_str.str.contains('--', regex=False)
        )

        audit_summary[name] = {
            "num_rows": num_rows,
            "num_cols": num_cols,
            "columns": columns,
            "sample_rows": sample_rows,
            "total_ids": total_ids,
            "unique_ids": unique_ids,
            "duplicate_ids": duplicate_ids,
            "empty_ids": empty_ids,
            "unexpected_prefix_count": unexpected_prefix_count,
            "malformed_ids_count": malformed_ids_count,
            "exact_dup_rows": exact_dup_rows,
            "dup_names": dup_names,
            "dup_addrs": dup_addrs,
            "dup_name_addr": dup_name_addr,
            "dup_name_addr_country": dup_name_addr_country,
            "dup_norm_name_addr": dup_norm_name_addr,
            "max_name_len": max_name_len,
            "min_name_len": min_name_len,
            "avg_name_len": avg_name_len,
            "max_addr_len": max_addr_len,
            "min_addr_len": min_addr_len,
            "avg_addr_len": avg_addr_len,
            "extremely_long_names": extremely_long_names,
            "extremely_long_addrs": extremely_long_addrs,
            "non_ascii_count": int(has_non_ascii.sum()),
            "latin_supp_count": int(has_latin_supp.sum()),
            "non_english_script_count": int(has_non_english_script.sum()),
            "leading_trailing_ws_count": int(leading_trailing_ws.sum()),
            "excessive_ws_count": int(excessive_ws.sum()),
            "tabs_or_newlines_count": int(tabs_or_newlines.sum()),
            "unusual_punct_count": int(unusual_punct.sum())
        }
        logger.info(f"Finished metrics for {name}")

    checklist["Columns inspected"] = True
    checklist["Row counts calculated"] = True
    checklist["Missing values checked"] = True
    checklist["Entity IDs checked"] = True
    checklist["Countries checked"] = True
    checklist["Duplicate IDs checked"] = True

    # -------------------------------------------------------------
    # STEP 9: GROUND TRUTH AUDIT
    # -------------------------------------------------------------
    logger.info("=== STEP 9: GROUND TRUTH AUDIT ===")
    gt_cols = list(gt_df.columns)
    gt_num_rows = len(gt_df)
    
    gt_s1_ids = gt_df["source1_entity_id"]
    gt_missing_s1_ids = int(((gt_s1_ids == '') | (gt_s1_ids.str.strip() == '')).sum())
    gt_unique_s1_ids = gt_s1_ids.nunique()
    gt_duplicate_s1_ids = gt_num_rows - gt_unique_s1_ids

    gt_matches_raw = gt_df["matched_entity_ids"]
    gt_empty_matches = int(((gt_matches_raw == '') | (gt_matches_raw.str.strip() == '')).sum())

    # Fast parsing of matched entities
    match_lists = gt_matches_raw.apply(lambda x: [i for i in x.split(',') if i] if x else [])
    match_counts = match_lists.apply(len)

    zero_match_count = int((match_counts == 0).sum())
    one_match_count = int((match_counts == 1).sum())
    multi_match_count = int((match_counts > 1).sum())

    zero_match_pct = round((zero_match_count / gt_num_rows) * 100, 4)
    one_match_pct = round((one_match_count / gt_num_rows) * 100, 4)
    multi_match_pct = round((multi_match_count / gt_num_rows) * 100, 4)

    # Detailed matched entity breakdown
    all_matched_ids = [m for sublist in match_lists for m in sublist]
    total_matched_id_references = len(all_matched_ids)
    
    s2_match_count = sum(1 for m in all_matched_ids if m.startswith("S2-"))
    s3_match_count = sum(1 for m in all_matched_ids if m.startswith("S3-"))
    other_prefix_match_count = total_matched_id_references - (s2_match_count + s3_match_count)

    # Check internal duplicate IDs in single matched list
    internal_dup_list_count = int(match_lists.apply(lambda lst: len(lst) != len(set(lst))).sum())

    # Cross-reference with train_source1 entity_ids
    train_s1_set = set(entity_dfs["train_source1"]["entity_id"])
    gt_s1_set = set(gt_s1_ids)
    s1_in_gt_not_in_s1 = len(gt_s1_set - train_s1_set)
    s1_in_s1_not_in_gt = len(train_s1_set - gt_s1_set)

    logger.info(f"Ground Truth Rows: {gt_num_rows:,}")
    logger.info(f"Ground Truth Singletons (0 matches): {zero_match_count:,} ({zero_match_pct}%)")
    logger.info(f"Ground Truth 1 match: {one_match_count:,} ({one_match_pct}%)")
    logger.info(f"Ground Truth Multi-matches (>=2): {multi_match_count:,} ({multi_match_pct}%)")

    checklist["Ground truth checked"] = True

    # -------------------------------------------------------------
    # STEP 10: SAVE AUDIT REPORTS
    # -------------------------------------------------------------
    logger.info("=== STEP 10: SAVING AUDIT REPORTS ===")

    # 1. dataset_statistics.csv
    df_stats = pd.DataFrame(dataset_stats)
    df_stats.to_csv(REPORTS_DIR / "dataset_statistics.csv", index=False)
    df_stats.to_csv(ALT_REPORTS_DIR / "dataset_statistics.csv", index=False)

    # 2. missing_values.csv
    df_missing = pd.DataFrame(missing_value_records)
    df_missing.to_csv(REPORTS_DIR / "missing_values.csv", index=False)
    df_missing.to_csv(ALT_REPORTS_DIR / "missing_values.csv", index=False)

    # 3. duplicate_id_report.csv
    df_dup_ids = pd.DataFrame(duplicate_id_records)
    df_dup_ids.to_csv(REPORTS_DIR / "duplicate_id_report.csv", index=False)
    df_dup_ids.to_csv(ALT_REPORTS_DIR / "duplicate_id_report.csv", index=False)

    # 4. country_distribution.csv
    df_countries = pd.DataFrame(country_distribution_records)
    df_countries.to_csv(REPORTS_DIR / "country_distribution.csv", index=False)
    df_countries.to_csv(ALT_REPORTS_DIR / "country_distribution.csv", index=False)

    # 5. ground_truth_audit.md
    gt_md_content = f"""# Ground Truth Audit Report (`train_ground_truth.tsv`)

## Summary Statistics
- **Total Rows**: `{gt_num_rows:,}`
- **Columns**: `{', '.join(gt_cols)}` (Count: `{len(gt_cols)}`)
- **Missing `source1_entity_id` Count**: `{gt_missing_s1_ids}`
- **Duplicate `source1_entity_id` Count**: `{gt_duplicate_s1_ids}`
- **Empty `matched_entity_ids` Count**: `{gt_empty_matches}`

## Match Distribution
- **Zero Matches (Singletons)**: `{zero_match_count:,}` (`{zero_match_pct}%`)
- **Exactly One Match**: `{one_match_count:,}` (`{one_match_pct}%`)
- **Multiple Matches (>=2)**: `{multi_match_count:,}` (`{multi_match_pct}%`)
- **Max Matches for Single S1 Entity**: `{match_counts.max()}`

## Matched Entity References
- **Total Matched Entity ID References**: `{total_matched_id_references:,}`
- **Source 2 References (`S2-`)**: `{s2_match_count:,}` ({round(s2_match_count/total_matched_id_references*100, 2) if total_matched_id_references > 0 else 0}%)
- **Source 3 References (`S3-`)**: `{s3_match_count:,}` ({round(s3_match_count/total_matched_id_references*100, 2) if total_matched_id_references > 0 else 0}%)
- **Unexpected Prefix References**: `{other_prefix_match_count}`

## Referential Integrity Checks
- **S1 IDs in Ground Truth not in `train_source1.tsv`**: `{s1_in_gt_not_in_s1}`
- **S1 IDs in `train_source1.tsv` not in Ground Truth**: `{s1_in_s1_not_in_gt}`
- **Rows with Internal Duplicate IDs in `matched_entity_ids`**: `{internal_dup_list_count}`
"""
    with open(REPORTS_DIR / "ground_truth_audit.md", "w", encoding="utf-8") as f:
        f.write(gt_md_content)
    with open(ALT_REPORTS_DIR / "ground_truth_audit.md", "w", encoding="utf-8") as f:
        f.write(gt_md_content)

    # 6. dataset_audit.md
    ds_md_lines = []
    ds_md_lines.append("# Phase 1 Dataset Audit Report — Amazon ML Challenge 2026\n")
    ds_md_lines.append(f"**Dataset Root Path**: `{DATASET_ROOT}`\n")
    ds_md_lines.append("## 1. File Verification Overview\n")
    ds_md_lines.append("| File Name | Path | Size (MB) | Exists | Openable | Parsable |")
    ds_md_lines.append("| --- | --- | --- | --- | --- | --- |")
    for fname, info in file_status.items():
        ds_md_lines.append(f"| `{fname}` | `{info['path']}` | `{info['size_mb']}` | `{info['exists']}` | `{info['openable']}` | `{info['tsv_parsable']}` |")
    
    ds_md_lines.append("\n## 2. Dataset Row & Column Summary\n")
    ds_md_lines.append("| Dataset | Total Rows | Columns Count | Columns |")
    ds_md_lines.append("| --- | --- | --- | --- |")
    for s in dataset_stats:
        ds_md_lines.append(f"| `{s['dataset']}` | `{s['row_count']:,}` | `{s['column_count']}` | `{s['columns']}` |")

    ds_md_lines.append("\n## 3. Missing Values Summary\n")
    ds_md_lines.append("| Dataset | Column | Total Rows | Missing Count | Missing % |")
    ds_md_lines.append("| --- | --- | --- | --- | --- |")
    for m in missing_value_records:
        ds_md_lines.append(f"| `{m['dataset']}` | `{m['column']}` | `{m['total_rows']:,}` | `{m['missing_count']:,}` | `{m['missing_percentage']}%` |")

    ds_md_lines.append("\n## 4. Entity ID Audit Summary\n")
    ds_md_lines.append("| Dataset | Total IDs | Unique IDs | Duplicate IDs | Empty IDs | Malformed/Unexpected Prefix |")
    ds_md_lines.append("| --- | --- | --- | --- | --- | --- |")
    for d in duplicate_id_records:
        ds_md_lines.append(f"| `{d['dataset']}` | `{d['total_entity_ids']:,}` | `{d['unique_entity_ids']:,}` | `{d['duplicate_entity_ids']:,}` | `{d['empty_entity_ids']}` | `{d['malformed_ids_count']}` |")

    ds_md_lines.append("\n## 5. Country Distribution Summary\n")
    ds_md_lines.append("| Dataset | Country | Count | Percentage |")
    ds_md_lines.append("| --- | --- | --- | --- |")
    for c in country_distribution_records:
        ds_md_lines.append(f"| `{c['dataset']}` | `{c['country']}` | `{c['count']:,}` | `{c['percentage']}%` |")

    ds_md_lines.append("\n## 6. Duplicate Business & Data Quality Audit\n")
    for dname, info in audit_summary.items():
        ds_md_lines.append(f"### `{dname}`")
        ds_md_lines.append(f"- **Exact Duplicate Rows**: `{info['exact_dup_rows']:,}`")
        ds_md_lines.append(f"- **Duplicate Business Names**: `{info['dup_names']:,}`")
        ds_md_lines.append(f"- **Duplicate Business Addresses**: `{info['dup_addrs']:,}`")
        ds_md_lines.append(f"- **Duplicate (Name + Address)**: `{info['dup_name_addr']:,}`")
        ds_md_lines.append(f"- **Duplicate (Name + Address + Country)**: `{info['dup_name_addr_country']:,}`")
        ds_md_lines.append(f"- **Duplicate Normalized (Name + Address)**: `{info['dup_norm_name_addr']:,}`")
        ds_md_lines.append(f"- **Name Lengths (Min / Max / Avg)**: `{info['min_name_len']} / {info['max_name_len']} / {info['avg_name_len']}`")
        ds_md_lines.append(f"- **Address Lengths (Min / Max / Avg)**: `{info['min_addr_len']} / {info['max_addr_len']} / {info['avg_addr_len']}`")
        ds_md_lines.append(f"- **Extremely Long Names (>150 chars)**: `{info['extremely_long_names']:,}`")
        ds_md_lines.append(f"- **Extremely Long Addresses (>300 chars)**: `{info['extremely_long_addrs']:,}`")
        ds_md_lines.append(f"- **Non-ASCII Character Rows**: `{info['non_ascii_count']:,}`")
        ds_md_lines.append(f"- **Latin-1 Supplement Rows**: `{info['latin_supp_count']:,}`")
        ds_md_lines.append(f"- **Non-English Script Rows**: `{info['non_english_script_count']:,}`")
        ds_md_lines.append(f"- **Leading / Trailing Whitespace Rows**: `{info['leading_trailing_ws_count']:,}`")
        ds_md_lines.append(f"- **Excessive Internal Spaces Rows**: `{info['excessive_ws_count']:,}`")
        ds_md_lines.append(f"- **Tabs or Newlines in Fields**: `{info['tabs_or_newlines_count']:,}`")
        ds_md_lines.append(f"- **Unusual Punctuation Rows**: `{info['unusual_punct_count']:,}`\n")

    full_md_content = "\n".join(ds_md_lines)
    with open(REPORTS_DIR / "dataset_audit.md", "w", encoding="utf-8") as f:
        f.write(full_md_content)
    with open(ALT_REPORTS_DIR / "dataset_audit.md", "w", encoding="utf-8") as f:
        f.write(full_md_content)

    checklist["Audit reports created"] = True

    # Copy audit_dataset.py into ALT_OUTPUT_DIR
    try:
        shutil.copy2(__file__, ALT_OUTPUT_DIR / "audit_dataset.py")
    except Exception:
        pass

    # -------------------------------------------------------------
    # STEP 11: AUTOMATIC VALIDATION CHECKLIST
    # -------------------------------------------------------------
    logger.info("==================================================")
    logger.info("PHASE 1 AUTOMATIC VALIDATION CHECKLIST")
    logger.info("==================================================")

    all_passed = True
    for item, status in checklist.items():
        status_str = "PASS" if status else "FAIL"
        logger.info(f"[{status_str}] {item}")
        if not status:
            all_passed = False

    logger.info("==================================================")
    if all_passed:
        logger.info("Phase 1 Dataset Audit COMPLETED SUCCESSFULLY!")
    else:
        logger.error("Phase 1 Validation FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main()
