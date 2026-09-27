# Phase 1 Dataset Audit Report — Amazon ML Challenge 2026

**Dataset Root Path**: `/Users/swapnil/Documents/ML/student_resource/dataset`

## 1. File Verification Overview

| File Name | Path | Size (MB) | Exists | Openable | Parsable |
| --- | --- | --- | --- | --- | --- |
| `train_source1` | `/Users/swapnil/Documents/ML/student_resource/dataset/train/train_source1.tsv` | `200.34` | `True` | `True` | `True` |
| `train_source2` | `/Users/swapnil/Documents/ML/student_resource/dataset/train/train_source2.tsv` | `466.63` | `True` | `True` | `True` |
| `train_source3` | `/Users/swapnil/Documents/ML/student_resource/dataset/train/train_source3.tsv` | `480.37` | `True` | `True` | `True` |
| `train_ground_truth` | `/Users/swapnil/Documents/ML/student_resource/dataset/train/train_ground_truth.tsv` | `121.13` | `True` | `True` | `True` |
| `test_source1` | `/Users/swapnil/Documents/ML/student_resource/dataset/test/test_source1.tsv` | `166.91` | `True` | `True` | `True` |
| `test_source2` | `/Users/swapnil/Documents/ML/student_resource/dataset/test/test_source2.tsv` | `485.86` | `True` | `True` | `True` |
| `test_source3` | `/Users/swapnil/Documents/ML/student_resource/dataset/test/test_source3.tsv` | `482.56` | `True` | `True` | `True` |

## 2. Dataset Row & Column Summary

| Dataset | Total Rows | Columns Count | Columns |
| --- | --- | --- | --- |
| `train_source1` | `2,206,821` | `4` | `entity_id, business_name, business_address, country` |
| `train_source2` | `5,034,616` | `4` | `entity_id, business_name, business_address, country` |
| `train_source3` | `5,285,603` | `4` | `entity_id, business_name, business_address, country` |
| `train_ground_truth` | `2,206,821` | `2` | `source1_entity_id, matched_entity_ids` |
| `test_source1` | `1,732,544` | `4` | `entity_id, business_name, business_address, country` |
| `test_source2` | `4,887,273` | `4` | `entity_id, business_name, business_address, country` |
| `test_source3` | `5,082,316` | `4` | `entity_id, business_name, business_address, country` |

## 3. Missing Values Summary

| Dataset | Column | Total Rows | Missing Count | Missing % |
| --- | --- | --- | --- | --- |
| `train_source1` | `entity_id` | `2,206,821` | `0` | `0.0%` |
| `train_source1` | `business_name` | `2,206,821` | `0` | `0.0%` |
| `train_source1` | `business_address` | `2,206,821` | `0` | `0.0%` |
| `train_source1` | `country` | `2,206,821` | `0` | `0.0%` |
| `train_source2` | `entity_id` | `5,034,616` | `0` | `0.0%` |
| `train_source2` | `business_name` | `5,034,616` | `0` | `0.0%` |
| `train_source2` | `business_address` | `5,034,616` | `168,967` | `3.3561%` |
| `train_source2` | `country` | `5,034,616` | `0` | `0.0%` |
| `train_source3` | `entity_id` | `5,285,603` | `0` | `0.0%` |
| `train_source3` | `business_name` | `5,285,603` | `0` | `0.0%` |
| `train_source3` | `business_address` | `5,285,603` | `175,916` | `3.3282%` |
| `train_source3` | `country` | `5,285,603` | `0` | `0.0%` |
| `train_ground_truth` | `source1_entity_id` | `2,206,821` | `0` | `0.0%` |
| `train_ground_truth` | `matched_entity_ids` | `2,206,821` | `123,247` | `5.5848%` |
| `test_source1` | `entity_id` | `1,732,544` | `0` | `0.0%` |
| `test_source1` | `business_name` | `1,732,544` | `0` | `0.0%` |
| `test_source1` | `business_address` | `1,732,544` | `0` | `0.0%` |
| `test_source1` | `country` | `1,732,544` | `0` | `0.0%` |
| `test_source2` | `entity_id` | `4,887,273` | `0` | `0.0%` |
| `test_source2` | `business_name` | `4,887,273` | `0` | `0.0%` |
| `test_source2` | `business_address` | `4,887,273` | `129,408` | `2.6479%` |
| `test_source2` | `country` | `4,887,273` | `0` | `0.0%` |
| `test_source3` | `entity_id` | `5,082,316` | `0` | `0.0%` |
| `test_source3` | `business_name` | `5,082,316` | `0` | `0.0%` |
| `test_source3` | `business_address` | `5,082,316` | `136,098` | `2.6779%` |
| `test_source3` | `country` | `5,082,316` | `0` | `0.0%` |

## 4. Entity ID Audit Summary

| Dataset | Total IDs | Unique IDs | Duplicate IDs | Empty IDs | Malformed/Unexpected Prefix |
| --- | --- | --- | --- | --- | --- |
| `train_source1` | `2,206,821` | `2,206,821` | `0` | `0` | `0` |
| `train_source2` | `5,034,616` | `5,034,616` | `0` | `0` | `0` |
| `train_source3` | `5,285,603` | `5,285,603` | `0` | `0` | `0` |
| `test_source1` | `1,732,544` | `1,732,544` | `0` | `0` | `0` |
| `test_source2` | `4,887,273` | `4,887,273` | `0` | `0` | `0` |
| `test_source3` | `5,082,316` | `5,082,316` | `0` | `0` | `0` |

## 5. Country Distribution Summary

| Dataset | Country | Count | Percentage |
| --- | --- | --- | --- |
| `train_source1` | `US` | `1,323,633` | `59.9792%` |
| `train_source1` | `India` | `883,188` | `40.0208%` |
| `train_source2` | `US` | `3,016,817` | `59.9215%` |
| `train_source2` | `India` | `2,017,799` | `40.0785%` |
| `train_source3` | `US` | `3,170,056` | `59.9753%` |
| `train_source3` | `India` | `2,115,547` | `40.0247%` |
| `test_source1` | `India` | `809,986` | `46.7513%` |
| `test_source1` | `US` | `663,106` | `38.2735%` |
| `test_source1` | `France` | `259,452` | `14.9752%` |
| `test_source2` | `India` | `2,312,565` | `47.3181%` |
| `test_source2` | `US` | `1,871,330` | `38.2899%` |
| `test_source2` | `France` | `703,378` | `14.392%` |
| `test_source3` | `India` | `2,405,000` | `47.3209%` |
| `test_source3` | `US` | `1,945,701` | `38.2837%` |
| `test_source3` | `France` | `731,615` | `14.3953%` |

## 6. Duplicate Business & Data Quality Audit

### `train_source1`
- **Exact Duplicate Rows**: `0`
- **Duplicate Business Names**: `667,592`
- **Duplicate Business Addresses**: `76,215`
- **Duplicate (Name + Address)**: `0`
- **Duplicate (Name + Address + Country)**: `0`
- **Duplicate Normalized (Name + Address)**: `0`
- **Name Lengths (Min / Max / Avg)**: `3 / 105 / 24.03`
- **Address Lengths (Min / Max / Avg)**: `11 / 256 / 52.07`
- **Extremely Long Names (>150 chars)**: `0`
- **Extremely Long Addresses (>300 chars)**: `0`
- **Non-ASCII Character Rows**: `554`
- **Latin-1 Supplement Rows**: `554`
- **Non-English Script Rows**: `0`
- **Leading / Trailing Whitespace Rows**: `0`
- **Excessive Internal Spaces Rows**: `968`
- **Tabs or Newlines in Fields**: `0`
- **Unusual Punctuation Rows**: `106`

### `train_source2`
- **Exact Duplicate Rows**: `0`
- **Duplicate Business Names**: `632,607`
- **Duplicate Business Addresses**: `697,354`
- **Duplicate (Name + Address)**: `25,891`
- **Duplicate (Name + Address + Country)**: `25,873`
- **Duplicate Normalized (Name + Address)**: `36,650`
- **Name Lengths (Min / Max / Avg)**: `2 / 104 / 25.1`
- **Address Lengths (Min / Max / Avg)**: `0 / 249 / 46.23`
- **Extremely Long Names (>150 chars)**: `0`
- **Extremely Long Addresses (>300 chars)**: `0`
- **Non-ASCII Character Rows**: `1,106,720`
- **Latin-1 Supplement Rows**: `291,422`
- **Non-English Script Rows**: `479,500`
- **Leading / Trailing Whitespace Rows**: `0`
- **Excessive Internal Spaces Rows**: `649,810`
- **Tabs or Newlines in Fields**: `0`
- **Unusual Punctuation Rows**: `19,410`

### `train_source3`
- **Exact Duplicate Rows**: `0`
- **Duplicate Business Names**: `633,994`
- **Duplicate Business Addresses**: `652,838`
- **Duplicate (Name + Address)**: `18,881`
- **Duplicate (Name + Address + Country)**: `18,860`
- **Duplicate Normalized (Name + Address)**: `24,031`
- **Name Lengths (Min / Max / Avg)**: `2 / 123 / 25.2`
- **Address Lengths (Min / Max / Avg)**: `0 / 240 / 46.71`
- **Extremely Long Names (>150 chars)**: `0`
- **Extremely Long Addresses (>300 chars)**: `0`
- **Non-ASCII Character Rows**: `989,672`
- **Latin-1 Supplement Rows**: `329,179`
- **Non-English Script Rows**: `394,374`
- **Leading / Trailing Whitespace Rows**: `0`
- **Excessive Internal Spaces Rows**: `579,914`
- **Tabs or Newlines in Fields**: `0`
- **Unusual Punctuation Rows**: `20,053`

### `test_source1`
- **Exact Duplicate Rows**: `0`
- **Duplicate Business Names**: `493,677`
- **Duplicate Business Addresses**: `55,061`
- **Duplicate (Name + Address)**: `0`
- **Duplicate (Name + Address + Country)**: `0`
- **Duplicate Normalized (Name + Address)**: `0`
- **Name Lengths (Min / Max / Avg)**: `3 / 92 / 23.84`
- **Address Lengths (Min / Max / Avg)**: `11 / 268 / 57.21`
- **Extremely Long Names (>150 chars)**: `0`
- **Extremely Long Addresses (>300 chars)**: `0`
- **Non-ASCII Character Rows**: `101,466`
- **Latin-1 Supplement Rows**: `100,511`
- **Non-English Script Rows**: `0`
- **Leading / Trailing Whitespace Rows**: `0`
- **Excessive Internal Spaces Rows**: `753`
- **Tabs or Newlines in Fields**: `0`
- **Unusual Punctuation Rows**: `123`

### `test_source2`
- **Exact Duplicate Rows**: `0`
- **Duplicate Business Names**: `576,232`
- **Duplicate Business Addresses**: `662,489`
- **Duplicate (Name + Address)**: `22,642`
- **Duplicate (Name + Address + Country)**: `22,641`
- **Duplicate Normalized (Name + Address)**: `32,012`
- **Name Lengths (Min / Max / Avg)**: `2 / 102 / 25.7`
- **Address Lengths (Min / Max / Avg)**: `0 / 269 / 50.41`
- **Extremely Long Names (>150 chars)**: `0`
- **Extremely Long Addresses (>300 chars)**: `0`
- **Non-ASCII Character Rows**: `1,450,645`
- **Latin-1 Supplement Rows**: `505,763`
- **Non-English Script Rows**: `550,530`
- **Leading / Trailing Whitespace Rows**: `0`
- **Excessive Internal Spaces Rows**: `573,995`
- **Tabs or Newlines in Fields**: `0`
- **Unusual Punctuation Rows**: `14,534`

### `test_source3`
- **Exact Duplicate Rows**: `0`
- **Duplicate Business Names**: `560,387`
- **Duplicate Business Addresses**: `625,880`
- **Duplicate (Name + Address)**: `16,305`
- **Duplicate (Name + Address + Country)**: `16,293`
- **Duplicate Normalized (Name + Address)**: `21,856`
- **Name Lengths (Min / Max / Avg)**: `2 / 103 / 25.66`
- **Address Lengths (Min / Max / Avg)**: `0 / 267 / 48.74`
- **Extremely Long Names (>150 chars)**: `0`
- **Extremely Long Addresses (>300 chars)**: `0`
- **Non-ASCII Character Rows**: `1,316,658`
- **Latin-1 Supplement Rows**: `548,399`
- **Non-English Script Rows**: `454,885`
- **Leading / Trailing Whitespace Rows**: `0`
- **Excessive Internal Spaces Rows**: `522,351`
- **Tabs or Newlines in Fields**: `0`
- **Unusual Punctuation Rows**: `15,324`
