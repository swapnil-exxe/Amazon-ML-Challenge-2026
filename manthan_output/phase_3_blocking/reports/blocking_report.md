# Phase 3 Blocking & Candidate Generation Report — Amazon ML Challenge 2026

## 1. Executive Summary

- **Training S1 Entities Processed**: `2,206,821`

- **Test S1 Entities Processed**: `1,732,544`

- **Ground Truth Pair-Level Candidate Recall**: `60.5225%`

- **Ground Truth S1-Level Complete Coverage**: `40.6828%`

- **Missed True Matches**: `3,015,432`

- **Total Training Candidates Generated**: `453,453,659`

- **Total Test Candidates Generated**: `401,025,269`


## 2. Candidate Distribution Statistics

| Split | Total S1 | Total Candidates | Mean | Median | P90 | P95 | P99 | Maximum | Zero Candidates | >100 Candidates | >500 Candidates | >1000 Candidates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Training` | `2,206,821` | `453,453,659` | `205.48` | `239.0` | `350.0` | `350.0` | `350.0` | `350` | `345,851` | `1,492,753` | `0` | `0` |
| `Test` | `1,732,544` | `401,025,269` | `231.47` | `332.0` | `350.0` | `350.0` | `350.0` | `350` | `178,892` | `1,276,421` | `0` | `0` |

## 3. Storage & Disk Space Capacity Analysis

- **Candidate TSV File Size (Train)**: `~5.2 GB`

- **Candidate TSV File Size (Test)**: `~4.1 GB`

- **Total Storage Required**: `~9.3 GB`

- **Current System Disk Free Space**: `17.0 GiB` (`/dev/disk3s5`)

- **Margin Remaining**: `~7.7 GiB` free space preserved safely for Arya ML feature engineering.


## 4. Multi-Pass Blocking Rules Implemented

1. **Country Partitioning**: Searches strictly restricted to identical `country_clean` labels.

2. **Significant Name Token Inverted Index**: Matches entities sharing significant name tokens (max posting size = 1,500).

3. **Address Token Inverted Index**: Matches entities sharing location/street tokens (max posting size = 500).

4. **Address Number Inverted Index**: Matches entities sharing street/unit numbers (max posting size = 500).

5. **Name Prefix Fallback**: 4-character prefix matching for sparse candidate lists (max posting size = 500).
