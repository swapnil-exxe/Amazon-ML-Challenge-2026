# Ground Truth Audit Report (`train_ground_truth.tsv`)

## Summary Statistics
- **Total Rows**: `2,206,821`
- **Columns**: `source1_entity_id, matched_entity_ids` (Count: `2`)
- **Missing `source1_entity_id` Count**: `0`
- **Duplicate `source1_entity_id` Count**: `0`
- **Empty `matched_entity_ids` Count**: `123247`

## Match Distribution
- **Zero Matches (Singletons)**: `123,247` (`5.5848%`)
- **Exactly One Match**: `119,157` (`5.3995%`)
- **Multiple Matches (>=2)**: `1,964,417` (`89.0157%`)
- **Max Matches for Single S1 Entity**: `11`

## Matched Entity References
- **Total Matched Entity ID References**: `7,638,365`
- **Source 2 References (`S2-`)**: `3,693,619` (48.36%)
- **Source 3 References (`S3-`)**: `3,944,746` (51.64%)
- **Unexpected Prefix References**: `0`

## Referential Integrity Checks
- **S1 IDs in Ground Truth not in `train_source1.tsv`**: `0`
- **S1 IDs in `train_source1.tsv` not in Ground Truth**: `0`
- **Rows with Internal Duplicate IDs in `matched_entity_ids`**: `0`
