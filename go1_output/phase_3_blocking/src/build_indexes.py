#!/usr/bin/env python3
"""
Phase 3 Inverted Index Builder Module
Builds inverted indexes over S2 and S3 candidate pools with posting list limits.
"""

from collections import defaultdict
import pandas as pd
from blocking_keys import get_name_keys, get_addr_keys, get_num_keys, get_prefix_keys

def build_blocking_indexes(df_s2, df_s3, max_name_posting=3000, max_addr_posting=1000, max_num_posting=1000, max_prefix_posting=1000):
    name_idx = defaultdict(set)
    addr_idx = defaultdict(set)
    num_idx = defaultdict(set)
    prefix_idx = defaultdict(set)

    def populate_index(df):
        for r in df.itertuples():
            eid = r.entity_id
            country = r.country_clean
            sig_toks = r.name_significant_tokens
            addr_toks = r.address_tokens
            addr_nums = r.address_numbers

            for key in get_name_keys(country, sig_toks):
                name_idx[key].add(eid)
                if len(key[1]) >= 4:
                    prefix_idx[(country, key[1][:4])].add(eid)

            for key in get_addr_keys(country, addr_toks):
                addr_idx[key].add(eid)

            for key in get_num_keys(country, addr_nums):
                num_idx[key].add(eid)

    populate_index(df_s2)
    populate_index(df_s3)

    token_stats = []
    for k, v in name_idx.items():
        token_stats.append({"key_type": "name_token", "country": k[0], "key": k[1], "posting_count": len(v)})
    for k, v in addr_idx.items():
        token_stats.append({"key_type": "address_token", "country": k[0], "key": k[1], "posting_count": len(v)})

    # Filter posting lists by maximum size
    filtered_name_idx = {k: v for k, v in name_idx.items() if len(v) <= max_name_posting}
    filtered_addr_idx = {k: v for k, v in addr_idx.items() if len(v) <= max_addr_posting}
    filtered_num_idx = {k: v for k, v in num_idx.items() if len(v) <= max_num_posting}
    filtered_prefix_idx = {k: v for k, v in prefix_idx.items() if len(v) <= max_prefix_posting}

    indexes = {
        "name": filtered_name_idx,
        "addr": filtered_addr_idx,
        "num": filtered_num_idx,
        "prefix": filtered_prefix_idx
    }

    return indexes, token_stats
