#!/usr/bin/env python3
"""
Phase 3 Blocking Keys Generation Functions
Provides functions to generate blocking keys from cleaned entity fields.
"""

def get_name_keys(country, name_sig_toks_str):
    if not country or not name_sig_toks_str:
        return []
    keys = []
    for t in name_sig_toks_str.split():
        if len(t) >= 2:
            keys.append((country, t))
    return keys

def get_addr_keys(country, addr_toks_str):
    if not country or not addr_toks_str:
        return []
    keys = []
    for t in addr_toks_str.split():
        if len(t) >= 3:
            keys.append((country, t))
    return keys

def get_num_keys(country, addr_nums_str):
    if not country or not addr_nums_str:
        return []
    keys = []
    for num in addr_nums_str.split():
        if len(num) >= 2:
            keys.append((country, num))
    return keys

def get_prefix_keys(country, name_sig_toks_str):
    if not country or not name_sig_toks_str:
        return []
    keys = []
    for t in name_sig_toks_str.split():
        if len(t) >= 4:
            keys.append((country, t[:4]))
    return keys
