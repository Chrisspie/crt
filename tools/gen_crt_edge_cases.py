# -*- coding: utf-8 -*-
"""Generate CRT edge-case sequences as JSON.
Run:
    python tools/gen_crt_edge_cases.py > crt_edge_cases.json
"""
from __future__ import annotations
import json
from typing import List, Dict, Any

def case(name: str, rows: List[List[float]], settings: Dict[str, Any], notes: str) -> Dict[str, Any]:
    return {
        "name": name,
        "rows": rows,
        "settings": settings,
        "notes": notes,
    }

def build_cases() -> List[Dict[str, Any]]:
    EPS = 1e-9
    cases: List[Dict[str, Any]] = []

    # 1) Basic BULL C2
    cases.append(case(
        "bull_basic_no_confirm",
        [
            [100,110,90,100],  # C1
            [95,105,89,100],   # C2 bull
            [100,109,95,100],  # neutral
        ],
        {"confirm_within": 0, "confirm_method": "high", "directions": ("bullish",)},
        "Bullish sweep of C1 low with close back inside; no confirmation requested."
    ))

    # 2) Basic BEAR C2
    cases.append(case(
        "bear_basic_no_confirm",
        [
            [100,110,90,100],  # C1
            [100,112,95,100],  # C2 bear
            [100,109,95,100],  # neutral
        ],
        {"confirm_within": 0, "confirm_method": "high", "directions": ("bearish",)},
        "Bearish sweep of C1 high with close back inside; no confirmation requested."
    ))

    # 3) Bull confirm by high within 1
    cases.append(case(
        "bull_confirm_by_high_within_1",
        [
            [100,110,90,100],  # C1
            [95,105,89,100],   # C2 bull
            [100,111,95,100],  # j=i+1 confirm by high
        ],
        {"confirm_within": 1, "confirm_method": "high", "directions": ("bullish",)},
        "Confirmation bar with High > C1_high occurs within 1 bar."
    ))

    # 4) Bear confirm by close within 2
    cases.append(case(
        "bear_confirm_by_close_within_2",
        [
            [100,110,90,100],  # C1
            [100,112,95,105],  # C2 bear
            [100,110,95,110],  # j=i+1 not confirming by close
            [100,110,88,89],   # j=i+2 confirm by close < C1_low
        ],
        {"confirm_within": 2, "confirm_method": "close", "directions": ("bearish",)},
        "Confirmation by close < C1_low on i+2; boundary touch at i+1 should not confirm."
    ))

    # 5) Equality edges
    cases.append(case(
        "equality_edges",
        [
            [100,110,90,100],  # C1
            [100,110,89,110],  # C2: Close == C1H (inside), High==C1H (no sweep)
        ],
        {"confirm_within": 0, "confirm_method": "high", "directions": ("bullish","bearish")},
        "Close equal to boundary is inside; High equal to boundary is not a sweep."
    ))

    # 6) Dual sweep (skip by default)
    cases.append(case(
        "dual_sweep_default_skip",
        [
            [100,110,90,100],  # C1
            [100,115,85,100],  # C2: dual sweep
        ],
        {"confirm_within": 0, "confirm_method": "high", "directions": ("bullish","bearish"), "skip_dual_sweep": True},
        "Dual sweep with close inside; should be skipped if skip_dual_sweep=True."
    ))

    # 7) Dual sweep (allow both)
    cases.append(case(
        "dual_sweep_allow_both",
        [
            [100,110,90,100],  # C1
            [100,115,85,100],  # C2: dual sweep
        ],
        {"confirm_within": 0, "confirm_method": "high", "directions": ("bullish","bearish"), "skip_dual_sweep": False},
        "Dual sweep with close inside; allow emitting both BULL and BEAR."
    ))

    # 8) Gap up / BEAR
    cases.append(case(
        "gap_up_bear",
        [
            [100,110,90,100],
            [120,125,100,105],
        ],
        {"confirm_within": 0, "confirm_method": "high", "directions": ("bearish",)},
        "Gap up open; swept above C1H; close back inside."
    ))

    # 9) Gap down / BULL
    cases.append(case(
        "gap_down_bull",
        [
            [100,110,90,100],
            [80,100,85,95],
        ],
        {"confirm_within": 0, "confirm_method": "high", "directions": ("bullish",)},
        "Gap down open; swept below C1L; close back inside."
    ))

    # 10) Narrow range C1
    cases.append(case(
        "narrow_c1_range",
        [
            [100,100,100,100],
            [100,101,99,100],
        ],
        {"confirm_within": 0, "confirm_method": "high", "directions": ("bullish","bearish")},
        "C1 has zero range; C2_position_in_range should be NaN in outputs."
    ))

    # 11) NaN inside series
    cases.append(case(
        "nan_inside_series",
        [
            [100,110,90,100],
            [float('nan'), float('nan'), float('nan'), float('nan')],
            [100,110,90,100],  # C1
            [95,105,89,100],   # C2 bull
        ],
        {"confirm_within": 0, "confirm_method": "high", "directions": ("bullish",)},
        "NaN row should be dropped; detection proceeds."
    ))

    # 12) require_midline strict
    cases.append(case(
        "require_midline_bull",
        [
            [100,110,90,90],   # C1 -> midline 100
            [100,109,89,101],  # C2 close >= midline
        ],
        {"confirm_within": 0, "confirm_method": "high", "directions": ("bullish",), "require_midline": True},
        "Require C2 close to be above midline for BULL."
    ))

    # 13) strict_vs_c1open bear
    cases.append(case(
        "strict_vs_c1open_bear",
        [
            [120,130,110,120],  # C1 Open=120
            [100,112,95,121],   # C2: close 121 > C1O -> should be blocked in BEAR strict_vs_c1open
        ],
        {"confirm_within": 0, "confirm_method": "high", "directions": ("bearish",), "strict_vs_c1open": True},
        "For BEAR with strict_vs_c1open, require C2 close <= C1 Open."
    ))

    # 14) directions filter
    cases.append(case(
        "directions_filter_bull_only",
        [
            [100,110,90,100],
            [95,105,89,100],
        ],
        {"confirm_within": 0, "confirm_method": "high", "directions": ("bullish",)},
        "Only bullish direction should be emitted."
    ))

    # 15) confirm_method case-insensitive 'CLOSE'
    cases.append(case(
        "confirm_method_close_case_insensitive",
        [
            [100,110,90,100],
            [95,105,89,100],
            [100,111,95,120],
        ],
        {"confirm_within": 1, "confirm_method": "CLOSE", "directions": ("bullish",)},
        "Uppercase CLOSE should behave as close method."
    ))

    return cases

def main():
