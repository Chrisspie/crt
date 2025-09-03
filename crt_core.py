# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import numpy as np, pandas as pd

def midline(low: float, high: float) -> float:
    return (float(low) + float(high)) / 2.0

def _find_c3(d: pd.DataFrame, i_c2: int, c1_low: float, c1_high: float, method: str, dir_tag: str) -> Tuple[Optional[int], Optional[int]]:
    n = len(d); c3_any_idx: Optional[int] = None
    for j in range(i_c2 + 1, n):
        H = float(d.iloc[j]["High"]); L = float(d.iloc[j]["Low"]); C = float(d.iloc[j]["Close"])
        if dir_tag == "BULL":
            cond_any = (H > c1_high) if method == "high" else (C > c1_high)
        else:
            cond_any = (L < c1_low) if method == "high" else (C < c1_low)
        if cond_any: c3_any_idx = j; break
    return None, c3_any_idx

def crt_scan(df: pd.DataFrame, lookback_bars: int = 30, require_midline: bool = False, strict_vs_c1open: bool = False,
             confirm_within: int = 0, confirm_method: str = "high", directions: Tuple[str, ...] = ("bullish","bearish")) -> List[Dict]:
    out: List[Dict] = []
    if df is None or df.empty or len(df) < 5: return out
    d = df.dropna(subset=["Open","High","Low","Close"]).copy()
    if len(d) < 5: return out
    n = len(d); start_idx = max(1, n - lookback_bars)
    for i in range(start_idx, n):
        C1, C2 = d.iloc[i-1], d.iloc[i]
        C1L, C1H = float(C1["Low"]), float(C1["High"])
        C2L, C2H, C2C, C1O = float(C2["Low"]), float(C2["High"]), float(C2["Close"]), float(C1["Open"])
        C1_mid = midline(C1L, C1H); close_in = (C1L <= C2C <= C1H)
        def _record(direction: str, swept_side: str):
            dir_tag = "BULL" if direction=="bullish" else "BEAR"
            c3_within_idx, c3_any_idx = _find_c3(d, i, C1L, C1H, confirm_method, dir_tag)
            confirmed_within = False
            if confirm_within and (i+1) < n:
                end_j = min(n-1, i+confirm_within)
                for j in range(i+1, end_j+1):
                    H = float(d.iloc[j]['High']); L = float(d.iloc[j]['Low']); C = float(d.iloc[j]['Close'])
                    if dir_tag == "BULL":
                        cond = (H > C1H) if confirm_method == "high" else (C > C1H)
                    else:
                        cond = (L < C1L) if confirm_method == "high" else (C < C1L)
                    if cond: confirmed_within=True; c3_within_idx=j; break
            confirm_rule = "no confirm"
            if confirm_within:
                if confirm_method == "high":
                    rule = "high>C1H" if dir_tag=="BULL" else "low<C1L"
                else:
                    rule = "close>C1H" if dir_tag=="BULL" else "close<C1L"
                confirm_rule = f"{rule} in {confirm_within}"
            out.append({
                "direction": dir_tag, "C1_date": d.index[i-1], "C2_date": d.index[i],
                "C3_date_within": d.index[c3_within_idx] if c3_within_idx is not None else pd.NaT,
                "C3_date_any": d.index[c3_any_idx] if c3_any_idx is not None else pd.NaT,
                "confirmed": bool(confirmed_within), "c3_happened": bool(c3_any_idx is not None),
                "confirm_rule": confirm_rule,
                "C1_low": C1L, "C1_high": C1H, "C1_mid": C1_mid, "C1_open": C1O,
                "C2_low": C2L, "C2_high": C2H, "C2_close": C2C,
                "C2_position_in_range": (C2C - C1L) / (C1H - C1L) if (C1H > C1L) else np.nan,
                "swept_side": swept_side
            })
        if "bullish" in directions:
            cond = (C2L < C1L) and close_in
            if require_midline: cond &= (C2C >= C1_mid)
            if strict_vs_c1open: cond &= (C2C >= C1O)
            if cond: _record("bullish","LOW")
        if "bearish" in directions:
            cond = (C2H > C1H) and close_in
            if require_midline: cond &= (C2C <= C1_mid)
            if strict_vs_c1open: cond &= (C2C <= C1O)
            if cond: _record("bearish","HIGH")
    out.sort(key=lambda r: (pd.Timestamp(r["C2_date"]) if pd.notna(r["C2_date"]) else pd.Timestamp(0)), reverse=True)
    return out

def get_key_level_and_confluence(htf_df: pd.DataFrame, c2_ts: pd.Timestamp, direction: str,
    c1_low: float, c1_high: float, c2_low: float, c2_high: float, key_window_months: int,
    key_interact: str, key_strict: str, htf_interval: str):
    if htf_df is None or htf_df.empty or c2_ts is None or pd.isna(c2_ts): return "-", float("nan"), pd.NaT, False
    bars_per = 1 if htf_interval=="1mo" else 3
    import math
    n_bars = max(1, math.ceil(key_window_months / bars_per))
    htf_hist = htf_df[htf_df.index <= c2_ts]
    if htf_hist.empty: return "-", float("nan"), pd.NaT, False
    win = htf_hist.tail(n_bars)
    if direction == "BULL":
        key_level_val = float(win["Low"].min()); key_date = win["Low"].idxmin()
        value = float(c1_low) if key_interact=="Tylko C1" else (float(c2_low) if key_interact=="Tylko C2" else min(float(c1_low), float(c2_low)))
        confluence = (value < key_level_val) if key_strict.startswith("strict") else (value <= key_level_val)
        return ("1M" if htf_interval=="1mo" else "3M"), key_level_val, key_date, confluence
    else:
        key_level_val = float(win["High"].max()); key_date = win["High"].idxmax()
        value = float(c1_high) if key_interact=="Tylko C1" else (float(c2_high) if key_interact=="Tylko C2" else max(float(c1_high), float(c2_high)))
        confluence = (value > key_level_val) if key_strict.startswith("strict") else (value >= key_level_val)
        return ("1M" if htf_interval=="1mo" else "3M"), key_level_val, key_date, confluence
