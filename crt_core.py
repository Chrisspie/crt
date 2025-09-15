# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import numpy as np, pandas as pd
import math

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
            # clearer, direction-aware confirm rule text (match tests: lowercase 'close'/'high'/'low')
            if confirm_within:
                if confirm_method == "high":
                    rule = "high>C1H" if dir_tag=="BULL" else "low<C1L"
                else:
                    rule = "close>C1H" if dir_tag=="BULL" else "close<C1L"
                confirm_rule = f"{rule} in {confirm_within}"
            else:
                confirm_rule = "no confirm"
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

# --- crt_core.py ---
def get_key_level_and_confluence(
    htf_df: pd.DataFrame,
    c2_ts: pd.Timestamp,
    direction: str,
    c1_low: float, c1_high: float,
    c2_low: float, c2_high: float,
    key_window_months: int,
    key_interact: str,
    key_rule_label: str,
    htf_interval: str
):
    """
    HTF confluence:
      - Kandydaci (per świeca HTF):
          BULL: Low, Close, Open
          BEAR: High, Close, Open
      - v (punkt kontaktu):
          'Tylko C1'      -> C1L / C1H
          'Tylko C2'      -> C2L / C2H
          'C1 lub C2'     -> wybierz (C1 vs C2), który daje mniejszy znormalizowany dystans do poziomu HTF
      - Tolerancja:
          touch  -> 1.5% * |key|
          strict -> 0.5%  * |key|  + warunek kierunkowy (BULL: v <= key, BEAR: v >= key)
    """
    tf_str = "1M" if htf_interval == "1mo" else "3M"

    if htf_df is None or htf_df.empty or c2_ts is None or pd.isna(c2_ts):
        return tf_str, float("nan"), pd.NaT, False

    bars_per = 1 if htf_interval == "1mo" else 3
    n_bars = max(1, math.ceil(float(key_window_months) / float(bars_per)))

    htf_hist = htf_df[htf_df.index <= c2_ts]
    if htf_hist.empty:
        return tf_str, float("nan"), pd.NaT, False

    win = htf_hist.tail(n_bars)

    # Zbuduj listę kandydatów poziomów z okna HTF
    cands = []
    for idx_, row in win.iterrows():
        if direction == "BULL":
            cands.extend([(float(row["Low"]), idx_), (float(row["Close"]), idx_), (float(row["Open"]), idx_)])
        else:
            cands.extend([(float(row["High"]), idx_), (float(row["Close"]), idx_), (float(row["Open"]), idx_)])

    # Zbiór możliwych 'v' zależnie od ustawienia
    v_candidates = []
    if key_interact == "Tylko C1":
        v_candidates.append(float(c1_low) if direction == "BULL" else float(c1_high))
    elif key_interact == "Tylko C2":
        v_candidates.append(float(c2_low) if direction == "BULL" else float(c2_high))
    else:  # "C1 lub C2" -> sprawdzimy obie, wybierzemy lepszą
        v_candidates.append(float(c1_low) if direction == "BULL" else float(c1_high))
        v_candidates.append(float(c2_low) if direction == "BULL" else float(c2_high))

    best = None  # (dist_norm, key_val, key_date, v_used)
    for v in v_candidates:
        # dobierz najbliższy poziom HTF do TEGO v
        best_val, best_date, best_dist = float("nan"), pd.NaT, float("inf")
        for val, dt_ in cands:
            if pd.isna(val):
                continue
            dist = abs(v - val)
            if dist < best_dist:
                best_val, best_date, best_dist = val, dt_, dist
        if pd.isna(best_val):
            continue
        denom = max(1.0, abs(best_val))
        dist_norm = best_dist / denom
        if (best is None) or (dist_norm < best[0]):
            best = (dist_norm, float(best_val), best_date, v)

    if best is None:
        return tf_str, float("nan"), pd.NaT, False

    dist_norm, key_val, key_date, v_used = best
    is_strict = str(key_rule_label).lower().startswith("strict")

    tol_pct = 0.005 if is_strict else 0.015  # 0.5% vs 1.5%
    dist_ok = dist_norm <= tol_pct

    directional_ok = True
    if is_strict:
        directional_ok = (v_used <= key_val) if direction == "BULL" else (v_used >= key_val)

    return tf_str, key_val, key_date, bool(dist_ok and directional_ok)
