# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import numpy as np, pandas as pd
import math

def midline(low: float, high: float) -> float:
    return (float(low) + float(high)) / 2.0

EPS: float = 1e-9

def _norm_method(method: str) -> str:
    m = str(method).strip().lower()
    return m if m in ("high", "close") else "high"

def _gt(a: float, b: float) -> bool:  # strict >
    return (a - b) > EPS

def _lt(a: float, b: float) -> bool:  # strict <
    return (b - a) > EPS

def _ge(a: float, b: float) -> bool:  # >= with eps
    return (a - b) >= -EPS

def _le(a: float, b: float) -> bool:  # <= with eps
    return (b - a) >= -EPS

def _find_c3(d: pd.DataFrame, i_c2: int, c1_low: float, c1_high: float, method: str, dir_tag: str) -> Tuple[Optional[int], Optional[int]]:
    n = len(d); c3_any_idx: Optional[int] = None
    method = _norm_method(method)
    for j in range(i_c2 + 1, n):
        H = float(d.iloc[j]["High"]); L = float(d.iloc[j]["Low"]); C = float(d.iloc[j]["Close"])
        if dir_tag == "BULL":
            cond_any = (_gt(H, c1_high)) if method == "high" else (_gt(C, c1_high))
        else:
            cond_any = (_lt(L, c1_low)) if method == "high" else (_lt(C, c1_low))
        if cond_any: c3_any_idx = j; break
    return None, c3_any_idx

def crt_scan(
    df: pd.DataFrame,
    lookback_bars: int = 30,
    require_midline: bool = False,
    strict_vs_c1open: bool = False,
    confirm_within: int = 0,
    confirm_method: str = "high",
    directions: Tuple[str, ...] = ("bullish","bearish"),
    *,
    c1_window_bars: int = 1,
    skip_dual_sweep: bool = True,
) -> List[Dict]:
    out: List[Dict] = []
    if df is None or df.empty: return out
    d = df.dropna(subset=["Open","High","Low","Close"]).copy()
    min_required = max(2, int(c1_window_bars) + 1)
    if len(d) < min_required:
        return out
    n = len(d); start_idx = max(1, n - lookback_bars)
    confirm_method = _norm_method(confirm_method)
    for i in range(start_idx, n):
        if i - 1 < 0:
            continue
        C2 = d.iloc[i]
        made_record = False
        max_back = max(1, int(c1_window_bars))
        max_candidates = max_back + 1
        lower_bound = max(0, i - max_candidates)
        for base_idx in range(i - 1, lower_bound - 1, -1):
            C1 = d.iloc[base_idx]
            if c1_window_bars <= 1:
                C1L, C1H = float(C1["Low"]), float(C1["High"])
            else:
                start_win = max(0, base_idx - (c1_window_bars - 1))
                win = d.iloc[start_win:base_idx+1]
                if win.empty or len(win) < c1_window_bars:
                    continue
                C1L, C1H = float(win["Low"].min()), float(win["High"].max())
            C2L, C2H, C2C = float(C2["Low"]), float(C2["High"]), float(C2["Close"])
            C1O = float(C1["Open"])
            C1_mid = midline(C1L, C1H)
            close_in = _ge(C2C, C1L) and _le(C2C, C1H)
            sweep_low = _lt(C2L, C1L)
            sweep_high = _gt(C2H, C1H)
            if not close_in:
                break
            if skip_dual_sweep and sweep_low and sweep_high:
                if _gt(C1H, C1L):
                    break

            def _record(direction: str, swept_side: str):
                dir_tag = "BULL" if direction == "bullish" else "BEAR"
                c3_within_idx, c3_any_idx = _find_c3(d, i, C1L, C1H, confirm_method, dir_tag)
                confirmed_within = False
                if confirm_within and (i + 1) < n:
                    end_j = min(n - 1, i + confirm_within)
                    for j in range(i + 1, end_j + 1):
                        H = float(d.iloc[j]["High"]); L = float(d.iloc[j]["Low"]); C_val = float(d.iloc[j]["Close"])
                        if dir_tag == "BULL":
                            cond = (H > C1H) if confirm_method == "high" else (C_val > C1H)
                        else:
                            cond = (L < C1L) if confirm_method == "high" else (C_val < C1L)
                        if cond:
                            confirmed_within = True
                            c3_within_idx = j
                            break
                if confirm_within:
                    if confirm_method == "high":
                        rule = "high>C1H" if dir_tag == "BULL" else "low<C1L"
                    else:
                        rule = "close>C1H" if dir_tag == "BULL" else "close<C1L"
                    confirm_rule = f"{rule} in {confirm_within}"
                else:
                    confirm_rule = "no confirm"
                out.append({
                    "direction": dir_tag,
                    "C1_date": d.index[base_idx],
                    "C2_date": d.index[i],
                    "C3_date_within": d.index[c3_within_idx] if c3_within_idx is not None else pd.NaT,
                    "C3_date_any": d.index[c3_any_idx] if c3_any_idx is not None else pd.NaT,
                    "confirmed": bool(confirmed_within),
                    "c3_happened": bool(c3_any_idx is not None),
                    "confirm_rule": confirm_rule,
                    "C1_low": C1L,
                    "C1_high": C1H,
                    "C1_mid": C1_mid,
                    "C1_open": C1O,
                    "C2_low": C2L,
                    "C2_high": C2H,
                    "C2_close": C2C,
                    "C2_position_in_range": (C2C - C1L) / (C1H - C1L) if (C1H > C1L) else np.nan,
                    "swept_side": swept_side,
                })

            if "bullish" in directions:
                cond = sweep_low
                if require_midline:
                    cond &= _ge(C2C, C1_mid)
                if strict_vs_c1open:
                    cond &= _ge(C2C, C1O)
                if cond:
                    _record("bullish", "LOW")
                    made_record = True
            if "bearish" in directions:
                cond = sweep_high
                if require_midline:
                    cond &= _le(C2C, C1_mid)
                if strict_vs_c1open:
                    cond &= _le(C2C, C1O)
                if cond:
                    _record("bearish", "HIGH")
                    made_record = True
            if made_record:
                break
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
        directional_ok = (v_used <= key_val + EPS) if direction == "BULL" else (v_used + EPS >= key_val)

    return tf_str, key_val, key_date, bool(dist_ok and directional_ok)
