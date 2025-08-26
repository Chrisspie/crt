import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple


def midline(low: float, high: float) -> float:
    return (float(low) + float(high)) / 2.0


def _find_c3(
    d: pd.DataFrame,
    i_c2: int,
    c1_low: float,
    c1_high: float,
    method: str,
    dir_tag: str,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Zwraca: (c3_idx_within_N, c3_idx_to_end)
    Tu wykrywamy 'do końca serii' (any) i zwracamy indeks C3 (jeśli wystąpił).
    """
    n = len(d)
    c3_any_idx = None

    for j in range(i_c2 + 1, n):
        H = float(d.iloc[j]["High"])
        L = float(d.iloc[j]["Low"])
        C = float(d.iloc[j]["Close"])
        if dir_tag == "BULL":
            cond_any = (H > c1_high) if method == "high" else (C > c1_high)
        else:  # BEAR
            cond_any = (L < c1_low) if method == "high" else (C < c1_low)
        if cond_any:
            c3_any_idx = j
            break

    return None, c3_any_idx


def crt_scan(
    df: pd.DataFrame,
    lookback_bars: int = 30,
    require_midline: bool = False,
    strict_vs_c1open: bool = False,
    confirm_within: int = 0,            # 0 = brak potwierdzenia C3
    confirm_method: str = "high",       # 'high' (knot) lub 'close' (zamknięcie)
    directions: Tuple[str, ...] = ("bullish", "bearish"),
) -> List[Dict]:
    """
    Zwraca listę setupów CRT (C1/C2/C3 + metadane).
    """
    out: List[Dict] = []
    if df is None or df.empty or len(df) < 5:
        return out

    d = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if len(d) < 5:
        return out

    n = len(d)
    start_idx = max(1, n - lookback_bars)

    for i in range(start_idx, n):
        C1 = d.iloc[i - 1]
        C2 = d.iloc[i]
        C1L, C1H = float(C1["Low"]), float(C1["High"])
        C2L, C2H, C2C, C1O = float(C2["Low"]), float(C2["High"]), float(C2["Close"]), float(C1["Open"])
        C1_mid = midline(C1L, C1H)
        close_in = (C1L <= C2C <= C1H)

        def _record(direction: str, swept_side: str):
            dir_tag = "BULL" if direction == "bullish" else "BEAR"
            c3_within_idx, c3_any_idx = _find_c3(d, i, C1L, C1H, confirm_method, dir_tag)

            confirmed_within = False
            if confirm_within and confirm_within > 0 and (i + 1) < n:
                end_j = min(n - 1, i + confirm_within)
                for j in range(i + 1, end_j + 1):
                    H = float(d.iloc[j]["High"])
                    L = float(d.iloc[j]["Low"])
                    C = float(d.iloc[j]["Close"])
                    if dir_tag == "BULL":
                        cond = (H > C1H) if confirm_method == "high" else (C > C1H)
                    else:  # BEAR
                        cond = (L < C1L) if confirm_method == "high" else (C < C1L)
                    if cond:
                        confirmed_within = True
                        c3_within_idx = j
                        break

            rec = {
                "direction": dir_tag,
                "C1_date": d.index[i - 1],
                "C2_date": d.index[i],
                "C3_date_within": d.index[c3_within_idx] if c3_within_idx is not None else pd.NaT,
                "C3_date_any": d.index[c3_any_idx] if c3_any_idx is not None else pd.NaT,
                "confirmed": bool(confirmed_within),
                "c3_happened": bool(c3_any_idx is not None),
                # Uwaga: reguła tekstowa zachowana zgodnie z main.py (bez zmiany znaku dla BEAR)
                "confirm_rule": (
                    f"{confirm_method}>{'C1H' if direction=='bullish' else 'C1L'} in {confirm_within}"
                    if confirm_within else "no confirm"
                ),
                "C1_low": C1L, "C1_high": C1H, "C1_mid": C1_mid, "C1_open": C1O,
                "C2_low": C2L, "C2_high": C2H, "C2_close": C2C,
                "C2_position_in_range": (C2C - C1L) / (C1H - C1L) if (C1H > C1L) else np.nan,
                "swept_side": swept_side,
            }
            out.append(rec)

        # Byczy: sweep LOW C1 + close back in range (+opcje)
        if "bullish" in directions:
            cond = (C2L < C1L) and close_in
            if require_midline:
                cond &= (C2C >= C1_mid)
            if strict_vs_c1open:
                cond &= (C2C >= C1O)
            if cond:
                _record("bullish", "LOW")

        # Niedźwiedzi: sweep HIGH C1 + close back in range (+opcje)
        if "bearish" in directions:
            cond = (C2H > C1H) and close_in
            if require_midline:
                cond &= (C2C <= C1_mid)
            if strict_vs_c1open:
                cond &= (C2C <= C1O)
            if cond:
                _record("bearish", "HIGH")

    out.sort(
        key=lambda r: (pd.Timestamp(r["C2_date"]) if pd.notna(r["C2_date"]) else pd.Timestamp(0)),
        reverse=True,
    )
    return out

