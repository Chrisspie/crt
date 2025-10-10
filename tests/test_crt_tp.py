
# -*- coding: utf-8 -*-
import pandas as pd, pytest

crt_core = pytest.importorskip("crt_core")
crt_scan = crt_core.crt_scan
midline = getattr(crt_core, "midline", None)

def df_from_rows(rows, start="2024-03-01", freq="H"):
    idx = pd.date_range(start, periods=len(rows), freq=freq)
    return pd.DataFrame(rows, columns=["Open","High","Low","Close"], index=idx)

# ----------------------------
# 1) TP1 at 50% of C1 range
# ----------------------------
@pytest.mark.xfail(reason="TP1=50% auto-calculation not implemented")
def test_tp1_midline_is_suggested():
    rows = [
        [100,110,90,100],   # C1
        [95,105,89,99],     # C2 bullish
        [100,104,96,101],   # continuation
    ]
    df = df_from_rows(rows)
    recs = crt_scan(df, lookback_bars=10, directions=("bullish",), return_targets=True)
    r = next((r for r in recs if r.get("direction")=="bullish"), None)
    assert r and abs(r.get("tp1") - ((90+110)/2.0)) < 1e-9

# ----------------------------
# 2) Final target = opposite bound (CRH/CRL)
# ----------------------------
@pytest.mark.xfail(reason="Final opposite-bound target not implemented")
def test_tp_min_is_opposite_bound():
    rows = [
        [100,110,90,100],   # C1
        [95,105,89,99],     # C2 bullish
    ]
    df = df_from_rows(rows)
    recs = crt_scan(df, lookback_bars=10, directions=("bullish",), return_targets=True)
    r = next((r for r in recs if r.get("direction")=="bullish"), None)
    assert r and r.get("tp_min") == 110.0

# ----------------------------
# 3) STDV projections for advanced targets (2–2.5 and 4)
# ----------------------------
@pytest.mark.xfail(reason="STDV target projections not implemented")
def test_stdv_targets_available():
    rows = [
        [100,110,90,100],
        [95,105,89,99],
        [100,120,98,118],   # impulse after confirmation
    ]
    df = df_from_rows(rows)
    recs = crt_scan(df, lookback_bars=20, directions=("bullish",), return_targets=True, stdv_targets=[2.0,2.5,4.0])
    r = next((r for r in recs if r.get("direction")=="bullish"), None)
    assert r and set([2.0,2.5,4.0]).issubset(set(r.get("stdv_targets", {}).keys()))
