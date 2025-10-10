
# -*- coding: utf-8 -*-
import math
import pandas as pd
import pytest

crt_core = pytest.importorskip("crt_core")
crt_scan = crt_core.crt_scan

def df_from_rows(rows, start="2025-01-01 00:00:00", freq="H"):
    idx = pd.date_range(start, periods=len(rows), freq=freq)
    return pd.DataFrame(rows, columns=["Open","High","Low","Close"], index=idx)

def approx(a, b, eps=1e-9):
    return abs(a-b) < eps

@pytest.mark.parametrize("direction,c1,c2", [
    # Bearish case: sweep HIGH then close back inside -> short
    ("bearish",
     [100, 110,  90, 100],   # C1 (C1H=110, C1L=90)
     [105, 111,  95, 104]    # C2 makes higher high (111) but closes inside (104)
    ),
    # Bullish case: sweep LOW then close back inside -> long
    ("bullish",
     [100, 110,  90, 100],   # C1
     [ 95, 105,  89,  96]    # C2 makes lower low (89) but closes inside (96)
    ),
])
def test_tp_contract_midline_and_opposite_bound(direction, c1, c2):
    rows = [
        [100,101,99,100],  # pre
        c1,                # C1
        c2,                # C2 (sweep + close-in-range)
        [100,120,98,110],  # follow-through bar
    ]
    df = df_from_rows(rows)

    recs = crt_scan(
        df,
        lookback_bars=50,
        directions=(direction,),
        return_targets=True,   # implementation should populate TP fields when requested
    )
    r = next((r for r in recs if r.get("direction")==direction), None)
    assert r is not None, "CRT record not produced"

    C1H = float(r.get("C1H"))
    C1L = float(r.get("C1L"))
    mid = (C1H + C1L)/2.0

    # Required by spec:
    # - tp1 must equal midline
    # - tp_min must be the opposite bound of C1 (CRL for short, CRH for long)
    assert "tp1" in r, "Missing tp1 in record"
    assert "tp_min" in r, "Missing tp_min in record"
    assert approx(float(r["tp1"]), mid), f"tp1 should equal midline {mid}, got {r['tp1']}"

    if direction == "bearish":
        assert approx(float(r["tp_min"]), C1L), f"tp_min for bearish must equal C1L {C1L}, got {r['tp_min']}"
        # sanity: tp1 (mid) should be above tp_min (C1L)
        assert float(r["tp1"]) > float(r["tp_min"]), "tp1 (midline) should be above tp_min (C1L) for shorts"
    else:
        assert approx(float(r["tp_min"]), C1H), f"tp_min for bullish must equal C1H {C1H}, got {r['tp_min']}"
        # sanity: tp1 (mid) should be below tp_min (C1H)
        assert float(r["tp1"]) < float(r["tp_min"]), "tp1 (midline) should be below tp_min (C1H) for longs"

def test_no_confusion_between_entry_and_targets():
    # Ensure implementation distinguishes entry (EP) from targets.
    # Build a bearish pattern again
    c1 = [100, 110, 90, 100]
    c2 = [105, 111, 95, 104]
    df = df_from_rows([[100,101,99,100], c1, c2, [100,120,98,110]])

    recs = crt_scan(df, lookback_bars=50, directions=("bearish",), require_ltf_entry=False, return_targets=True)
    r = next((r for r in recs if r.get("direction")=="bearish"), None)
    assert r is not None

    # Targets present
    assert "tp1" in r and "tp_min" in r

    # If Entry Price is provided, it should be separate from targets
    # Accept either 'EP' or 'entry_price' as the canonical field name.
    ep = r.get("EP", r.get("entry_price", None))
    if ep is not None:
        assert float(ep) not in (float(r["tp1"]), float(r["tp_min"])), "Entry price must not duplicate TP values"
