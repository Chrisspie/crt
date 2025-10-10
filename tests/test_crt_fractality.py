
# -*- coding: utf-8 -*-
import pandas as pd, pytest

crt_core = pytest.importorskip("crt_core")
crt_scan = crt_core.crt_scan

def df_from_rows(rows, start="2024-03-01", freq="H"):
    idx = pd.date_range(start, periods=len(rows), freq=freq)
    return pd.DataFrame(rows, columns=["Open","High","Low","Close"], index=idx)

@pytest.mark.xfail(reason="Explicit fractality invariance checks not implemented")
def test_fractality_same_logic_across_tf():
    # Same synthetic pattern once as (HTF,LTF)=(15m,1m) and (D1,15m).
    # Expect: same CRT flags/direction/confirm logic irrespective of TF scale.
    pattern = [
        [100,110,90,100],  # C1
        [95,105,89,99],    # C2 bullish sweep
        [100,113,95,112],  # confirmation by high/close
    ]
    df1 = df_from_rows(pattern, freq="15min")
    df2 = df_from_rows(pattern, freq="D")
    r1 = next((r for r in crt_scan(df1, lookback_bars=50, directions=("bullish",)) if r.get("direction")=="bullish"), None)
    r2 = next((r for r in crt_scan(df2, lookback_bars=50, directions=("bullish",)) if r.get("direction")=="bullish"), None)
    assert r1 and r2
    comparable = ["direction","confirmed","bias","C1H","C1L"]
    assert {k:r1.get(k) for k in comparable} == {k:r2.get(k) for k in comparable}
