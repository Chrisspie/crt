
# -*- coding: utf-8 -*-
import pandas as pd, pytest

crt_core = pytest.importorskip("crt_core")
crt_scan = crt_core.crt_scan

def df_from_rows(rows, start="2024-03-01", freq="H"):
    idx = pd.date_range(start, periods=len(rows), freq=freq)
    return pd.DataFrame(rows, columns=["Open","High","Low","Close"], index=idx)

@pytest.mark.xfail(reason="Manipulation Block pattern not implemented")
def test_manipulation_block_pattern_detected():
    # Pattern: false BOS by body close beyond bound, next candle is engulfing -> treat as MB
    rows = [
        [100,110,90,100],    # C1
        [95,111,89,109],     # C2 (bearish sweep fail -> close just beyond? then next engulfing flips)
        [110,100,89,95],     # engulfing reversal -> MB
    ]
    df = df_from_rows(rows)
    recs = crt_scan(df, lookback_bars=20, directions=("bearish","bullish"), detect_manipulation_block=True)
    # Expect a special tag regardless of final direction
    assert any(r.get("pattern") == "ManipulationBlock" for r in recs)
