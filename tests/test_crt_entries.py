
# -*- coding: utf-8 -*-
import pandas as pd, pytest

crt_core = pytest.importorskip("crt_core")
crt_scan = crt_core.crt_scan

def df_from_rows(rows, start="2024-03-01", freq="5min"):
    idx = pd.date_range(start, periods=len(rows), freq=freq)
    return pd.DataFrame(rows, columns=["Open","High","Low","Close"], index=idx)

# ----------------------------
# 1) Standard LTF Entry: ChoCh/BOS -> OB/FVG
# ----------------------------
@pytest.mark.xfail(reason="LTF entry model (ChoCh/BOS -> OB/FVG) not implemented")
def test_ltf_entry_standard_model():
    # HTF sweep assumed; we pass a hint to focus on LTF sequence.
    rows = [
        [100,110,90,100],   # pre
        [95,105,89,99],     # C2 bullish on HTF (hint)
        [100,103,97,102],   # ChoCh up on LTF
        [101,106,100,105],  # FVG + OB creation
    ]
    df = df_from_rows(rows)
    recs = crt_scan(df, lookback_bars=50, directions=("bullish",), htf_sweep_hint=True, require_ltf_entry=True)
    r = next((r for r in recs if r.get("direction")=="bullish"), None)
    assert r and r.get("entry_model") == "standard" and "EP" in r and "SL" in r

# ----------------------------
# 2) Knot-ChoCh exception inside HTF POI
# ----------------------------
@pytest.mark.xfail(reason="Knot-ChoCh exception not implemented")
def test_knot_choc_exception_requires_htf_poi():
    rows = [
        [100,110,90,100],
        [95,105,89,99],     # C2 bullish
        [100,105,95,100],   # wick-based ChoCh only
    ]
    df = df_from_rows(rows)
    recs = crt_scan(df, lookback_bars=50, directions=("bullish",), htf_poi_hint=True, allow_knot_choc=True, require_ltf_entry=True)
    r = next((r for r in recs if r.get("direction")=="bullish"), None)
    assert r and r.get("entry_model") == "knot_choc_exception"

# ----------------------------
# 3) Propulsion Block (PB) entry
# ----------------------------
@pytest.mark.xfail(reason="Propulsion Block entry not implemented")
def test_propulsion_block_entry_after_escape():
    rows = [
        [100,110,90,100],
        [95,105,89,99],     # C2 bullish
        [100,115,100,114],  # escape without tagging Extreme OB
        [112,118,111,117],  # PB formation
    ]
    df = df_from_rows(rows)
    recs = crt_scan(df, lookback_bars=50, directions=("bullish",), require_ltf_entry=True, detect_pb=True)
    r = next((r for r in recs if r.get("direction")=="bullish"), None)
    assert r and r.get("entry_model") == "propulsion_block"

# ----------------------------
# 4) CISD (Change In State of Delivery)
# ----------------------------
@pytest.mark.xfail(reason="CISD entry model not implemented")
@pytest.mark.parametrize("side", ["long","short"])
def test_cisd_fast_entry(side):
    # Long: last bearish -> bullish engulfing; Short: last bullish -> bearish engulfing
    if side == "long":
        rows = [
            [100,110,90,100],     # context
            [95,105,89,99],       # C2 bullish
            [100,102,95,95],      # last bearish candle
            [95,106,94,105],      # bullish engulfing -> CISD
        ]
    else:
        rows = [
            [100,110,90,100],
            [105,111,99,106],     # C2 bearish (sweep up, close inside)
            [100,105,99,105],     # last bullish candle
            [106,95,94,96],       # bearish engulfing -> CISD
        ]
    df = df_from_rows(rows, freq="5min")
    recs = crt_scan(df, lookback_bars=100, directions=("bullish" if side=="long" else "bearish",),
                    htf_poi_hint=True, require_ltf_entry=True, allow_cisd=True)
    r = next((r for r in recs if r.get("direction")==("bullish" if side=="long" else "bearish")), None)
    assert r and r.get("entry_model") == "CISD" and r.get("entry_execution") == "market_on_close_engulfing"
