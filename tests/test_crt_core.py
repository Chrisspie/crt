# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
from crt_core import crt_scan, get_key_level_and_confluence, midline

def _weekly_df():
    idx = pd.date_range("2024-01-05", periods=8, freq="W-FRI")
    data = {"Open":[10,10.5,10,10.2,9.8,9.4,9.6,9.7],
            "High":[11,11.2,10.8,10.6,10,9.9,10.2,10.1],
            "Low":[9,10,9.6,9.7,9.2,9.0,9.3,9.4],
            "Close":[10,10.1,10.2,10,9.9,9.6,9.8,9.9]}
    return pd.DataFrame(data, index=idx)

def test_midline(): assert midline(10,12)==11.0

def test_crt_scan_small_lookback():
    d=_weekly_df()
    setups = crt_scan(d, lookback_bars=3, confirm_within=2, confirm_method="high")
    assert isinstance(setups, list)

def test_key_level_touch_bull():
    idx = pd.date_range("2023-01-31", periods=12, freq="M")
    htf=pd.DataFrame({"Open":np.linspace(10,9,len(idx)),
                      "High":np.linspace(11,12,len(idx)),
                      "Low":np.linspace(9,8,len(idx)),
                      "Close":np.linspace(10,9.5,len(idx))}, index=idx)
    tf, kl, kd, conf = get_key_level_and_confluence(htf, pd.to_datetime("2023-12-29"), "BULL",
        9.2, 10.0, 9.0, 9.9, 6, "C1 lub C2", "touch (≤, ≥)", "1mo")
    assert tf in ("1M","3M") and isinstance(kl, float)
