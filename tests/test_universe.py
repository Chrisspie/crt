# -*- coding: utf-8 -*-
import pandas as pd
from universe import parse_us_tickers, parse_gpw_tickers, build_universe_df, GPW_ALIASES

def test_alias_map_applied():
    df = parse_gpw_tickers("DIN, BUD, AMX, CIGAMES, LIVE, PEK")
    vals = set(df["yahoo_ticker"].tolist())
    # Expect mapped tickers to .WA
    assert "DNP.WA" in vals and "BDX.WA" in vals and "AMC.WA" in vals and "CIG.WA" in vals and "LVC.WA" in vals and "PBX.WA" in vals

def test_parse_us_passthrough():
    df = parse_us_tickers("AAPL, MSFT")
    assert df.loc[0, "yahoo_ticker"] == "AAPL"
