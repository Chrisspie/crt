# -*- coding: utf-8 -*-
import pandas as pd
from universe import parse_us_tickers, parse_gpw_tickers, build_universe_df, GPW_ALIASES

def test_alias_map_applied():
    df = parse_gpw_tickers("DIN, BUD, AMX, CIGAMES, LIVE, PEK")
    vals = set(df["yahoo_ticker"].tolist())
    assert {"DNP.WA","BDX.WA","AMC.WA","CIG.WA","LVC.WA","PBX.WA"}.issubset(vals)

def test_parse_us_passthrough():
    df = parse_us_tickers("AAPL, MSFT")
    assert df.loc[0, "yahoo_ticker"] == "AAPL"
