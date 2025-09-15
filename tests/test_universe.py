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


def test_parse_gpw_tickers_preserves_prefix_and_suffix():
    raw = " ^wig20 , pkn.wa ,  xyz "
    df = parse_gpw_tickers(raw)
    vals = df["yahoo_ticker"].tolist()
    assert "^WIG20" in vals
    assert "PKN.WA" in vals
    assert "XYZ.WA" in vals


def test_build_universe_df_combines_sources(monkeypatch):
    import universe

    fake_wig_all = pd.DataFrame({"company": ["AAA"], "yahoo_ticker": ["AAA.WA"], "group": ["WIG"]})
    fake_wig20 = pd.DataFrame({"company": ["BBB"], "yahoo_ticker": ["BBB.WA"], "group": ["WIG20"]})
    fake_mwig40 = pd.DataFrame({"company": ["CCC"], "yahoo_ticker": ["CCC.WA"], "group": ["mWIG40"]})
    fake_sp = pd.DataFrame({"company": ["DDD"], "yahoo_ticker": ["DDD"], "group": ["S&P500"]})

    monkeypatch.setattr(universe, "get_wig_all", lambda: fake_wig_all)
    monkeypatch.setattr(universe, "get_wig20_hardcoded", lambda: fake_wig20)
    monkeypatch.setattr(universe, "get_mwig40_hardcoded", lambda: fake_mwig40)
    monkeypatch.setattr(universe, "fetch_sp500_companies", lambda: fake_sp)

    df = build_universe_df(True, True, True, True, "DIN", "AAPL")
    assert set(df["yahoo_ticker"]) == {"AAA.WA", "BBB.WA", "CCC.WA", "DDD", "DNP.WA", "AAPL"}
    assert df["Active"].all()
