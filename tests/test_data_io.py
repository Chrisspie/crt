# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, yfinance as yf
from pandas import MultiIndex
import data_io

def test_load_many_weekly_ohlcv_multi(monkeypatch):
    idx = pd.date_range("2024-01-05", periods=5, freq="W-FRI")
    cols = MultiIndex.from_product([["AAA","BBB"], ["Open","High","Low","Close"]])
    df = pd.DataFrame(np.random.rand(len(idx), len(cols)), index=idx, columns=cols)
    def fake_download(tickers, *args, **kwargs): return df
    monkeypatch.setattr(yf, "download", fake_download)
    out = data_io.load_many_weekly_ohlcv(["AAA","BBB"], period="2y")
    assert set(out.keys()) >= {"AAA","BBB"}

def test_stringio_read_html(monkeypatch):
    # ensure we pass something through without raising
    import requests
    html = "<table><tr><th>Symbol</th><th>Name</th></tr><tr><td>AAA</td><td>Alpha</td></tr></table>"
    def fake_get(url, headers=None, timeout=20):
        class R: text = html
        return R()
    monkeypatch.setattr(requests, "get", fake_get)
    df = data_io.fetch_wiki_table(["http://example.com"], "IDX")
    assert "yahoo_ticker" in df.columns or df.empty
