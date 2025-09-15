# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, yfinance as yf
from pandas import MultiIndex
import data_io

def test_load_many_weekly_ohlcv_multi(monkeypatch):
    data_io.clear_all_cached_data()
    idx = pd.date_range("2024-01-05", periods=5, freq="W-FRI")
    cols = MultiIndex.from_product([["AAA","BBB"], ["Open","High","Low","Close"]])
    df = pd.DataFrame(np.random.rand(len(idx), len(cols)), index=idx, columns=cols)
    def fake_download(tickers, *args, **kwargs): return df
    monkeypatch.setattr(yf, "download", fake_download)
    out = data_io.load_many_weekly_ohlcv(["AAA","BBB"], period="2y")
    assert set(out.keys()) >= {"AAA","BBB"}

def test_load_many_weekly_ohlcv_reports_missing(monkeypatch):
    data_io.clear_all_cached_data()
    idx = pd.date_range("2024-01-05", periods=4, freq="W-FRI")
    cols = MultiIndex.from_product([["AAA"], ["Open","High","Low","Close"]])
    df = pd.DataFrame(np.random.rand(len(idx), len(cols)), index=idx, columns=cols)

    def fake_download(tickers, *args, **kwargs):
        assert sorted(tickers) == ["AAA", "BBB"]
        return df

    monkeypatch.setattr(yf, "download", fake_download)
    out = data_io.load_many_weekly_ohlcv(["AAA", "BBB"], period="1y", retries=0)
    assert "AAA" in out and isinstance(out["AAA"], pd.DataFrame)
    assert "BBB" not in out
    assert "BBB" in out["__failed__"].tolist()


def test_load_many_weekly_ohlcv_bulk_retry(monkeypatch):
    data_io.clear_all_cached_data()
    idx = pd.date_range("2024-01-05", periods=3, freq="W-FRI")
    base = pd.DataFrame(
        {
            "Open": np.linspace(10, 12, len(idx)),
            "High": np.linspace(11, 13, len(idx)),
            "Low": np.linspace(9, 11, len(idx)),
            "Close": np.linspace(10, 12, len(idx)),
        },
        index=idx,
    )
    calls: list[str] = []

    def fake_download(tickers, *args, **kwargs):
        if isinstance(tickers, list):
            raise RuntimeError("bulk fail")
        calls.append(tickers)
        return base

    monkeypatch.setattr(yf, "download", fake_download)
    out = data_io.load_many_weekly_ohlcv(["AAA", "BBB"], period="2y", retries=2)
    assert set(out.keys()) >= {"AAA", "BBB", "__failed__"}
    assert sorted(calls) == ["AAA", "BBB"]
    for ticker in ["AAA", "BBB"]:
        assert isinstance(out[ticker], pd.DataFrame)
        pd.testing.assert_index_equal(out[ticker].index, idx)


def test_load_many_weekly_single_ticker(monkeypatch):
    data_io.clear_all_cached_data()
    idx = pd.date_range("2024-01-05", periods=3, freq="W-FRI")
    df = pd.DataFrame(
        {
            "Open": [1, 2, 3],
            "High": [2, 3, 4],
            "Low": [0.5, 1.5, 2.5],
            "Close": [1.5, 2.5, 3.5],
        },
        index=idx,
    )

    def fake_download(tickers, *args, **kwargs):
        assert tickers == ["AAA"]
        return df

    monkeypatch.setattr(yf, "download", fake_download)
    out = data_io.load_many_weekly_ohlcv(["AAA"], period="1y")
    assert "AAA" in out and isinstance(out["AAA"], pd.DataFrame)
    assert out["__failed__"].empty


def test_load_many_htf_multi(monkeypatch):
    data_io.clear_all_cached_data()
    idx = pd.date_range("2023-01-31", periods=3, freq="ME")
    cols = MultiIndex.from_product([["AAA"], ["Open", "High", "Low", "Close"]])
    df = pd.DataFrame(np.random.rand(len(idx), len(cols)), index=idx, columns=cols)

    def fake_download(tickers, *args, **kwargs):
        return df

    monkeypatch.setattr(yf, "download", fake_download)
    out = data_io.load_many_htf_ohlcv(["AAA"], interval="1mo", period="max")
    assert "AAA" in out and isinstance(out["AAA"], pd.DataFrame)


def test_stringio_read_html(monkeypatch):
    import requests
    html = "<table><tr><th>Symbol</th><th>Name</th></tr><tr><td>AAA</td><td>Alpha</td></tr></table>"
    def fake_get(url, headers=None, timeout=20):
        class R: text = html
        return R()
    monkeypatch.setattr(requests, "get", fake_get)
    df = data_io.fetch_wiki_table(["http://example.com"], "IDX")
    assert "yahoo_ticker" in df.columns or df.empty


def test_fetch_wiki_table_returns_empty_on_failure(monkeypatch):
    import requests

    def fake_get(url, headers=None, timeout=20):
        raise requests.RequestException("boom")

    monkeypatch.setattr(requests, "get", fake_get)
    data_io.fetch_wiki_table.clear()
    df = data_io.fetch_wiki_table(["http://example.com"], "IDX")
    assert df.empty


def test_load_many_weekly_ohlcv_stooq_fallback(monkeypatch):
    data_io.clear_all_cached_data()

    def fake_download(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr(yf, "download", fake_download)

    csv = """Date,Open,High,Low,Close,Volume\n2023-12-22,10,11,9,10.5,1000\n2024-01-05,11,12,10,11.5,1200\n"""

    class FakeResp:
        ok = True

        def __init__(self, text):
            self.text = text

    calls: list[str] = []

    def fake_get(url, headers=None, timeout=15):
        calls.append(url)
        return FakeResp(csv)

    monkeypatch.setattr(data_io.requests, "get", fake_get)

    out = data_io.load_many_weekly_ohlcv(["ALE.WA", "CDR.WA"], start="2024-01-01", period="5y", retries=0)

    assert set(out.keys()) >= {"ALE.WA", "CDR.WA", "__failed__"}
    assert calls  # ensure fallback attempted
    for ticker in ["ALE.WA", "CDR.WA"]:
        df = out[ticker]
        assert not df.empty
        assert (df.index >= pd.Timestamp("2024-01-01")).all()
        assert list(df.columns) == ["Open", "High", "Low", "Close"]
    assert out["__failed__"].empty


def test_load_many_htf_ohlcv_stooq_fallback(monkeypatch):
    data_io.clear_all_cached_data()

    def fake_download(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr(yf, "download", fake_download)

    csv = """Date,Open,High,Low,Close,Volume\n2024-01-31,100,110,95,108,10000\n2024-02-29,108,120,100,115,9000\n"""

    class FakeResp:
        ok = True

        def __init__(self, text):
            self.text = text

    def fake_get(url, headers=None, timeout=15):
        return FakeResp(csv)

    monkeypatch.setattr(data_io.requests, "get", fake_get)

    out = data_io.load_many_htf_ohlcv(["PKN.WA"], interval="1mo", period="max", retries=0)

    assert set(out.keys()) >= {"PKN.WA", "__failed__"}
    df = out["PKN.WA"]
    assert not df.empty
    assert list(df.columns) == ["Open", "High", "Low", "Close"]
    assert out["__failed__"].empty
