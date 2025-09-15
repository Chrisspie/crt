# -*- coding: utf-8 -*-
import pandas as pd
from tv_chart import yahoo_to_tv_symbol, tradingview_embed_html, build_plotly_chart

def test_symbol_map_pl():
    assert yahoo_to_tv_symbol("PKN.WA") == "GPW:PKN"

def test_symbol_map_trims_and_uppercases():
    assert yahoo_to_tv_symbol("  msft ") == "MSFT"

def test_tv_html_contains_symbol():
    html = tradingview_embed_html("GPW:PKN", height=400, interval="D")
    assert "GPW:PKN" in html and "tv.js" in html

def test_tv_html_invalid_height_defaults():
    html = tradingview_embed_html("GPW:PKN", height=-10, interval="w")
    assert "height:520px" in html
    assert "interval: 'W'" in html

def test_build_plotly_chart_annotations():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    daily = pd.DataFrame({"Open":[1,2,3,4,5,6,7,8,9,10],
                          "High":[2,3,4,5,6,7,8,9,10,11],
                          "Low":[0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5],
                          "Close":[1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5]}, index=idx)
    rec = {"Kierunek":"BULL","Stop":1.0,"TP1":6.0,"TP2":8.0,"KeyLevel":0.9,"C2":"2024-01-07"}
    fig = build_plotly_chart(daily, rec, "TEST")
    assert len(fig.layout.annotations) >= 1 and hasattr(fig.layout, "shapes")


def test_build_plotly_chart_bearish_annotation():
    idx = pd.date_range("2024-02-01", periods=5, freq="D")
    daily = pd.DataFrame({
        "Open": [10, 11, 12, 13, 14],
        "High": [11, 12, 13, 14, 15],
        "Low": [9, 10, 11, 12, 13],
        "Close": [10.5, 11.5, 12.5, 13.5, 14.5],
    }, index=idx)
    rec = {"Kierunek": "BEAR", "C2": idx[2].date()}
    fig = build_plotly_chart(daily, rec, "TST")
    assert any("BEAR" in ann.text for ann in fig.layout.annotations)
