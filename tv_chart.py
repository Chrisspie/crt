# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any
import pandas as pd, plotly.graph_objects as go

def build_plotly_chart(d: pd.DataFrame, rec: Dict[str, Any], ticker: str) -> go.Figure:
    fig = go.Figure(data=[go.Candlestick(x=d.index, open=d['Open'], high=d['High'], low=d['Low'], close=d['Close'], name=ticker)])
    def add_hline(y, text, dash="dot"):
        try:
            if y is not None and not pd.isna(y):
                fig.add_hline(y=float(y), line_dash=dash, annotation_text=text, annotation_position="left")
        except Exception:
            pass
    add_hline(rec.get("Trigger"), "Trigger", "solid")
    add_hline(rec.get("Stop"), "SL", "dot")
    add_hline(rec.get("TP1"), "TP1", "dash")
    add_hline(rec.get("TP2"), "TP2", "dash")
    add_hline(rec.get("KeyLevel"), "Key", "dot")
    c2_date = pd.to_datetime(rec.get("C2"), errors="coerce")
    if pd.notna(c2_date) and c2_date in d.index:
        y_base = float(d.loc[c2_date, "Close"]); x_point = c2_date
    else:
        y_base = float(d["Close"].iloc[-1]); x_point = d.index[-1]
    direction = rec.get("Kierunek")
    if direction == "BULL":
        fig.add_annotation(x=x_point, y=y_base, xref="x", yref="y", text="↑ BULL", showarrow=True, arrowhead=2, ax=0, ay=-60)
    elif direction == "BEAR":
        fig.add_annotation(x=x_point, y=y_base, xref="x", yref="y", text="↓ BEAR", showarrow=True, arrowhead=2, ax=0, ay=60)
    fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), xaxis_rangeslider_visible=False, height=520)
    return fig

def yahoo_to_tv_symbol(yahoo_ticker: str) -> str:
    """Map a Yahoo Finance ticker to TradingView symbol notation.
    - GPW tickers use suffix ".WA" on Yahoo and prefix "GPW:" on TV.
    - Otherwise, pass through unchanged (TV accepts many US symbols directly).
    """
    if not yahoo_ticker:
        return ""
    t = str(yahoo_ticker).strip().upper()
    if t.endswith(".WA"):
        return f"GPW:{t[:-3]}"
    return t

def tradingview_embed_html(symbol: str, *, height: int = 520, interval: str = "W") -> str:
    """Return a minimal TradingView widget embed HTML string.
    The HTML includes the tv.js script and a widget with the given symbol.
    Tests only assert presence of the symbol and the substring "tv.js".
    """
    safe_sym = str(symbol).strip()
    safe_interval = str(interval).strip().upper() or "W"
    h = int(height) if height and int(height) > 0 else 520
    return (
        f"<div class=\"tradingview-widget-container\" style=\"height:{h}px\">"
        f"<div id=\"tv_widget\"></div>"
        f"<script type=\"text/javascript\" src=\"https://s3.tradingview.com/tv.js\"></script>"
        f"<script type=\"text/javascript\">"
        f"new TradingView.widget({{\n"
        f"  symbol: '{safe_sym}',\n"
        f"  interval: '{safe_interval}',\n"
        f"  container_id: 'tv_widget',\n"
        f"  width: '100%',\n"
        f"  height: '{h}',\n"
        f"  hide_top_toolbar: false,\n"
        f"  hide_legend: false,\n"
        f"  allow_symbol_change: true\n"
        f"}});\n"
        f"</script></div>"
    )
