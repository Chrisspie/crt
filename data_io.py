# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Tuple
import pandas as pd, numpy as np, yfinance as yf, requests, streamlit as st
from io import StringIO

HEADERS = {"User-Agent": "Mozilla/5.0"}
YF_THREADS = True

WIG20_HC_TICKERS = ["PKN","KGH","PKO","PEO","PZU","PGE","CDR","ALE","DNP","LPP","OPL","CPS","ALR","ING","MBK","TPE","JSW","CCC","KTY"]
MWIG40_HC_TICKERS = ["XTB","PLW","TEN","KRU","GPW","BDX","BHW","LWB","AMC","ASB","11B","CIG","AFR","MLG","STP","PKP","MAB","NEU","OPN","VRG","WPL","WRT","DOM","MRC","PHN","PEK","IPF","TIM","MFO","PBX","BRS","FTE","BUD","APS","DVR","TOR","CIGAMES","LIVE","AMX","DIN"]
SP500_FALLBACK = ["AAPL","MSFT","NVDA","GOOGL","GOOG","AMZN","META","BRK-B","AVGO","TSLA","UNH","LLY","XOM","JPM","V","JNJ","PG","MA","HD","COST","MRK","ABBV","PEP","KO","TMO","BAC","WMT","ADBE","NFLX","CRM","CSCO","INTC","QCOM","NKE","LIN","ACN","MCD","ORCL","TXN","AMD"]

@st.cache_data(ttl=60*60*24, show_spinner=False)
def get_wig20_hardcoded() -> pd.DataFrame:
    df = pd.DataFrame({"gpw_ticker": sorted(set(WIG20_HC_TICKERS))})
    df["company"] = df["gpw_ticker"]; df["yahoo_ticker"] = df["gpw_ticker"].astype(str)+".WA"; df["group"]="WIG20"
    return df[["company","gpw_ticker","yahoo_ticker","group"]]

@st.cache_data(ttl=60*60*24, show_spinner=False)
def get_mwig40_hardcoded() -> pd.DataFrame:
    df = pd.DataFrame({"gpw_ticker": sorted(set(MWIG40_HC_TICKERS))})
    df["company"] = df["gpw_ticker"]; df["yahoo_ticker"] = df["gpw_ticker"].astype(str)+".WA"; df["group"]="mWIG40"
    return df[["company","gpw_ticker","yahoo_ticker","group"]]

@st.cache_data(ttl=60*60*24, show_spinner=False)
def fetch_wiki_table(urls: List[str], idx_name: str) -> pd.DataFrame:
    for url in urls:
        try:
            html = requests.get(url, headers=HEADERS, timeout=20).text
            tables = pd.read_html(StringIO(html))  # avoid FutureWarning
            best=None
            for t in tables:
                cols=[str(c).lower() for c in t.columns]
                if any(any(x in c for x in ["ticker","symbol","kod"]) for c in cols): best=t.copy(); break
                if t.shape[1]>=2 and best is None: best=t.copy()
            if best is None or best.empty: continue
            best.columns=[str(c).strip() for c in best.columns]
            name_candidates=[c for c in best.columns if any(x in c.lower() for x in ["spółka","company","nazwa"])]
            tick_candidates=[c for c in best.columns if any(x in c.lower() for x in ["ticker","symbol","kod"])]
            name_col=name_candidates[0] if name_candidates else best.columns[0]
            tick_col=tick_candidates[0] if tick_candidates else (best.columns[1] if best.shape[1]>1 else best.columns[0])
            out=best[[name_col,tick_col]].copy(); out.columns=["company","gpw_ticker"]
            out["company"]=out["company"].astype(str).str.strip()
            out["gpw_ticker"]=(out["gpw_ticker"].astype(str).str.upper().str.strip()
                               .str.replace(r"\s+","",regex=True).str.replace(r"[^A-Z0-9]","",regex=True))
            out=out.dropna(subset=["gpw_ticker"]).drop_duplicates(subset=["gpw_ticker"])
            out["yahoo_ticker"]=out["gpw_ticker"].astype(str)+".WA"; out["group"]=idx_name
            return out.reset_index(drop=True)
        except Exception: continue
    return pd.DataFrame(columns=["company","gpw_ticker","yahoo_ticker","group"])

@st.cache_data(ttl=60*60*24, show_spinner=False)
def fetch_swig80() -> pd.DataFrame:
    return fetch_wiki_table(["https://pl.wikipedia.org/wiki/SWIG80","https://en.wikipedia.org/wiki/SWIG80"], "sWIG80")

@st.cache_data(ttl=60*60*24, show_spinner=False)
def get_wig_all() -> pd.DataFrame:
    frames=[]
    for df in [get_wig20_hardcoded(), get_mwig40_hardcoded(), fetch_swig80()]:
        if df is not None and not df.empty: frames.append(df[["yahoo_ticker"]])
    if not frames: return pd.DataFrame(columns=["company","yahoo_ticker","group"])
    tick=pd.concat(frames,ignore_index=True).drop_duplicates()
    out=pd.DataFrame({"yahoo_ticker": tick["yahoo_ticker"]}); out["company"]=out["yahoo_ticker"].str.replace(".WA","",regex=False); out["group"]="WIG"
    return out[["company","yahoo_ticker","group"]]

@st.cache_data(ttl=7*24*60*60, show_spinner=False)
def fetch_sp500_companies() -> pd.DataFrame:
    try:
        html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=HEADERS, timeout=20).text
        tables = pd.read_html(StringIO(html)); table=None
        for t in tables:
            cols=[str(c).lower() for c in t.columns]
            if any("symbol" in c for c in cols): table=t; break
        if table is None or table.empty: raise ValueError("No 'Symbol' column")
        sym=table[table.columns[0]].astype(str).str.strip().str.upper().str.replace(".","-",regex=False)
        names=table[table.columns[1]].astype(str)
        df=pd.DataFrame({"company":names,"yahoo_ticker":sym}); df["group"]="S&P500"; return df[["company","yahoo_ticker","group"]]
    except Exception:
        df=pd.DataFrame({"yahoo_ticker":SP500_FALLBACK}); df["company"]=df["yahoo_ticker"]; df["group"]="S&P500"; return df[["company","yahoo_ticker","group"]]

@st.cache_data(ttl=60*60*12, show_spinner=False)
def load_weekly_ohlcv(yahoo_ticker: str, period: str = "5y") -> pd.DataFrame:
    df = yf.download(yahoo_ticker, interval="1wk", period=period, auto_adjust=False, progress=False, threads=YF_THREADS)
    if isinstance(df.columns, pd.MultiIndex): df.columns=df.columns.get_level_values(0)
    if not df.empty: df=df.dropna(subset=["Open","High","Low","Close"])
    return df

def _extract_single(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["Open","High","Low","Close"] if c in df.columns]
    if not cols: return pd.DataFrame()
    return df[cols].dropna(subset=cols)

@st.cache_data(ttl=60*60*12, show_spinner=False)
def load_many_weekly_ohlcv(tickers: list[str], *, period: str = "5y", start: str | None = None, retries: int = 1) -> Dict[str, pd.DataFrame]:
    if not tickers: return {}
    uniq = sorted(set([t for t in tickers if t]))
    if not uniq: return {}
    kwargs = dict(interval="1wk", auto_adjust=False, progress=False, threads=True)
    try:
        df = yf.download(uniq, start=start, **kwargs) if start else yf.download(uniq, period=period, **kwargs)
    except Exception:
        df = pd.DataFrame()
    out: Dict[str, pd.DataFrame] = {}
    missing = set(uniq)
    if isinstance(df.columns, pd.MultiIndex):
        top = df.columns.get_level_values(0)
        for t in uniq:
            if t in top:
                sub = _extract_single(df[t])
                if not sub.empty:
                    out[t] = sub; 
                    if t in missing: missing.remove(t)
    elif not df.empty and len(uniq)==1:
        sub = _extract_single(df)
        if not sub.empty:
            out[uniq[0]] = sub; missing.discard(uniq[0])
    # Retry individually for missing tickers
    if missing and retries>0:
        for t in list(missing):
            try:
                df1 = yf.download(t, start=start, **kwargs) if start else yf.download(t, period=period, **kwargs)
                sub = _extract_single(df1)
                if not sub.empty:
                    out[t] = sub; missing.discard(t)
            except Exception:
                pass
    # Return dict plus list of failed for UI
    out["__failed__"] = pd.Series(sorted(missing)) if missing else pd.Series([], dtype=str)
    return out

@st.cache_data(ttl=60*60*12, show_spinner=False)
def load_htf_ohlcv(yahoo_ticker: str, interval: str = "1mo", period: str = "max") -> pd.DataFrame:
    df = yf.download(yahoo_ticker, interval=interval, period=period, auto_adjust=False, progress=False, threads=YF_THREADS)
    if isinstance(df.columns, pd.MultiIndex): df.columns=df.columns.get_level_values(0)
    if not df.empty: df=df.dropna(subset=["Open","High","Low","Close"])
    return df

@st.cache_data(ttl=60*60*12, show_spinner=False)
def load_many_htf_ohlcv(tickers: list[str], *, interval: str = "1mo", period: str = "max", retries: int = 1) -> Dict[str, pd.DataFrame]:
    if not tickers: return {}
    uniq = sorted(set([t for t in tickers if t]))
    if not uniq: return {}
    kwargs = dict(interval=interval, auto_adjust=False, progress=False, threads=True)
    try:
        df = yf.download(uniq, period=period, **kwargs)
    except Exception:
        df = pd.DataFrame()
    out: Dict[str, pd.DataFrame] = {}
    missing = set(uniq)
    if isinstance(df.columns, pd.MultiIndex):
        top = df.columns.get_level_values(0)
        for t in uniq:
            if t in top:
                sub = _extract_single(df[t])
                if not sub.empty:
                    out[t] = sub; missing.discard(t)
    elif not df.empty and len(uniq)==1:
        sub = _extract_single(df)
        if not sub.empty:
            out[uniq[0]] = sub; missing.discard(uniq[0])
    if missing and retries>0:
        for t in list(missing):
            try:
                df1 = yf.download(t, period=period, **kwargs)
                sub = _extract_single(df1)
                if not sub.empty:
                    out[t] = sub; missing.discard(t)
            except Exception:
                pass
    out["__failed__"] = pd.Series(sorted(missing)) if missing else pd.Series([], dtype=str)
    return out
