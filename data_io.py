# -*- coding: utf-8 -*-
from __future__ import annotations
import datetime as dt
from collections import OrderedDict
from typing import List, Dict, Tuple
import pandas as pd, numpy as np, yfinance as yf, requests, streamlit as st
from io import StringIO

HEADERS = {"User-Agent": "Mozilla/5.0"}
YF_THREADS = True

WIG20_HC_TICKERS = ["PKN","KGH","PKO","PEO","PZU","PGE","CDR","ALE","DNP","LPP","OPL","CPS","ALR","ING","MBK","TPE","JSW","CCC","KTY"]
MWIG40_HC_TICKERS = ["XTB","PLW","TEN","KRU","GPW","BDX","BHW","LWB","AMC","ASB","11B","CIG","MLG","STP","PKP","MAB","NEU","OPN","VRG","WPL","DOM","MRC","PHN","TIM","MFO","PBX","BRS","FTE","TOR","LVC","DNP"]
SP500_FALLBACK = ["AAPL","MSFT","NVDA","GOOGL","GOOG","AMZN","META","BRK-B","AVGO","TSLA","UNH","LLY","XOM","JPM","V","JNJ","PG","MA","HD","COST","MRK","ABBV","PEP","KO","TMO","BAC","WMT","ADBE","NFLX","CRM","CSCO","INTC","QCOM","NKE","LIN","ACN","MCD","ORCL","TXN","AMD"]

STOOQ_BASE_URLS = (
    "https://stooq.pl",
    "https://stooq.com",
)

STOOQ_INTERVAL_MAP = {
    "1wk": "w",
    "1mo": "m",
    "3mo": "q",
}

DATA_SOURCE_SPECS: Dict[str, Dict[str, object]] = OrderedDict(
    [
        ("yahoo", {"label": "Yahoo Finance (yfinance)", "supports_batch": True}),
        ("stooq", {"label": "Stooq.pl CSV", "supports_batch": False}),
    ]
)

DEFAULT_SOURCE_PRIORITY: List[str] = list(DATA_SOURCE_SPECS.keys())
_CACHE_REGISTRY_KEY = "_data_cache_registry"


def get_available_data_sources() -> OrderedDict:
    """Return metadata about supported free data sources."""
    return OrderedDict(DATA_SOURCE_SPECS)


def get_default_source_priority() -> List[str]:
    return DEFAULT_SOURCE_PRIORITY[:]


def normalize_source_priority(priority: List[str] | None) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for src in priority or []:
        if src in DATA_SOURCE_SPECS and src not in seen:
            out.append(src); seen.add(src)
    for src in DEFAULT_SOURCE_PRIORITY:
        if src not in seen:
            out.append(src); seen.add(src)
    return out


def source_display_name(source: str) -> str:
    spec = DATA_SOURCE_SPECS.get(source, {})
    return str(spec.get("label", source))


def _ensure_cache_registry() -> Dict[Tuple[str, str], Dict[str, object]]:
    if _CACHE_REGISTRY_KEY not in st.session_state:
        st.session_state[_CACHE_REGISTRY_KEY] = {}
    return st.session_state[_CACHE_REGISTRY_KEY]


def clear_cache_registry() -> None:
    st.session_state.pop(_CACHE_REGISTRY_KEY, None)


def _record_cache_entry(
    *,
    ticker: str,
    interval: str,
    df: pd.DataFrame,
    source: str,
    period: str | None,
    start: str | None,
) -> None:
    if df is None or df.empty:
        return
    registry = _ensure_cache_registry()
    try:
        idx = pd.to_datetime(df.index)
    except Exception:
        idx = df.index
    first = pd.to_datetime(idx.min(), errors="coerce")
    last = pd.to_datetime(idx.max(), errors="coerce")
    registry[(ticker, interval)] = {
        "ticker": ticker,
        "interval": interval,
        "source": source_display_name(source),
        "rows": int(len(df)),
        "first": first.date().isoformat() if pd.notna(first) else "-",
        "last": last.date().isoformat() if pd.notna(last) else "-",
        "param_period": period or "-",
        "param_start": start or "-",
        "updated": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def get_cache_inventory() -> pd.DataFrame:
    registry = _ensure_cache_registry()
    if not registry:
        return pd.DataFrame(columns=["Ticker", "TF", "Źródło", "Wiersze", "Zakres", "Parametry", "Aktualizacja"])
    rows = []
    for meta in registry.values():
        rows.append(
            {
                "Ticker": meta.get("ticker", "-"),
                "TF": meta.get("interval", "-"),
                "Źródło": meta.get("source", "-"),
                "Wiersze": meta.get("rows", 0),
                "Zakres": f"{meta.get('first','-')} → {meta.get('last','-')}",
                "Parametry": f"start={meta.get('param_start','-')}, period={meta.get('param_period','-')}",
                "Aktualizacja": meta.get("updated", "-"),
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values(by=["TF", "Ticker"], ascending=[True, True], ignore_index=True)


def clear_all_cached_data() -> None:
    st.cache_data.clear()
    clear_cache_registry()


def _yahoo_to_stooq_symbol(yahoo_ticker: str) -> str | None:
    if not yahoo_ticker:
        return None
    ticker = yahoo_ticker.strip()
    if "." in ticker:
        base, suffix = ticker.rsplit(".", 1)
        if suffix.upper() in {"WA", "PL"}:
            return base.lower()
    return None


def _period_to_offset(period: str | None) -> pd.DateOffset | None:
    if not period or period == "max":
        return None
    if period.endswith("y") and period[:-1].isdigit():
        return pd.DateOffset(years=int(period[:-1]))
    if period.endswith("mo") and period[:-2].isdigit():
        return pd.DateOffset(months=int(period[:-2]))
    return None


def _limit_history(df: pd.DataFrame, *, start: str | None, period: str | None) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if start:
        start_ts = pd.to_datetime(start, errors="coerce")
        if pd.notna(start_ts):
            out = out.loc[out.index >= start_ts]
    offset = _period_to_offset(period)
    if offset is not None and not out.empty:
        tz = out.index.tz if hasattr(out.index, "tz") else None
        now = pd.Timestamp.today(tz=tz)
        cutoff = now - offset
        out = out.loc[out.index >= cutoff]
    return out


@st.cache_data(ttl=60*60*12, show_spinner=False)
def _yahoo_download_many_cached(
    tickers: Tuple[str, ...],
    *,
    interval: str,
    period: str | None,
    start: str | None,
) -> pd.DataFrame:
    kwargs = dict(interval=interval, auto_adjust=False, progress=False, threads=YF_THREADS)
    try:
        if start:
            return yf.download(list(tickers), start=start, **kwargs)
        return yf.download(list(tickers), period=period or "max", **kwargs)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60*60*12, show_spinner=False)
def _yahoo_download_single_cached(
    ticker: str,
    *,
    interval: str,
    period: str | None,
    start: str | None,
) -> pd.DataFrame:
    kwargs = dict(interval=interval, auto_adjust=False, progress=False, threads=YF_THREADS)
    try:
        if start:
            data = yf.download(ticker, start=start, **kwargs)
        else:
            data = yf.download(ticker, period=period or "max", **kwargs)
    except Exception:
        data = pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data = data.copy()
        data.columns = data.columns.get_level_values(0)
    return data


def fetch_stooq_ohlcv(
    yahoo_ticker: str,
    *,
    interval: str = "1wk",
    start: str | None = None,
    period: str | None = None,
) -> pd.DataFrame:
    symbol = _yahoo_to_stooq_symbol(yahoo_ticker)
    if not symbol:
        return pd.DataFrame()
    freq = STOOQ_INTERVAL_MAP.get(interval, "w")
    for base in STOOQ_BASE_URLS:
        url = f"{base}/q/d/l/?s={symbol}&c=0&i={freq}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
        except Exception:
            continue
        if not getattr(resp, "ok", False):
            continue
        text = getattr(resp, "text", "")
        if not text or "Date" not in text:
            continue
        try:
            raw = pd.read_csv(StringIO(text))
        except Exception:
            continue
        if raw.empty or "Date" not in raw.columns:
            continue
        raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
        raw = raw.dropna(subset=["Date"]).set_index("Date").sort_index()
        raw = raw.loc[~raw.index.duplicated(keep="last")]
        cols = [c for c in ["Open", "High", "Low", "Close"] if c in raw.columns]
        if len(cols) < 4:
            continue
        for col in cols:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        data = raw[cols].dropna()
        data = _limit_history(data, start=start, period=period)
        if not data.empty:
            return data
    return pd.DataFrame()

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

def load_weekly_ohlcv(
    yahoo_ticker: str,
    period: str = "5y",
    *,
    start: str | None = None,
    source_priority: List[str] | None = None,
) -> pd.DataFrame:
    data_map = load_many_weekly_ohlcv(
        [yahoo_ticker],
        period=period,
        start=start,
        retries=1,
        source_priority=source_priority,
    )
    return data_map.get(yahoo_ticker, pd.DataFrame())

def _extract_single(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    cols = [c for c in ["Open","High","Low","Close"] if c in df.columns]
    if not cols:
        return pd.DataFrame()
    sub = df[cols]
    try:
        return sub.dropna(subset=cols)
    except KeyError:
        # Fallback when pandas complains about subset on unexpected column index
        return sub.dropna(how="any")


def _load_many_from_yahoo(
    tickers: List[str],
    *,
    interval: str,
    period: str | None,
    start: str | None,
    retries: int,
) -> Tuple[Dict[str, pd.DataFrame], set[str]]:
    uniq = sorted(set([t for t in tickers if t]))
    if not uniq:
        return {}, set()
    bulk = _yahoo_download_many_cached(tuple(uniq), interval=interval, period=period, start=start)
    out: Dict[str, pd.DataFrame] = {}
    missing: set[str] = set(uniq)
    if isinstance(bulk.columns, pd.MultiIndex):
        top = bulk.columns.get_level_values(0)
        for t in uniq:
            if t in top:
                sub = _extract_single(bulk[t])
                if not sub.empty:
                    out[t] = sub; missing.discard(t)
    elif not bulk.empty and len(uniq) == 1:
        sub = _extract_single(bulk)
        if not sub.empty:
            out[uniq[0]] = sub; missing.discard(uniq[0])
    if missing and retries > 0:
        for t in list(missing):
            single = _yahoo_download_single_cached(t, interval=interval, period=period, start=start)
            sub = _extract_single(single)
            if not sub.empty:
                out[t] = sub; missing.discard(t)
    return out, missing

def load_many_weekly_ohlcv(
    tickers: list[str],
    *,
    period: str = "5y",
    start: str | None = None,
    retries: int = 1,
    source_priority: List[str] | None = None,
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    uniq = sorted(set([t for t in tickers if t]))
    missing: set[str] = set(uniq)
    priority = normalize_source_priority(source_priority)
    for source in priority:
        if not missing:
            break
        if source == "yahoo":
            fetched, missing = _load_many_from_yahoo(
                list(missing), interval="1wk", period=period, start=start, retries=retries
            )
            for ticker, df in fetched.items():
                out[ticker] = df
                _record_cache_entry(
                    ticker=ticker,
                    interval="1wk",
                    df=df,
                    source=source,
                    period=period,
                    start=start,
                )
        elif source == "stooq":
            for ticker in list(missing):
                fallback = fetch_stooq_ohlcv(ticker, interval="1wk", start=start, period=period)
                if not fallback.empty:
                    out[ticker] = fallback
                    _record_cache_entry(
                        ticker=ticker,
                        interval="1wk",
                        df=fallback,
                        source=source,
                        period=period,
                        start=start,
                    )
                    missing.discard(ticker)
        else:
            continue
    out["__failed__"] = pd.Series(sorted(missing)) if missing else pd.Series([], dtype=str)
    return out

def load_htf_ohlcv(
    yahoo_ticker: str,
    *,
    interval: str = "1mo",
    period: str = "max",
    source_priority: List[str] | None = None,
) -> pd.DataFrame:
    data_map = load_many_htf_ohlcv(
        [yahoo_ticker],
        interval=interval,
        period=period,
        retries=1,
        source_priority=source_priority,
    )
    return data_map.get(yahoo_ticker, pd.DataFrame())


def load_many_htf_ohlcv(
    tickers: list[str],
    *,
    interval: str = "1mo",
    period: str = "max",
    retries: int = 1,
    source_priority: List[str] | None = None,
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    uniq = sorted(set([t for t in tickers if t]))
    missing: set[str] = set(uniq)
    priority = normalize_source_priority(source_priority)
    for source in priority:
        if not missing:
            break
        if source == "yahoo":
            fetched, missing = _load_many_from_yahoo(
                list(missing), interval=interval, period=period, start=None, retries=retries
            )
            for ticker, df in fetched.items():
                out[ticker] = df
                _record_cache_entry(
                    ticker=ticker,
                    interval=interval,
                    df=df,
                    source=source,
                    period=period,
                    start=None,
                )
        elif source == "stooq":
            for ticker in list(missing):
                fallback = fetch_stooq_ohlcv(ticker, interval=interval, period=period)
                if not fallback.empty:
                    out[ticker] = fallback
                    _record_cache_entry(
                        ticker=ticker,
                        interval=interval,
                        df=fallback,
                        source=source,
                        period=period,
                        start=None,
                    )
                    missing.discard(ticker)
        else:
            continue
    out["__failed__"] = pd.Series(sorted(missing)) if missing else pd.Series([], dtype=str)
    return out
