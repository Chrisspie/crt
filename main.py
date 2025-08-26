# -*- coding: utf-8 -*-
import datetime as dt
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# ─────────────────────────────────────────
# USTAWIENIA I NAGŁÓWEK
# ─────────────────────────────────────────
st.set_page_config(page_title="CRT Scanner – WIG20/mWIG40 + US", layout="wide")
st.title("📈 CRT Scanner – WIG20 / mWIG40 / US")
st.caption(
    "CRT: C1=range, C2=sweep+close back in range, C3=wyjście w przeciwną stronę. "
    "Dane: Yahoo Finance (1wk). Edukacyjnie, nie jest to rekomendacja inwestycyjna."
)

HEADERS = {"User-Agent": "Mozilla/5.0"}
YF_THREADS = False  # stabilniej w niektórych środowiskach

# ─────────────────────────────────────────
# FALLBACK LISTY (gdy Wikipedia nie działa)
# ─────────────────────────────────────────
def fallback_wig20() -> pd.DataFrame:
    # Subset pewniaków – wystarczy, żeby appka działała offline.
    data = [
        ("Orlen", "PKN"), ("KGHM", "KGH"), ("PKO BP", "PKO"), ("Bank Pekao", "PEO"),
        ("PZU", "PZU"), ("PGE", "PGE"), ("CD Projekt", "CDR"), ("Allegro", "ALE"),
        ("Dino Polska", "DNP"), ("LPP", "LPP"), ("Orange Polska", "OPL"), ("Cyfrowy Polsat", "CPS"),
    ]
    df = pd.DataFrame(data, columns=["company", "gpw_ticker"])
    df["yahoo_ticker"] = df["gpw_ticker"].astype(str) + ".WA"
    df["group"] = "WIG20 (fallback)"
    return df

# ─────────────────────────────────────────
# POBIERANIE SKŁADÓW INDEKSÓW
# ─────────────────────────────────────────
@st.cache_data(ttl=60 * 60 * 24)  # 24h
def fetch_wiki_table(urls: List[str], idx_name: str) -> pd.DataFrame:
    """
    Pobiera skład indeksu z Wikipedii, heurystycznie wybiera tabelę z Tickerami.
    Zwraca kolumny: company, gpw_ticker, yahoo_ticker, group
    """
    for url in urls:
        try:
            html = requests.get(url, headers=HEADERS, timeout=20).text
            tables = pd.read_html(html)
            best = None
            for t in tables:
                cols = [str(c).lower() for c in t.columns]
                # POPRAWKA: poprawny 'any' zagnieżdżony (wcześniej była zła składnia)
                if any(any(x in c for x in ["ticker", "symbol", "kod"]) for c in cols):
                    best = t.copy()
                    break
                # fallback – 2+ kolumnowe tabele (Company/Ticker)
                if t.shape[1] >= 2 and best is None:
                    best = t.copy()
            if best is None or best.empty:
                continue

            best.columns = [str(c).strip() for c in best.columns]
            name_candidates = [c for c in best.columns if any(x in c.lower() for x in ["spółka", "company", "nazwa"])]
            tick_candidates = [c for c in best.columns if any(x in c.lower() for x in ["ticker", "symbol", "kod"])]

            name_col = name_candidates[0] if name_candidates else best.columns[0]
            tick_col = tick_candidates[0] if tick_candidates else (best.columns[1] if best.shape[1] > 1 else best.columns[0])

            out = best[[name_col, tick_col]].copy()
            out.columns = ["company", "gpw_ticker"]
            # POPRAWKA: .str.strip() zamiast .strip() na Series
            out["company"] = out["company"].astype(str).str.strip()
            out["gpw_ticker"] = (
                out["gpw_ticker"].astype(str).str.upper().str.strip()
                .str.replace(r"\s+", "", regex=True)
                .str.replace(r"[^A-Z0-9]", "", regex=True)
            )
            out = out.dropna(subset=["gpw_ticker"])
            out = out[out["gpw_ticker"].str.len() > 0].drop_duplicates(subset=["gpw_ticker"])
            if out.empty:
                continue

            out["yahoo_ticker"] = out["gpw_ticker"].astype(str) + ".WA"
            out["group"] = idx_name
            return out.reset_index(drop=True)
        except Exception:
            continue
    return pd.DataFrame(columns=["company", "gpw_ticker", "yahoo_ticker", "group"])

@st.cache_data(ttl=60 * 60 * 24)
def fetch_wig20() -> pd.DataFrame:
    urls = ["https://pl.wikipedia.org/wiki/WIG20", "https://en.wikipedia.org/wiki/WIG20"]
    df = fetch_wiki_table(urls, "WIG20")
    if df.empty:
        df = fallback_wig20()
    return df

@st.cache_data(ttl=60 * 60 * 24)
def fetch_mwig40() -> pd.DataFrame:
    urls = ["https://pl.wikipedia.org/wiki/MWIG40", "https://en.wikipedia.org/wiki/MWIG40"]
    return fetch_wiki_table(urls, "mWIG40")

# ─────────────────────────────────────────
# DANE CENOWE
# ─────────────────────────────────────────
@st.cache_data(ttl=60 * 60 * 12)  # 12h
def load_weekly_ohlcv(yahoo_ticker: str, period: str = "5y") -> pd.DataFrame:
    df = yf.download(
        yahoo_ticker,
        interval="1wk",
        period=period,
        auto_adjust=False,
        progress=False,
        threads=YF_THREADS,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if not df.empty:
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df

# ─────────────────────────────────────────
# PARSERY INPUTÓW
# ─────────────────────────────────────────
def parse_us_tickers(raw: str) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=["company", "yahoo_ticker", "group"])
    parts = [p.strip().upper() for p in raw.replace(";", ",").split(",")]
    parts = [p for p in parts if p]
    if not parts:
        return pd.DataFrame(columns=["company", "yahoo_ticker", "group"])
    df = pd.DataFrame({"company": parts, "yahoo_ticker": parts})
    df["group"] = "US"
    return df

def parse_gpw_tickers(raw: str) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=["company", "yahoo_ticker", "group"])
    parts = [p.strip().upper() for p in raw.replace(";", ",").split(",")]
    parts = [p for p in parts if p]
    if not parts:
        return pd.DataFrame(columns=["company", "yahoo_ticker", "group"])
    df = pd.DataFrame({"company": parts})
    df["yahoo_ticker"] = df["company"].astype(str) + ".WA"
    df["group"] = "GPW (manual)"
    return df

# ─────────────────────────────────────────
# CRT – LOGIKA
# ─────────────────────────────────────────
def midline(low: float, high: float) -> float:
    return (float(low) + float(high)) / 2.0

def _find_c3(
    d: pd.DataFrame,
    i_c2: int,
    c1_low: float,
    c1_high: float,
    method: str,
    direction: str,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Zwraca: (c3_idx_within_N, c3_idx_to_end)
    c3_idx_within_N ustawiamy później w crt_scan (bo zależy od confirm_within).
    Tu wykrywamy 'do końca serii' (any) oraz zbieramy indeksy, żeby filtr N zrobić później.
    """
    n = len(d)
    c3_any_idx = None
    # "method == 'high'" rozumiemy jako potwierdzenie knotem (High/Low),
    # "method == 'close'" jako zamknięciem świecy.
    for j in range(i_c2 + 1, n):
        H, L, C = float(d.iloc[j]["High"]), float(d.iloc[j]["Low"]), float(d.iloc[j]["Close"])
        if direction == "BULL":
            cond_any = (H > c1_high) if method == "high" else (C > c1_high)
        else:  # BEAR
            cond_any = (L < c1_low) if method == "high" else (C < c1_low)
        if cond_any:
            c3_any_idx = j
            break
    # c3_idx_within_N policzymy w crt_scan – tu zwracamy tylko c3_any_idx jako druga wartość
    return None, c3_any_idx

def crt_scan(
    df: pd.DataFrame,
    lookback_bars: int = 30,
    require_midline: bool = False,
    strict_vs_c1open: bool = False,
    confirm_within: int = 0,            # 0 = brak potwierdzenia C3
    confirm_method: str = "high",       # 'high' (wick) lub 'close' (zamknięcie)
    directions: Tuple[str, ...] = ("bullish", "bearish"),
) -> List[Dict]:
    """
    Zwraca listę setupów CRT. Każdy rekord zawiera:
    - confirmed (czy C3 wystąpiła w ciągu N świec – jeśli N>0)
    - c3_happened (czy C3 wystąpiła w ogóle – do końca serii)
    - idxy/datę C1/C2/C3, poziomy C1/C2, pozycję C2 w zakresie itd.
    """
    out = []
    if df is None or df.empty or len(df) < 5:
        return out

    d = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if len(d) < 5:
        return out

    n = len(d)
    start_idx = max(1, n - lookback_bars)

    # POPRAWKA: obejmujemy też ostatnią świecę jako potencjalną C2
    for i in range(start_idx, n):
        C1 = d.iloc[i - 1]
        C2 = d.iloc[i]
        C1L, C1H = float(C1["Low"]), float(C1["High"])
        C2L, C2H, C2C, C1O = float(C2["Low"]), float(C2["High"]), float(C2["Close"]), float(C1["Open"])
        C1_mid = midline(C1L, C1H)
        close_in = (C1L <= C2C <= C1H)

        def _record(direction: str, swept_side: str):
            c3_within_idx, c3_any_idx = _find_c3(d, i, C1L, C1H, confirm_method, direction)

            # policz c3_within_idx jeśli N>0
            confirmed_within = False
            if confirm_within and confirm_within > 0 and (i + 1) < n:
                end_j = min(n - 1, i + confirm_within)
                for j in range(i + 1, end_j + 1):
                    H, L, C = float(d.iloc[j]["High"]), float(d.iloc[j]["Low"]), float(d.iloc[j]["Close"])
                    if direction == "BULL":
                        cond = (H > C1H) if confirm_method == "high" else (C > C1H)
                    else:
                        cond = (L < C1L) if confirm_method == "high" else (C < C1L)
                    if cond:
                        confirmed_within = True
                        c3_within_idx = j
                        break

            rec = {
                "direction": "BULL" if direction == "bullish" else "BEAR",
                "C1_date": d.index[i - 1],
                "C2_date": d.index[i],
                "C3_date_within": d.index[c3_within_idx] if c3_within_idx is not None else pd.NaT,
                "C3_date_any": d.index[c3_any_idx] if c3_any_idx is not None else pd.NaT,
                "confirmed": bool(confirmed_within),
                "c3_happened": bool(c3_any_idx is not None),  # niezależnie od confirm_within
                "confirm_rule": f"{confirm_method}>{'C1H' if direction=='bullish' else 'C1L'} in {confirm_within}" if confirm_within else "no confirm",
                "C1_low": C1L, "C1_high": C1H, "C1_mid": C1_mid, "C1_open": C1O,
                "C2_low": C2L, "C2_high": C2H, "C2_close": C2C,
                "C2_position_in_range": (C2C - C1L) / (C1H - C1L) if (C1H > C1L) else np.nan,
                "swept_side": swept_side,
            }
            out.append(rec)

        # Byczy: sweep LOW C1 + close back in range (+opcje)
        if "bullish" in directions:
            cond = (C2L < C1L) and close_in
            if require_midline:
                cond &= (C2C >= C1_mid)
            if strict_vs_c1open:
                cond &= (C2C >= C1O)
            if cond:
                _record("bullish", "LOW")

        # Niedźwiedzi: sweep HIGH C1 + close back in range (+opcje)
        if "bearish" in directions:
            cond = (C2H > C1H) and close_in
            if require_midline:
                cond &= (C2C <= C1_mid)
            if strict_vs_c1open:
                cond &= (C2C <= C1O)
            if cond:
                _record("bearish", "HIGH")

    # Najnowsze C2 na górze
    out.sort(key=lambda r: (pd.Timestamp(r["C2_date"]) if pd.notna(r["C2_date"]) else pd.Timestamp(0)), reverse=True)
    return out

# Override crt_scan with fixed core implementation
from crt_core import crt_scan as _crt_scan_fixed
crt_scan = _crt_scan_fixed

# ─────────────────────────────────────────
# SIDEBAR – ŹRÓDŁA UNIWERSUM + USTAWIENIA
# ─────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Dane & CRT")

    st.subheader("Indeksy GPW")
    use_wig20 = st.toggle("WIG20", value=True)
    use_mwig40 = st.toggle("mWIG40", value=True)

    st.subheader("Ręczny input (GPW)")
    if "gpw_input" not in st.session_state:
        st.session_state.gpw_input = ""
    col_btn, col_txt = st.columns([1, 2])
    with col_btn:
        if st.button("Wklej przykład WIG20"):
            st.session_state.gpw_input = "PKN, KGH, PKO, PEO, PZU, PGE, CDR, ALE, DNP, LPP, OPL, CPS"
            st.rerun()
    gpw_raw = st.text_area("GPW tickery (np. PKN, KGH, PKO…)", key="gpw_input", height=70)

    st.subheader("Amerykańskie spółki (US)")
    raw_us = st.text_area("US tickery (np. AAPL, MSFT, NVDA)", placeholder="AAPL, MSFT, NVDA", height=70)

    st.divider()
    st.subheader("Zakres danych")
    period = st.selectbox("Okres historii (1wk)", options=["2y", "5y", "10y", "max"], index=1)

    st.subheader("Parametry CRT")
    directions_map = {
        "Bycze + Niedźwiedzie": ("bullish", "bearish"),
        "Tylko bycze": ("bullish",),
        "Tylko niedźwiedzie": ("bearish",),
    }
    dir_choice = st.selectbox("Kierunek", list(directions_map.keys()), index=0)
    directions = directions_map[dir_choice]
    lookback_bars = st.slider("Lookback (tygodnie)", 10, 120, 40, step=5)
    require_midline = st.checkbox("Wymagaj midline 50% (C2C po właściwej stronie)", value=False)
    strict_vs_c1open = st.checkbox("Surowszy wariant (C2C vs C1O)", value=False)

    st.subheader("Potwierdzenie C3 (dla skanu ogólnego)")
    confirm_on = st.checkbox("Wymagaj potwierdzenia C3", value=True)
    confirm_within = st.number_input("C3 w ≤ X świec", 1, 8, 3, 1, disabled=not confirm_on)
    confirm_method = st.selectbox("Sposób potwierdzenia", ["high", "close"], index=0, disabled=not confirm_on)

    st.divider()
    st.subheader("Tryb szukania okazji")
    opportunity_mode = st.checkbox("Okazje C3 (ostatnie 2 tygodnie)", value=True,
                                   help="Pokaż układy, w których C2 było w jednym z dwóch ostatnich zamkniętych tygodni i C3 JESZCZE nie nastąpiło.")
    st.divider()
    if st.button("🧹 Wyczyść cache"):
        st.cache_data.clear()
        st.success("Cache wyczyszczony.")
        st.rerun()

# ─────────────────────────────────────────
# UNIWERSUM + PANEL AKTYWACJI
# ─────────────────────────────────────────
def build_universe_df(use_wig20: bool, use_mwig40: bool, gpw_raw: str, raw_us: str) -> pd.DataFrame:
    frames = []
    if use_wig20:
        wig20 = fetch_wig20()
        if not wig20.empty:
            frames.append(wig20[["company", "yahoo_ticker", "group"]])
    if use_mwig40:
        mwig40 = fetch_mwig40()
        if not mwig40.empty:
            frames.append(mwig40[["company", "yahoo_ticker", "group"]])

    gpw_manual = parse_gpw_tickers(gpw_raw)
    if not gpw_manual.empty:
        frames.append(gpw_manual[["company", "yahoo_ticker", "group"]])

    usdf = parse_us_tickers(raw_us)
    if not usdf.empty:
        frames.append(usdf[["company", "yahoo_ticker", "group"]])

    if not frames:
        return pd.DataFrame(columns=["company", "yahoo_ticker", "group"])

    uni = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["yahoo_ticker"])
    uni["Active"] = True
    uni["company"] = uni["company"].fillna(uni["yahoo_ticker"].str.replace(".WA", "", regex=False))
    return uni

if "active_map" not in st.session_state:
    st.session_state.active_map = {}

universe_df = build_universe_df(use_wig20, use_mwig40, gpw_raw, raw_us)

if universe_df.empty:
    st.warning("Brak spółek do skanowania. Włącz indeksy lub podaj tickery GPW/US w sidebarze.")
    st.stop()

# Zastosuj poprzednie wybory Active
if st.session_state.active_map:
    universe_df["Active"] = universe_df["yahoo_ticker"].map(st.session_state.active_map).fillna(True)

st.subheader("🎛️ Panel aktywnych spółek")
colA, colB, colC = st.columns([1, 1, 2])
with colA:
    if st.button("Zaznacz wszystkie"):
        universe_df["Active"] = True
with colB:
    if st.button("Odznacz wszystkie"):
        universe_df["Active"] = False

edited_df = st.data_editor(
    universe_df.sort_values(["group", "yahoo_ticker"]).reset_index(drop=True),
    key="universe_editor",
    use_container_width=True,
    height=360,
    column_config={
        "company": st.column_config.TextColumn("Spółka", disabled=True),
        "yahoo_ticker": st.column_config.TextColumn("Ticker (Yahoo)", disabled=True),
        "group": st.column_config.TextColumn("Grupa", disabled=True),
        "Active": st.column_config.CheckboxColumn("Aktywna"),
    },
)
st.session_state.active_map = dict(zip(edited_df["yahoo_ticker"], edited_df["Active"]))
active_tickers = edited_df.loc[edited_df["Active"], "yahoo_ticker"].tolist()
meta_map = edited_df.set_index("yahoo_ticker")[["company", "group", "Active"]].to_dict(orient="index")

st.caption(f"Aktywnych tickerów: **{len(active_tickers)}** / {len(edited_df)}")

# ─────────────────────────────────────────
# SKANOWANIE CRT
# ─────────────────────────────────────────
if not active_tickers:
    st.info("Zaznacz przynajmniej jedną spółkę jako aktywną.")
    st.stop()

st.subheader("🔎 Wyniki skanowania CRT (tygodnie)")
progress = st.progress(0.0, text="Pobieranie danych...")

rows = []
for i, yt in enumerate(active_tickers, start=1):
    progress.progress(i / max(1, len(active_tickers)), text=f"Skanowanie: {yt}")
    try:
        df = load_weekly_ohlcv(yt, period=period)
        if df is None or df.empty or len(df) < 5:
            continue

        setups = crt_scan(
            df=df,
            lookback_bars=lookback_bars,
            require_midline=require_midline,
            strict_vs_c1open=strict_vs_c1open,
            confirm_within=(int(confirm_within) if confirm_on else 0),
            confirm_method=confirm_method if confirm_on else "high",
            directions=directions,
        )

        if not setups:
            # w trybie ogólnym pokaż info; w opportunity_mode i tak filtrujemy
            if not opportunity_mode:
                rows.append({
                    "Ticker": yt, "Spółka": meta_map.get(yt, {}).get("company", yt.replace(".WA", "")),
                    "Grupa": meta_map.get(yt, {}).get("group", ""),
                    "Kierunek": "-", "C1": pd.NaT, "C2": pd.NaT, "C3": pd.NaT, "Potwierdzenie": "-",
                    "Zasada potwierdzenia": "-", "C1L": np.nan, "C1H": np.nan, "Mid(50%)": np.nan, "C1O": np.nan,
                    "C2L": np.nan, "C2H": np.nan, "C2C": np.nan, "C2 pos w C1%": np.nan, "Sweep": "-",
                    "Trigger": np.nan, "Stop": np.nan, "TP1": np.nan, "TP2": np.nan, "R:TP1": np.nan, "R:TP2": np.nan,
                })
            continue

        # 2 ostatnie zamknięte tygodnie
        last_two = pd.Index(df.index[-2:])

        for rec in setups:
            c1_ts = pd.to_datetime(rec["C1_date"])
            c2_ts = pd.to_datetime(rec["C2_date"])

            # TRYB OKAZJI: C2 w 2 ostatnich tygodniach + C3 NIGDY nie nastąpiła (do końca serii)
            if opportunity_mode:
                if (c2_ts not in last_two) or rec.get("c3_happened", False):
                    continue

            C1L, C1H = rec["C1_low"], rec["C1_high"]
            rng = (C1H - C1L) if pd.notna(C1H) and pd.notna(C1L) else np.nan

            if rec["direction"] == "BULL":
                trigger = C1H
                stop = rec["C2_low"]
                tp1 = C1H + 0.5 * rng if pd.notna(rng) else np.nan
                tp2 = C1H + 1.0 * rng if pd.notna(rng) else np.nan
                risk = trigger - stop if pd.notna(trigger) and pd.notna(stop) else np.nan
                r_tp1 = (tp1 - trigger) / risk if pd.notna(risk) and risk > 0 and pd.notna(tp1) else np.nan
                r_tp2 = (tp2 - trigger) / risk if pd.notna(risk) and risk > 0 and pd.notna(tp2) else np.nan
            else:
                trigger = C1L
                stop = rec["C2_high"]
                tp1 = C1L - 0.5 * rng if pd.notna(rng) else np.nan
                tp2 = C1L - 1.0 * rng if pd.notna(rng) else np.nan
                risk = stop - trigger if pd.notna(trigger) and pd.notna(stop) else np.nan
                r_tp1 = (trigger - tp1) / risk if pd.notna(risk) and risk > 0 and pd.notna(tp1) else np.nan
                r_tp2 = (trigger - tp2) / risk if pd.notna(risk) and risk > 0 and pd.notna(tp2) else np.nan

            rows.append({
                "Ticker": yt,
                "Spółka": meta_map.get(yt, {}).get("company", yt.replace(".WA", "")),
                "Grupa": meta_map.get(yt, {}).get("group", ""),
                "Kierunek": rec["direction"],
                "C1": c1_ts.date() if pd.notna(c1_ts) else pd.NaT,
                "C2": c2_ts.date() if pd.notna(c2_ts) else pd.NaT,
                "C3_any": (pd.to_datetime(rec.get("C3_date_any")) .date() if pd.notna(rec.get("C3_date_any")) else pd.NaT),
                "Potwierdzenie_w_N": "TAK" if rec.get("confirmed", False) else "NIE",
                "C3_happened": "TAK" if rec.get("c3_happened", False) else "NIE",
                "Zasada potwierdzenia": rec["confirm_rule"],
                "C1L": round(C1L, 2), "C1H": round(C1H, 2),
                "Mid(50%)": round(rec["C1_mid"], 2), "C1O": round(rec["C1_open"], 2),
                "C2L": round(rec["C2_low"], 2), "C2H": round(rec["C2_high"], 2), "C2C": round(rec["C2_close"], 2),
                "C2 pos w C1%": round(100 * rec["C2_position_in_range"], 1) if pd.notna(rec["C2_position_in_range"]) else np.nan,
                "Sweep": rec["swept_side"],
                "Trigger": round(trigger, 2) if pd.notna(trigger) else np.nan,
                "Stop": round(stop, 2) if pd.notna(stop) else np.nan,
                "TP1": round(tp1, 2) if pd.notna(tp1) else np.nan,
                "TP2": round(tp2, 2) if pd.notna(tp2) else np.nan,
                "R:TP1": round(r_tp1, 2) if pd.notna(r_tp1) else np.nan,
                "R:TP2": round(r_tp2, 2) if pd.notna(r_tp2) else np.nan,
            })

    except Exception as e:
        rows.append({
            "Ticker": yt, "Spółka": meta_map.get(yt, {}).get("company", yt.replace(".WA", "")),
            "Grupa": meta_map.get(yt, {}).get("group", ""),
            "Kierunek": "ERR", "C1": pd.NaT, "C2": pd.NaT, "C3_any": pd.NaT,
            "Potwierdzenie_w_N": "ERR", "C3_happened": "ERR", "Zasada potwierdzenia": str(e),
            "C1L": np.nan, "C1H": np.nan, "Mid(50%)": np.nan, "C1O": np.nan,
            "C2L": np.nan, "C2H": np.nan, "C2C": np.nan, "C2 pos w C1%": np.nan, "Sweep": "-",
            "Trigger": np.nan, "Stop": np.nan, "TP1": np.nan, "TP2": np.nan, "R:TP1": np.nan, "R:TP2": np.nan,
        })

progress.empty()
out_df = pd.DataFrame(rows)

# ─────────────────────────────────────────
# WIDOK / SORT / EKSPORT
# ─────────────────────────────────────────
if not out_df.empty:
    # sort: w trybie okazji – najnowsze C2; w ogólnym – (potwierdzone, potem C2)
    out_df["C2_sort"] = pd.to_datetime(out_df["C2"], errors="coerce")
    if opportunity_mode:
        out_df = out_df.sort_values(by=["C2_sort", "Grupa", "Ticker"], ascending=[False, True, True])
    else:
        out_df["confirmed_flag"] = out_df["Potwierdzenie_w_N"].eq("TAK").astype(int)
        out_df = out_df.sort_values(by=["confirmed_flag", "C2_sort", "Grupa", "Ticker"],
                                    ascending=[False, False, True, True]).drop(columns=["confirmed_flag"])
    out_df = out_df.drop(columns=["C2_sort"])

    st.dataframe(out_df, use_container_width=True, height=560)

    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Pobierz wyniki (CSV)",
        data=csv_bytes,
        file_name=f"crt_opportunities_{dt.date.today().isoformat()}.csv" if opportunity_mode else f"crt_scan_{dt.date.today().isoformat()}.csv",
        mime="text/csv",
    )
else:
    st.info("Brak wyników dla bieżących ustawień.")

with st.expander("ℹ️ Uwaga dot. 'high' vs 'close' w potwierdzeniu"):
    st.markdown("""
- **high** – potwierdzenie knotem: BULL = `High > C1H`, BEAR = `Low < C1L`  
- **close** – potwierdzenie zamknięciem: BULL = `Close > C1H`, BEAR = `Close < C1L`  
W trybie „Okazje C3” i tak filtrujemy **C3_happened==NIE** (do **końca serii**), niezależnie od tego,
czy potwierdzenie w N świec jest włączone.
""")
