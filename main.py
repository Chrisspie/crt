# -*- coding: utf-8 -*-
from __future__ import annotations
import datetime as dt, numpy as np, pandas as pd, streamlit as st
from data_io import load_many_weekly_ohlcv, load_many_htf_ohlcv
from universe import build_universe_df
from crt_core import crt_scan, get_key_level_and_confluence
from tv_chart import build_plotly_chart
from datetime import date, timedelta

st.set_page_config(page_title="CRT Scanner – FAST+FIX (WIG / US)", layout="wide")
st.title("⚡ CRT Scanner – FAST + Robust (WIG / mWIG40 / US)")
st.caption("Batch Yahoo fetch + aliasy GPW + retry + wykres W1 z poziomami.")

with st.sidebar:
    st.header("⚙️ Dane & CRT")
    st.subheader("Indeksy GPW")
    use_wig_all = st.toggle("WIG", value=False)
    use_wig20 = st.toggle("WIG20", value=True)
    use_mwig40 = st.toggle("mWIG40", value=True)

    st.subheader("Polska (GPW)")
    st.session_state.setdefault("gpw_input","")
    gpw_raw = st.text_area("GPW tickery (np. PKN, KGH, PKO…)", key="gpw_input", height=70)

    st.subheader("Amerykańskie spółki (US)")
    use_sp500 = st.toggle("Dołącz spółki z S&P500 (cache 7d)", value=False)
    raw_us = st.text_area("US tickery (np. AAPL, MSFT, NVDA)", placeholder="AAPL, MSFT, NVDA", height=70)

    st.divider()
    st.subheader("Zakres danych")
    period = st.selectbox("Okres historii (1wk)", options=["2y","5y","10y","max"], index=1)

    st.subheader("Parametry CRT")
    directions_map = {
        "Bycze + Niedźwiedzie": ("bullish","bearish"),
        "Tylko bycze": ("bullish",),
        "Tylko niedźwiedzie": ("bearish",),
    }
    directions = directions_map[st.selectbox("Kierunek", list(directions_map.keys()), index=0)]
    lookback_bars = st.slider("Lookback (tygodnie)", 10, 120, 40, step=5)
    require_midline = st.checkbox("Wymagaj midline 50% (C2C po właściwej stronie)", value=False)
    strict_vs_c1open = st.checkbox("Surowszy wariant (C2C vs C1O)", value=False)

    st.subheader("Potwierdzenie C3")
    confirm_on = st.checkbox("Wymagaj potwierdzenia C3", value=True)
    confirm_within = st.number_input("C3 w ≤ X świec", 1, 8, 3, 1, disabled=not confirm_on)
    confirm_method = st.selectbox("Sposób potwierdzenia", ["high","close"], index=0, disabled=not confirm_on)

    st.divider()
    st.subheader("Tryb szukania okazji")
    opportunity_mode = st.checkbox("Okazje C3 (ostatnie 2 tygodnie)", value=True)

    st.divider()
    st.subheader("Key Level (HTF)")
    key_on = st.checkbox("Włącz Key Level (konfluencja HTF)", value=True)
    key_tf_label = st.selectbox("TF poziomu", ["1M (miesięczny)","3M (kwartalny)"], index=0, disabled=not key_on)
    key_window_months = st.slider("Okno (miesiące)", 3, 36, 12, step=3, disabled=not key_on)
    key_interact = st.selectbox("Interakcja z poziomem", ["C1 lub C2","Tylko C1","Tylko C2"], index=0, disabled=not key_on)
    key_strict = st.selectbox("Reguła kontaktu", ["strict (<, >)","touch (≤, ≥)"], index=0, disabled=not key_on)
    key_require = st.checkbox("Wymagaj konfluencji (filtruj wyniki)", value=True, disabled=not key_on)

    st.divider()
    if st.button("🧹 Wyczyść cache"):
        st.cache_data.clear(); st.success("Cache wyczyszczony."); st.rerun()

st.session_state.setdefault("active_map", {})
universe_df = build_universe_df(use_wig_all, use_wig20, use_mwig40, use_sp500, gpw_raw, raw_us)
if universe_df.empty:
    st.warning("Brak spółek do skanowania."); st.stop()

# Avoid downcast warning
active_series = universe_df["yahoo_ticker"].map(st.session_state["active_map"])
universe_df["Active"] = active_series.where(active_series.notna(), True).astype(bool)

st.subheader("🎛️ Panel aktywnych spółek")
colA, colB, _ = st.columns([1,1,2])
with colA:
    if st.button("Zaznacz wszystkie"): universe_df["Active"] = True
with colB:
    if st.button("Odznacz wszystkie"): universe_df["Active"] = False

edited_df = st.data_editor(
    universe_df.sort_values(["group","yahoo_ticker"]).reset_index(drop=True),
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
st.session_state["active_map"] = dict(zip(edited_df["yahoo_ticker"], edited_df["Active"]))
active_tickers = edited_df.loc[edited_df["Active"], "yahoo_ticker"].tolist()
meta_map = edited_df.set_index("yahoo_ticker")[["company","group","Active"]].to_dict(orient="index")
st.caption(f"Aktywnych tickerów: **{len(active_tickers)}** / {len(edited_df)}")

if not active_tickers:
    st.info("Zaznacz przynajmniej jedną spółkę."); st.stop()

st.subheader("🔎 Wyniki skanowania CRT (tygodnie)")

# ---- Batch downloads with retries ----
if opportunity_mode:
    effective_weeks = 2 + int(confirm_within) + 6  # cushion for CRT logic
    start_date = (date.today() - timedelta(weeks=effective_weeks)).isoformat()
    data_map = load_many_weekly_ohlcv(active_tickers, period="max", start=start_date, retries=1)
    effective_lookback = 3
else:
    data_map = load_many_weekly_ohlcv(active_tickers, period=period, retries=1)
    effective_lookback = lookback_bars

# Report failed tickers (no data / delisted / timezone missing)
failed_list = data_map.get("__failed__")
if failed_list is not None and len(failed_list) > 0:
    failed = ", ".join(failed_list.tolist()[:40]) + ("…" if len(failed_list) > 40 else "")
    st.warning(f"Brak danych dla: {failed}")

htf_interval = "1mo" if key_tf_label.startswith("1M") else "3mo"
htf_map = load_many_htf_ohlcv(active_tickers, interval=htf_interval, period="max", retries=1) if key_on else {}
htf_failed = htf_map.get("__failed__")
if key_on and htf_failed is not None and len(htf_failed) > 0:
    miss = ", ".join(htf_failed.tolist()[:40]) + ("…" if len(htf_failed) > 40 else "")
    st.info(f"HTF: brak danych dla: {miss}")

progress = st.progress(0.0, text="Analiza...")
rows = []
total = len(active_tickers)

for i, yt in enumerate(active_tickers, start=1):
    progress.progress(i/max(1,total), text=f"Skanowanie: {yt}")
    try:
        df = data_map.get(yt)
        if df is None or df.empty or len(df) < 5:
            continue

        setups = crt_scan(
            df=df,
            lookback_bars=effective_lookback,
            require_midline=require_midline,
            strict_vs_c1open=strict_vs_c1open,
            confirm_within=(int(confirm_within) if confirm_on else 0),
            confirm_method=(confirm_method if confirm_on else "high"),
            directions=directions,
        )

        htf_df = htf_map.get(yt, pd.DataFrame()) if key_on else pd.DataFrame()
        last_two = pd.Index(df.index[-2:])

        for rec in setups:
            c1_ts = pd.to_datetime(rec["C1_date"]); c2_ts = pd.to_datetime(rec["C2_date"])

            if opportunity_mode and ((c2_ts not in last_two) or rec.get("c3_happened", False)):
                continue

            C1L, C1H = rec["C1_low"], rec["C1_high"]
            C2L, C2H, C2C = rec["C2_low"], rec["C2_high"], rec["C2_close"]
            rng = (C1H - C1L) if pd.notna(C1H) and pd.notna(C1L) else np.nan

            key_tf_str, key_level_val, key_date, confluence = ("-", float("nan"), pd.NaT, False)
            if key_on and not htf_df.empty:
                key_tf_str, key_level_val, key_date, confluence = get_key_level_and_confluence(
                    htf_df, c2_ts, rec["direction"], C1L, C1H, C2L, C2H,
                    key_window_months, key_interact, key_strict, htf_interval
                )

            if rec["direction"] == "BULL":
                trigger = C1H; stop = C2L
                tp1 = C1H + 0.5*rng if pd.notna(rng) else np.nan
                tp2 = C1H + 1.0*rng if pd.notna(rng) else np.nan
                risk = trigger - stop if pd.notna(trigger) and pd.notna(stop) else np.nan
                r_tp1 = (tp1 - trigger)/risk if pd.notna(risk) and risk>0 and pd.notna(tp1) else np.nan
                r_tp2 = (tp2 - trigger)/risk if pd.notna(risk) and risk>0 and pd.notna(tp2) else np.nan
            else:
                trigger = C1L; stop = C2H
                tp1 = C1L - 0.5*rng if pd.notna(rng) else np.nan
                tp2 = C1L - 1.0*rng if pd.notna(rng) else np.nan
                risk = stop - trigger if pd.notna(trigger) and pd.notna(stop) else np.nan
                r_tp1 = (trigger - tp1)/risk if pd.notna(risk) and risk>0 and pd.notna(tp1) else np.nan
                r_tp2 = (trigger - tp2)/risk if pd.notna(risk) and risk>0 and pd.notna(tp2) else np.nan

            if key_on and key_require and not confluence:
                continue

            rows.append({
                "Ticker": yt, "Spółka": meta_map.get(yt,{}).get("company", yt.replace(".WA","")),
                "Grupa": meta_map.get(yt,{}).get("group",""),
                "Kierunek": rec["direction"],
                "C1": c1_ts.date() if pd.notna(c1_ts) else pd.NaT,
                "C2": c2_ts.date() if pd.notna(c2_ts) else pd.NaT,
                "C3_any": (pd.to_datetime(rec.get("C3_date_any")).date() if pd.notna(rec.get("C3_date_any")) else pd.NaT),
                "Potwierdzenie_w_N": "TAK" if rec.get("confirmed", False) else "NIE",
                "C3_happened": "TAK" if rec.get("c3_happened", False) else "NIE",
                "Zasada potwierdzenia": rec["confirm_rule"],
                "C1L": round(C1L,2), "C1H": round(C1H,2),
                "Mid(50%)": round(rec["C1_mid"],2), "C1O": round(rec["C1_open"],2),
                "C2L": round(C2L,2), "C2H": round(C2H,2), "C2C": round(C2C,2),
                "C2 pos w C1%": round(100*rec["C2_position_in_range"],1) if pd.notna(rec["C2_position_in_range"]) else np.nan,
                "Sweep": rec["swept_side"],
                "Trigger": round(trigger,2) if pd.notna(trigger) else np.nan,
                "Stop": round(stop,2) if pd.notna(stop) else np.nan,
                "TP1": round(tp1,2) if pd.notna(tp1) else np.nan,
                "TP2": round(tp2,2) if pd.notna(tp2) else np.nan,
                "R:TP1": round(r_tp1,2) if pd.notna(r_tp1) else np.nan,
                "R:TP2": round(r_tp2,2) if pd.notna(r_tp2) else np.nan,
                "KeyTF": key_tf_str,
                "KeyLevel": round(key_level_val,2) if pd.notna(key_level_val) else np.nan,
                "KeyDate": key_date.date() if pd.notna(key_date) else pd.NaT,
                "Confluence": "TAK" if confluence else "NIE" if key_on else "-",
            })

    except Exception as e:
        rows.append({
            "Ticker": yt, "Spółka": meta_map.get(yt,{}).get("company", yt.replace(".WA","")),
            "Grupa": meta_map.get(yt,{}).get("group",""),
            "Kierunek": "ERR", "C1": pd.NaT, "C2": pd.NaT, "C3_any": pd.NaT,
            "Potwierdzenie_w_N": "ERR", "C3_happened": "ERR", "Zasada potwierdzenia": str(e),
            "C1L": np.nan, "C1H": np.nan, "Mid(50%)": np.nan, "C1O": np.nan,
            "C2L": np.nan, "C2H": np.nan, "C2C": np.nan, "C2 pos w C1%": np.nan, "Sweep": "-",
            "Trigger": np.nan, "Stop": np.nan, "TP1": np.nan, "TP2": np.nan, "R:TP1": np.nan, "R:TP2": np.nan,
            "KeyTF": "-", "KeyLevel": np.nan, "KeyDate": pd.NaT, "Confluence": "-",
        })

progress.empty()
out_df = pd.DataFrame(rows)
if not out_df.empty:
    out_df["C2_sort"] = pd.to_datetime(out_df["C2"], errors="coerce")
    out_df = out_df.sort_values(by=["C2_sort","Grupa","Ticker"], ascending=[False,True,True]).drop(columns=["C2_sort"])
    st.dataframe(out_df, use_container_width=True, height=560)
    st.download_button("📥 Pobierz wyniki (CSV)", data=out_df.to_csv(index=False).encode("utf-8"),
                       file_name=f"crt_scan_{dt.date.today().isoformat()}.csv", mime="text/csv")
else:
    st.info("Brak wyników dla bieżących ustawień.")

st.divider()
col_chart_btn, _ = st.columns([1,3])
with col_chart_btn:
    if "show_chart" not in st.session_state: st.session_state["show_chart"]=False
    if st.button("Pokaż wykres" if not st.session_state["show_chart"] else "Ukryj wykres"):
        st.session_state["show_chart"] = not st.session_state["show_chart"]; st.rerun()

if st.session_state.get("show_chart", False):
    st.subheader("📊 Wykres (W1, z poziomami)")
    if not out_df.empty:
        valid_mask = out_df["Kierunek"].isin(["BULL","BEAR"])
        tickers_for_chart = out_df.loc[valid_mask, "Ticker"].dropna().unique().tolist() or out_df["Ticker"].dropna().unique().tolist()
    else:
        tickers_for_chart = []
    if not tickers_for_chart:
        st.info("Brak tickerów do wyświetlenia na wykresie.")
    else:
        sel = st.selectbox("Wybierz ticker do wykresu (W1)", options=tickers_for_chart, key="chart_ticker")
        rec = out_df[out_df["Ticker"] == sel].sort_values(by="C2", ascending=False).iloc[0].to_dict()
        # Single fetch to avoid caching issue with batch map for new selection
        from data_io import load_weekly_ohlcv
        weekly = load_weekly_ohlcv(sel, period="max")
        if weekly is None or weekly.empty:
            st.info("Brak danych tygodniowych do rysowania wykresu.")
        else:
            st.plotly_chart(build_plotly_chart(weekly, rec, sel), use_container_width=True)
