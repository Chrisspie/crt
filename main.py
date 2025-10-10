# -*- coding: utf-8 -*-
from __future__ import annotations
import datetime as dt, numpy as np, pandas as pd, streamlit as st

import streamlit.components.v1 as components

TIMEFRAME_SPECS = {
    "W1": {"label": "W1 (tygodniowy)", "interval": "1wk", "tv_interval": "W", "fallback_period": "5y"},
    "D1": {"label": "D1 (dzienny)", "interval": "1d", "tv_interval": "D", "fallback_period": "5y"},
    "H4": {"label": "H4 (4 godziny)", "interval": "240m", "tv_interval": "240", "fallback_period": "730d"},
    "H1": {"label": "H1 (1 godzina)", "interval": "60m", "tv_interval": "60", "fallback_period": "730d"},
    "M15": {"label": "M15 (15 minut)", "interval": "15m", "tv_interval": "15", "fallback_period": "60d"},
}
TIMEFRAME_ORDER = ["W1", "D1", "H4", "H1", "M15"]
DEFAULT_SCAN_TIMEFRAMES = TIMEFRAME_ORDER[:]


def resolve_interval_period(interval: str, requested: str, fallback: str) -> str:
    interval_key = (interval or "").lower()
    requested_val = requested or fallback
    if interval_key in ("1wk", "1w", "1mo", "3mo"):
        return requested_val
    if interval_key == "1d":
        return requested_val
    if interval_key in ("60m", "240m"):
        return requested_val if requested_val == "2y" else fallback
    if interval_key == "15m":
        return fallback
    return fallback

def _round_price(val, *, places: int = 2):
    """Return rounded float or NaN when value is missing."""
    try:
        if val is None or pd.isna(val):
            return np.nan
        return round(float(val), places)
    except Exception:
        return np.nan

from data_io import (
    load_many_weekly_ohlcv,
    load_many_interval_ohlcv,
    load_many_htf_ohlcv,
    get_available_data_sources,
    get_default_source_priority,
    normalize_source_priority,
    get_cache_inventory,
    source_display_name,
    clear_all_cached_data,
)
from universe import build_universe_df
from crt_core import crt_scan, get_key_level_and_confluence
from tv_chart import build_plotly_chart
from datetime import date, timedelta

st.set_page_config(page_title="CRT Scanner – FAST+FIX (HTF touch default)", layout="wide")
st.title("⚡ CRT Scanner – FAST + HTF (touch)")
st.caption("HTF konfluencja: najbliższy poziom (Open/Close/Low/High) z tolerancją. Domyślnie: touch (1%).")


st.session_state.setdefault("show_data_settings", False)
st.session_state.setdefault("data_source_priority", get_default_source_priority())
st.session_state["data_source_priority"] = normalize_source_priority(st.session_state["data_source_priority"])
st.session_state.setdefault("scan_timeframes", DEFAULT_SCAN_TIMEFRAMES[:])
st.session_state["scan_timeframes"] = [tf for tf in TIMEFRAME_ORDER if tf in st.session_state["scan_timeframes"]] or DEFAULT_SCAN_TIMEFRAMES[:]
st.session_state.setdefault("primary_scan_timeframe", st.session_state["scan_timeframes"][0])
if st.session_state["primary_scan_timeframe"] not in st.session_state["scan_timeframes"]:
    st.session_state["primary_scan_timeframe"] = st.session_state["scan_timeframes"][0]
st.session_state.setdefault("scan_enable_all_timeframes", True)


def _toggle_data_settings() -> None:
    st.session_state["show_data_settings"] = not st.session_state.get("show_data_settings", False)

top_cols = st.columns([3, 1])
with top_cols[1]:
    st.button("⚙️ Ustawienia danych", on_click=_toggle_data_settings)

if st.session_state.get("show_data_settings", False):
    with st.container(border=True):
        st.subheader("Ustawienia danych i cache")
        sources_meta = get_available_data_sources()
        st.markdown(
            """**Aktualnie używane darmowe źródła:**
- **Yahoo Finance (yfinance)** – szerokie pokrycie globalnych spółek, darmowe API (limitowane prędkością).
- **Stooq.pl** – bezpłatne dane dzienne/tygodniowe dla GPW i wybranych indeksów, świetne jako fallback.

Inne darmowe alternatywy (wymagają własnego klucza API lub mają ostrzejsze limity): Alpha Vantage, Twelve Data, Finnhub.
"""
        )
        available_keys = list(sources_meta.keys())
        current_priority = normalize_source_priority(st.session_state.get("data_source_priority"))
        primary = st.radio(
            "Preferowane źródło (pierwsze w kolejce)",
            options=available_keys,
            format_func=lambda k: str(sources_meta[k]["label"]),
            index=available_keys.index(current_priority[0]) if current_priority else 0,
            key="data_source_primary",
        )
        ordered = [primary] + [key for key in available_keys if key != primary]
        st.session_state["data_source_priority"] = ordered
        if len(ordered) > 1:
            st.caption("Fallback: " + ", ".join(source_display_name(k) for k in ordered[1:]))

        st.divider()
        st.markdown("**Interwaly skanowania**")
        st.caption("Wybierz interwaly uzywane przy aktywnej opcji wielointerwalowego skanu.")
        st.multiselect("Interwaly (W1=tydzien)", options=TIMEFRAME_ORDER, key="scan_timeframes", format_func=lambda tf: TIMEFRAME_SPECS[tf]["label"])

        cache_df = get_cache_inventory()
        st.markdown("**Cache danych**")
        if cache_df.empty:
            st.info("Cache jest pusty – brak pobranych zestawów.")
        else:
            st.dataframe(cache_df, use_container_width=True, height=min(420, 80 + 26 * len(cache_df)))

current_source_priority = normalize_source_priority(st.session_state.get("data_source_priority"))


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

    st.subheader("Interwaly skanowania")
    available_scan_tf = [tf for tf in st.session_state["scan_timeframes"] if tf in TIMEFRAME_SPECS]
    scan_all_timeframes = st.checkbox("Skanuj wszystkie interwaly", value=st.session_state.get("scan_enable_all_timeframes", True), key="scan_enable_all_timeframes")
    st.selectbox("Interwal bazowy", options=available_scan_tf, key="primary_scan_timeframe", format_func=lambda tf: TIMEFRAME_SPECS[tf]["label"])
    st.caption("Konfiguracja interwalow znajduje sie w panelu ustawien danych.")

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
    # default set to 'touch' (index=1)
    key_rule_label = st.selectbox("Reguła kontaktu", ["strict (<, >)","touch (≤, ≥)"], index=1, disabled=not key_on)
    key_require = st.checkbox("Wymagaj konfluencji (filtruj wyniki)", value=True, disabled=not key_on)

    st.divider()
    if st.button("🧹 Wyczyść cache"):
        clear_all_cached_data(); st.success("Cache wyczyszczony."); st.rerun()

st.session_state.setdefault("active_map", {})
with st.spinner("Ładowanie listy tickerów…"):
    universe_df = build_universe_df(use_wig_all, use_wig20, use_mwig40, use_sp500, gpw_raw, raw_us)
if universe_df.empty:
    st.warning("Brak spółek do skanowania."); st.stop()

active_series = universe_df["yahoo_ticker"].map(st.session_state["active_map"])
universe_df["Active"] = active_series.where(active_series.notna(), True).astype(bool)

panel_caption = st.empty()
with st.expander("🎛️ Panel aktywnych spółek", expanded=True):
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
panel_caption.caption(f"Aktywnych tickerów: **{len(active_tickers)}** / {len(edited_df)}")

if not active_tickers:
    st.info("Zaznacz przynajmniej jedną spółkę."); st.stop()

st.subheader("🔎 Wyniki skanowania CRT")

# Incremental scan with cancel + logs
state = st.session_state.get("scan_state")

def emit_logs(expanded: bool):
    exp = st.expander("Logi skanowania", expanded=expanded)
    box = exp.empty();
    logs = (state or {}).get("logs", [])
    if logs:
        box.code("\n".join(logs[-600:]), language="text")
    return box

def log_msg(msg: str):
    ts = dt.datetime.now().strftime("%H:%M:%S")
    if state is not None:
        state.setdefault("logs", []).append(f"[{ts}] {msg}")
        st.session_state["scan_state"] = state

def render_scan_results(df: pd.DataFrame) -> None:
    """Render scan results table with a matching download button."""
    if df is None or df.empty:
        st.info("Brak wyników dla bieżących ustawień.")
        return
    view_df = df.copy()
    preferred = ["Ticker", "TF", "Interwał", "Spółka", "Grupa", "Kierunek"]
    ordered_cols = [col for col in preferred if col in view_df.columns]
    ordered_cols += [col for col in view_df.columns if col not in ordered_cols]
    view_df = view_df[ordered_cols]
    st.dataframe(view_df, use_container_width=True, height=560)
    st.download_button(
        "📥 Pobierz wyniki (CSV)",
        data=view_df.to_csv(index=False).encode("utf-8"),
        file_name=f"crt_scan_{dt.date.today().isoformat()}.csv",
        mime="text/csv",
    )




# Start button (only when not running)
if not state or not state.get("running"):
    start_scan = st.button("Rozpocznij skanowanie")
    logs_box = emit_logs(expanded=False)
    if start_scan:
        logs_box.empty()
        source_priority_current = current_source_priority[:]
        tickers = active_tickers[:]
        configured_timeframes = [tf for tf in st.session_state["scan_timeframes"] if tf in TIMEFRAME_SPECS]
        if not configured_timeframes:
            configured_timeframes = DEFAULT_SCAN_TIMEFRAMES[:]
        primary_timeframe = st.session_state.get("primary_scan_timeframe", configured_timeframes[0])
        if primary_timeframe not in configured_timeframes:
            primary_timeframe = configured_timeframes[0]
            st.session_state["primary_scan_timeframe"] = primary_timeframe
        multi_timeframes = st.session_state.get("scan_enable_all_timeframes", True)
        active_timeframes = configured_timeframes if multi_timeframes else [primary_timeframe]
        params = dict(
            directions=directions, lookback_bars=lookback_bars, require_midline=require_midline,
            strict_vs_c1open=strict_vs_c1open, confirm_on=confirm_on,
            confirm_within=int(confirm_within), confirm_method=confirm_method,
            opportunity_mode=opportunity_mode, period=period,
            key_on=key_on, key_tf=("1mo" if key_tf_label.startswith("1M") else "3mo"),
            key_window_months=int(key_window_months), key_interact=key_interact, key_rule_label=key_rule_label,
            key_require=bool(key_require),
            data_sources=source_priority_current,
            multi_timeframes=multi_timeframes,
            scan_timeframes=active_timeframes,
            configured_timeframes=configured_timeframes,
            primary_timeframe=primary_timeframe,
        )
        # Prepare data in advance
        timeframe_labels = ", ".join(TIMEFRAME_SPECS[tf]["label"] for tf in active_timeframes) if active_timeframes else TIMEFRAME_SPECS[params["primary_timeframe"]]["label"]
        data_progress = st.progress(0.0, text="Przygotowywanie danych do skanu...")
        start_date = None
        data_map = {}
        failure_map = {}
        with st.spinner("Ladowanie danych dla skanu..."):
            if params["opportunity_mode"]:
                effective_weeks = 2 + params["confirm_within"] + 6 if params["confirm_on"] else 2 + 6
                start_date = (date.today() - timedelta(weeks=effective_weeks)).isoformat()
                base_period = "max"
            else:
                base_period = params["period"]
            tf_list = active_timeframes or [params["primary_timeframe"]]
            for tf_idx, tf in enumerate(tf_list):
                spec = TIMEFRAME_SPECS.get(tf, {})
                interval = spec.get("interval", "1wk")
                tf_label = spec.get("label", tf)
                fallback_period = spec.get("fallback_period", base_period)
                if params["opportunity_mode"]:
                    tf_period = "max" if interval in ("1wk", "1d") else fallback_period
                    tf_start = start_date
                else:
                    tf_period = resolve_interval_period(interval, base_period, fallback_period)
                    tf_start = None
                step_progress = 0.1 + 0.4 * ((tf_idx + 1) / max(1, len(tf_list)))
                data_progress.progress(step_progress, text=f"Ladowanie danych ({tf_label})...")
                tf_data = load_many_interval_ohlcv(
                    tickers,
                    interval=interval,
                    period=tf_period,
                    start=tf_start,
                    retries=1,
                    source_priority=source_priority_current,
                )
                failure_map[tf] = tf_data.pop("__failed__", pd.Series([], dtype=str))
                data_map[tf] = tf_data
            data_progress.progress(0.6, text="Wczytywanie danych HTF...")
            htf_map = load_many_htf_ohlcv(
                tickers, interval=(params["key_tf"]), period="max", retries=1, source_priority=source_priority_current
            ) if params["key_on"] else {}
        data_progress.progress(1.0, text="Dane zaladowane.")
        data_progress.empty()

        if params["opportunity_mode"]:
            log_first = f"Start skanowania (opportunity; {timeframe_labels}). Tickerow: {len(tickers)}. Start: {start_date}."
        else:
            log_first = f"Start skanowania ({params['period']}; {timeframe_labels}). Tickerow: {len(tickers)}."
        # Build initial state
        source_names = ", ".join(source_display_name(src) for src in source_priority_current)
        log_lines = [
            f"[{dt.datetime.now().strftime('%H:%M:%S')}] {log_first}",
            f"[{dt.datetime.now().strftime('%H:%M:%S')}] Zrodla danych (priorytet): {source_names}",
        ]
        for tf in active_timeframes:
            fails = failure_map.get(tf)
            if fails is None or len(fails) == 0:
                continue
            fail_str = ", ".join(fails.tolist()[:40])
            if len(fails) > 40:
                fail_str += "..."
            st.warning(f"Brak danych ({tf}): {fail_str}")
            log_lines.append(f"[{dt.datetime.now().strftime('%H:%M:%S')}] Brak danych ({tf}): {fail_str}")
        scan_targets = [(ticker, tf) for ticker in tickers for tf in active_timeframes]
        st.session_state["scan_state"] = {
            "running": True, "cancel": False,
            "idx": 0, "total": len(scan_targets), "targets": scan_targets,
            "rows": [], "logs": log_lines,
            "params": params, "data_map": data_map, "htf_map": htf_map,
        }
        if params["key_on"]:
            htf_failed = htf_map.get("__failed__")
            if htf_failed is not None and len(htf_failed) > 0:
                miss = ", ".join(htf_failed.tolist()[:40])
                if len(htf_failed) > 40:
                    miss += "..."
                st.info(f"HTF: brak danych dla: {miss}")
                log_lines.append(f"[{dt.datetime.now().strftime('%H:%M:%S')}] Brak danych (HTF): {miss}")
        st.rerun()
else:
    # Running: show cancel, progress, and process one step
    state = st.session_state.get("scan_state")
    cols = st.columns([1,3])
    with cols[0]:
        if st.button("Przerwij", type="primary"):
            state["cancel"] = True; st.session_state["scan_state"] = state; st.rerun()
    logs_box = emit_logs(expanded=True)
    i, total = state.get("idx", 0), state.get("total", 0)
    progress = st.progress(i/max(1,total), text=(f"Skanowanie: {i}/{total}"))

    with st.spinner("Skanowanie w toku…"):
        # Finalize if cancelled or done
        if state.get("cancel") or i >= total:
            out_df = pd.DataFrame(state.get("rows", []))
            state["running"] = False
            st.session_state["scan_state"] = state
            if not out_df.empty:
                out_df["C2_sort"] = pd.to_datetime(out_df["C2"], errors="coerce")
                out_df = out_df.sort_values(by=["C2_sort","Grupa","Ticker"], ascending=[False,True,True]).drop(columns=["C2_sort"])
                log_msg("Zakończono skanowanie." if not state.get("cancel") else "Skanowanie przerwane.")
            else:
                log_msg("Zakończono skanowanie: brak wyników." if not state.get("cancel") else "Skanowanie przerwane: brak wyników.")
            # Cache for chart section
            st.session_state["scan_out_df"] = out_df
        else:
            # Process one target per run
            targets = state.get("targets", [])
            params = state["params"]; data_map = state["data_map"]; htf_map = state["htf_map"]
            if i >= len(targets):
                state["idx"] = len(targets)
                st.session_state["scan_state"] = state
                st.rerun()
            yt, tf = targets[i]
            tf_label = TIMEFRAME_SPECS.get(tf, {}).get("label", tf)
            progress.progress(i/max(1,total), text=f"Skanowanie: {yt} ({tf_label}) [{i+1}/{total}]")
            log_msg(f"Skanuję: {yt} ({tf_label})…")
            try:
                tf_data_map = data_map.get(tf, {})
                df = tf_data_map.get(yt)
                if df is None or df.empty or len(df) < 5:
                    log_msg(f"Pominięto (brak danych): {yt} [{tf}]")
                else:
                    setups = crt_scan(
                        df=df,
                        lookback_bars=(3 if params["opportunity_mode"] else params["lookback_bars"]),
                        require_midline=params["require_midline"],
                        strict_vs_c1open=params["strict_vs_c1open"],
                        confirm_within=(params["confirm_within"] if params["confirm_on"] else 0),
                        confirm_method=(params["confirm_method"] if params["confirm_on"] else "high"),
                        directions=params["directions"],
                        return_targets=True,
                    )
                    htf_df = htf_map.get(yt, pd.DataFrame()) if params["key_on"] else pd.DataFrame()
                    last_two = pd.Index(df.index[-2:])
                    kept = 0
                    for rec in setups:
                        c1_ts = pd.to_datetime(rec["C1_date"]); c2_ts = pd.to_datetime(rec["C2_date"])
                        if params["opportunity_mode"] and ((c2_ts not in last_two) or rec.get("c3_happened", False)):
                            continue
                        C1L, C1H = rec["C1_low"], rec["C1_high"]
                        C2L, C2H, C2C = rec["C2_low"], rec["C2_high"], rec["C2_close"]
                        rng = (C1H - C1L) if pd.notna(C1H) and pd.notna(C1L) else np.nan
                        key_tf_str, key_level_val, key_date, confluence = ("-", float("nan"), pd.NaT, False)
                        if params["key_on"] and not htf_df.empty:
                            key_tf_str, key_level_val, key_date, confluence = get_key_level_and_confluence(
                                htf_df, c2_ts, rec["direction"], C1L, C1H, C2L, C2H,
                                params["key_window_months"], params["key_interact"], params["key_rule_label"], params["key_tf"]
                            )
                        dir_upper = str(rec.get("direction", "")).upper()
                        dir_norm = str(rec.get("direction_label", dir_upper)).lower()
                        rng = (C1H - C1L) if pd.notna(C1H) and pd.notna(C1L) else np.nan
                        tp_mid_val = rec.get("tp1", np.nan)
                        tp_min_val = rec.get("tp_min", np.nan)
                        tp_ext_val = rec.get("tp_ext", np.nan)
                        tp2_val = np.nan
                        if dir_upper == "BULL":
                            trigger = C1H
                            stop = C2L
                            tp2_val = C1H + rng if pd.notna(rng) else np.nan
                            if pd.isna(tp_ext_val) and pd.notna(rng):
                                tp_ext_val = C1H + 0.5 * rng
                        elif dir_upper == "BEAR":
                            trigger = C1L
                            stop = C2H
                            tp2_val = C1L - rng if pd.notna(rng) else np.nan
                            if pd.isna(tp_ext_val) and pd.notna(rng):
                                tp_ext_val = C1L - 0.5 * rng
                        else:
                            trigger = np.nan
                            stop = np.nan
                        tp_mid_val = float(tp_mid_val) if tp_mid_val is not None and not pd.isna(tp_mid_val) else np.nan
                        tp_min_val = float(tp_min_val) if tp_min_val is not None and not pd.isna(tp_min_val) else np.nan
                        tp_ext_val = float(tp_ext_val) if tp_ext_val is not None and not pd.isna(tp_ext_val) else np.nan
                        tp2_val = float(tp2_val) if tp2_val is not None and not pd.isna(tp2_val) else np.nan
                        risk = np.nan
                        if pd.notna(trigger) and pd.notna(stop):
                            if dir_upper == "BULL":
                                candidate = trigger - stop
                            elif dir_upper == "BEAR":
                                candidate = stop - trigger
                            else:
                                candidate = np.nan
                            if pd.notna(candidate) and candidate > 0:
                                risk = candidate
                        r_tp1 = np.nan
                        r_tp_ext = np.nan
                        r_tp2 = np.nan
                        if pd.notna(risk) and risk > 0 and pd.notna(trigger):
                            if pd.notna(tp_mid_val):
                                if dir_upper == "BULL":
                                    r_tp1 = (tp_mid_val - trigger) / risk
                                elif dir_upper == "BEAR":
                                    r_tp1 = (trigger - tp_mid_val) / risk
                            if pd.notna(tp_ext_val):
                                if dir_upper == "BULL":
                                    r_tp_ext = (tp_ext_val - trigger) / risk
                                elif dir_upper == "BEAR":
                                    r_tp_ext = (trigger - tp_ext_val) / risk
                            if pd.notna(tp2_val):
                                if dir_upper == "BULL":
                                    r_tp2 = (tp2_val - trigger) / risk
                                elif dir_upper == "BEAR":
                                    r_tp2 = (trigger - tp2_val) / risk
                        ep_conflict = False
                        if pd.notna(trigger):
                            try:
                                ep_float = float(trigger)
                                if pd.notna(tp_mid_val) and abs(ep_float - float(tp_mid_val)) <= 1e-6:
                                    ep_conflict = True
                                if pd.notna(tp_min_val) and abs(ep_float - float(tp_min_val)) <= 1e-6:
                                    ep_conflict = True
                            except Exception:
                                pass
                        if params["key_on"] and params["key_require"] if "key_require" in params else False:
                            if not confluence:
                                pass  # skip
                        row = {
                            "Ticker": yt, "TF": tf, "Interwa?": tf_label, "Sp??ka": meta_map.get(yt,{}).get("company", yt.replace(".WA","")),
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
                            "EP": _round_price(trigger),
                            "Trigger": _round_price(trigger),
                            "Stop": _round_price(stop),
                            "TP1": _round_price(tp_mid_val),
                            "TP_MIN": _round_price(tp_min_val),
                            "TP_ext": _round_price(tp_ext_val),
                            "TP2": _round_price(tp2_val),
                            "R:TP1": _round_price(r_tp1, places=2),
                            "R:TP_ext": _round_price(r_tp_ext, places=2),
                            "R:TP2": _round_price(r_tp2, places=2),
                            "KeyTF": key_tf_str,
                            "KeyLevel": round(key_level_val,2) if pd.notna(key_level_val) else np.nan,
                            "KeyDate": (pd.to_datetime(key_date).date() if pd.notna(key_date) else pd.NaT),
                            "Confluence": "TAK" if confluence else ("-" if not params["key_on"] else "NIE"),
                        }
                        if ep_conflict:
                            row["EP_conflict_with_target"] = True

                        # Filter by confluence if required
                        if params.get("key_on") and params.get("key_require") and not confluence:
                            pass
                        else:
                            state["rows"].append(row); kept += 1
                    log_msg(f"OK: {yt} [{tf}] – setupów: {len(setups)}, zachowano: {kept}.")
            except Exception as e:
                log_msg(f"Błąd: {yt} [{tf}] – {e}")
            # Advance and rerun
            state["idx"] = i + 1
            st.session_state["scan_state"] = state
            st.rerun()

# Provide out_df to chart section
out_df = st.session_state.get("scan_out_df", pd.DataFrame())

if (not state or not state.get("running")) and "scan_out_df" in st.session_state:
    render_scan_results(out_df)

st.divider()
col_chart_btn, _ = st.columns([1,3])
with col_chart_btn:
    if "show_chart" not in st.session_state: st.session_state["show_chart"]=False
    if st.button("Pokaż wykres" if not st.session_state["show_chart"] else "Ukryj wykres"):
        st.session_state["show_chart"] = not st.session_state["show_chart"]; st.rerun()

if st.session_state.get("show_chart", False):
    st.markdown('<div id="chart-anchor"></div>', unsafe_allow_html=True)
    st.subheader("📊 Wykres (dane ze skanu)")
    scan_state_current = st.session_state.get("scan_state", {})
    scan_params = scan_state_current.get("params", {})
    data_map_all = scan_state_current.get("data_map", {})
    option_lookup = {}
    if not out_df.empty:
        if "TF" in out_df.columns:
            combos_base = out_df.dropna(subset=[col for col in ["Ticker", "TF"] if col in out_df.columns])
            use_cols = [col for col in ["Ticker", "TF", "Interwa?"] if col in combos_base.columns]
            if use_cols:
                combos_df = combos_base.loc[:, use_cols]
            else:
                combos_df = combos_base
            combos_df = combos_df.drop_duplicates().sort_values(by=[c for c in ["Ticker", "TF"] if c in combos_df.columns])
            for _, row in combos_df.iterrows():
                ticker_val = row.get("Ticker")
                tf_code = row.get("TF") or scan_params.get("primary_timeframe", "W1")
                tf_display = row.get("Interwa?") or TIMEFRAME_SPECS.get(tf_code, {}).get("label", tf_code)
                if ticker_val:
                    label = f"{ticker_val} ({tf_code})" if tf_code else ticker_val
                    option_lookup[label] = (ticker_val, tf_code, tf_display)
        else:
            combos_df = (
                out_df.dropna(subset=["Ticker"])
                .loc[:, ["Ticker"]]
                .drop_duplicates()
                .sort_values(by=["Ticker"])
            )
            primary_tf = scan_params.get("primary_timeframe", "W1")
            primary_label = TIMEFRAME_SPECS.get(primary_tf, {}).get("label", primary_tf)
            for _, row in combos_df.iterrows():
                ticker_val = row["Ticker"]
                option_lookup[ticker_val] = (ticker_val, primary_tf, primary_label)
    options = list(option_lookup.keys())

    
    if not options:
        st.info("Brak instrumentów do wyświetlenia na wykresie.")
    else:
        def _on_chart_select_change() -> None:
            st.session_state["scroll_to_chart_anchor"] = True

        sel_label = st.selectbox(
            "Wybierz instrument do wykresu",
            options=options,
            key="chart_target",
            on_change=_on_chart_select_change,
        )
        ticker_sel, tf_sel, tf_display = option_lookup.get(sel_label, (None, None, None))
        if not ticker_sel or not tf_sel:
            st.info("Brak danych do wyświetlenia na wykresie.")
        else:
            mask = out_df["Ticker"] == ticker_sel
            if "TF" in out_df.columns:
                mask &= out_df["TF"] == tf_sel
            rec_candidates = out_df.loc[mask].sort_values(by="C2", ascending=False)
            if rec_candidates.empty:
                st.info("Nie znaleziono rekordu dla wybranego interwału.")
            else:
                rec = rec_candidates.iloc[0].to_dict()
                tf_display = rec.get("Interwał") or tf_display or tf_sel
                chart_sources = normalize_source_priority(
                    scan_params.get("data_sources", current_source_priority)
                )
                chart_df = pd.DataFrame()
                if isinstance(data_map_all, dict):
                    chart_df = data_map_all.get(tf_sel, {}).get(ticker_sel, pd.DataFrame())
                if chart_df is None or chart_df.empty:
                    spec_default = TIMEFRAME_SPECS.get("W1") or next(iter(TIMEFRAME_SPECS.values()))
                    spec = TIMEFRAME_SPECS.get(tf_sel, spec_default)
                    interval = spec.get("interval", "1wk")
                    fallback_period = spec.get("fallback_period", scan_params.get("period", "5y"))
                    base_period = scan_params.get("period", "5y")
                    effective_period = resolve_interval_period(interval, base_period, fallback_period)
                    fetched = load_many_interval_ohlcv(
                        [ticker_sel],
                        interval=interval,
                        period=effective_period,
                        start=None,
                        retries=1,
                        source_priority=chart_sources,
                    )
                    chart_df = fetched.get(ticker_sel, pd.DataFrame())
                if chart_df is None or chart_df.empty:
                    st.info("Brak danych dla wybranego interwału do rysowania wykresu.")
                else:
                    st.plotly_chart(build_plotly_chart(chart_df, rec, ticker_sel), use_container_width=True)
                    st.caption(f"Interwał: {tf_display}")
    if st.session_state.pop("scroll_to_chart_anchor", False):
        components.html(
            "<script>const anchor=window.parent.document.getElementById('chart-anchor'); if(anchor){anchor.scrollIntoView({behavior:'instant', block:'start'});}</script>",
            height=0,
        )
