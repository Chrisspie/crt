# -*- coding: utf-8 -*-
from __future__ import annotations
import pandas as pd, re
from data_io import get_wig_all, get_wig20_hardcoded, get_mwig40_hardcoded, fetch_sp500_companies

# Common GPW alias fixes for yfinance
GPW_ALIASES = {
    "DIN": "DNP",   # Dino
    "BUD": "BDX",   # Budimex
    "AMX": "AMC",   # Amica
    "CIGAMES": "CIG",  # CI Games
    "LIVE": "LVC",  # LiveChat
    "PEK": "PBX",   # Pekabex
}

def _fix_gpw_symbol(sym: str) -> str:
    s = sym.strip().upper()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^A-Z0-9\.\^]", "", s)
    if s in GPW_ALIASES:
        s = GPW_ALIASES[s]
    return s

def parse_us_tickers(raw: str) -> pd.DataFrame:
    if not raw: return pd.DataFrame(columns=["company","yahoo_ticker","group"])
    parts=[p.strip().upper() for p in str(raw).replace(";",",").split(",")]; parts=[p for p in parts if p]
    if not parts: return pd.DataFrame(columns=["company","yahoo_ticker","group"])
    df=pd.DataFrame({"company":parts,"yahoo_ticker":parts}); df["group"]="US (manual)"; return df

def parse_gpw_tickers(raw: str) -> pd.DataFrame:
    if not raw: return pd.DataFrame(columns=["company","yahoo_ticker","group"])
    parts=[_fix_gpw_symbol(p) for p in str(raw).replace(";",",").split(",") if p.strip()]
    if not parts: return pd.DataFrame(columns=["company","yahoo_ticker","group"])
    def _to_yf(sym:str)->str:
        if sym.startswith("^"): return sym
        if "." in sym: return sym
        return sym + ".WA"
    df=pd.DataFrame({"company":parts})
    df["yahoo_ticker"]=[_to_yf(x) for x in df["company"].astype(str)]; df["group"]="GPW (manual)"; return df

def build_universe_df(use_wig_all: bool, use_wig20: bool, use_mwig40: bool, use_sp500: bool, gpw_raw: str, raw_us: str) -> pd.DataFrame:
    frames=[]
    if use_wig_all:
        wig_all=get_wig_all(); 
        if not wig_all.empty: frames.append(wig_all[["company","yahoo_ticker","group"]])
    if use_wig20:
        wig20=get_wig20_hardcoded(); 
        if not wig20.empty: frames.append(wig20[["company","yahoo_ticker","group"]])
    if use_mwig40:
        mwig40=get_mwig40_hardcoded(); 
        if not mwig40.empty: frames.append(mwig40[["company","yahoo_ticker","group"]])
    gpw_manual=parse_gpw_tickers(gpw_raw); 
    if not gpw_manual.empty: frames.append(gpw_manual[["company","yahoo_ticker","group"]])
    if use_sp500:
        spx=fetch_sp500_companies(); 
        if not spx.empty: frames.append(spx[["company","yahoo_ticker","group"]])
    usdf=parse_us_tickers(raw_us); 
    if not usdf.empty: frames.append(usdf[["company","yahoo_ticker","group"]])
    if not frames: return pd.DataFrame(columns=["company","yahoo_ticker","group"])
    uni=pd.concat(frames,ignore_index=True).drop_duplicates(subset=["yahoo_ticker"]).reset_index(drop=True)
    uni["Active"]=True; uni["company"]=uni["company"].fillna(uni["yahoo_ticker"].str.replace(".WA","",regex=False))
    return uni
