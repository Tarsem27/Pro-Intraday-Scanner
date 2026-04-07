
# ==============================
# file: data_utils.py
# ==============================
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(close, 12) - ema(close, 26)
    signal = ema(line, 9)
    hist = line - signal
    return line, signal, hist


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    typical = (out["High"] + out["Low"] + out["Close"]) / 3
    vol = out["Volume"].fillna(0)
    out["VWAP"] = (typical * vol).cumsum() / vol.cumsum().replace(0, np.nan)
    return out


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Volume" not in out.columns:
        out["Volume"] = 0
    out = add_vwap(out)
    out["EMA9"] = ema(out["Close"], 9)
    out["EMA20"] = ema(out["Close"], 20)
    out["SMA50"] = sma(out["Close"], 50)
    out["RSI14"] = rsi(out["Close"], 14)
    out["ATR14"] = atr(out, 14)
    out["MACD"], out["MACD_SIGNAL"], out["MACD_HIST"] = macd(out["Close"])
    out["RollingVol20"] = out["Volume"].rolling(20).mean().replace(0, np.nan)
    out["RelVol"] = out["Volume"] / out["RollingVol20"]
    out["SpreadProxyPct"] = (out["High"] - out["Low"]).rolling(3).mean() / out["Close"].replace(0, np.nan) * 100
    return out


def normalize_score(x: float, low: float, high: float) -> float:
    if pd.isna(x):
        return 0.0
    if x <= low:
        return 0.0
    if x >= high:
        return 1.0
    return (x - low) / (high - low)


def parse_symbol_input(raw: str) -> List[str]:
    items = []
    for p in raw.replace("\n", ",").split(","):
        token = p.strip().upper()
        if token:
            items.append(token)
    return list(dict.fromkeys(items))


@st.cache_data(ttl=120, show_spinner=False)
def download_history(symbol: str, period: str, interval: str, include_prepost: bool) -> pd.DataFrame:
    try:
        df = yf.download(
            tickers=symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            prepost=include_prepost,
            threads=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.rename(columns=str.title)
        keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
        return df[keep].dropna(how="all")
    except Exception:
        return pd.DataFrame()


def regime_from_symbol(symbol: str, period: str = "1d", interval: str = "5m") -> Dict:
    df = download_history(symbol, period=period, interval=interval, include_prepost=True)
    if df.empty or len(df) < 25:
        return {"symbol": symbol, "state": "UNKNOWN", "bias": 0, "change_pct": np.nan}

    df = add_vwap(df)
    df["EMA9"] = ema(df["Close"], 9)
    df["EMA20"] = ema(df["Close"], 20)
    price = float(df["Close"].iloc[-1])
    session_open = float(df["Open"].iloc[0])
    change_pct = ((price - session_open) / session_open * 100) if session_open else np.nan
    above_vwap = price > float(df["VWAP"].iloc[-1]) if pd.notna(df["VWAP"].iloc[-1]) else False
    trend_up = bool(df["EMA9"].iloc[-1] > df["EMA20"].iloc[-1])
    intraday_range = (df["High"].max() - df["Low"].min()) / price * 100 if price else 0

    if abs(change_pct) < 0.35 and intraday_range < 1.0:
        state = "CHOPPY"
        bias = 0
    elif change_pct >= 0.35 and above_vwap and trend_up:
        state = "TRENDING UP"
        bias = 1
    elif change_pct <= -0.35 and (not above_vwap) and (not trend_up):
        state = "TRENDING DOWN"
        bias = -1
    else:
        state = "MIXED"
        bias = 0

    return {"symbol": symbol, "state": state, "bias": bias, "change_pct": round(change_pct, 2)}


def detect_market_regime() -> Dict:
    spy = regime_from_symbol("SPY")
    qqq = regime_from_symbol("QQQ")
    vix = download_history("^VIX", period="5d", interval="30m", include_prepost=False)
    vix_last = float(vix["Close"].iloc[-1]) if not vix.empty else np.nan

    score = spy["bias"] + qqq["bias"]
    if score >= 2:
        overall = "RISK-ON TREND UP"
    elif score <= -2:
        overall = "RISK-OFF TREND DOWN"
    elif spy["state"] == "CHOPPY" and qqq["state"] == "CHOPPY":
        overall = "CHOPPY"
    else:
        overall = "MIXED"

    return {
        "overall": overall,
        "spy": spy,
        "qqq": qqq,
        "vix": round(vix_last, 2) if pd.notna(vix_last) else np.nan,
    }


