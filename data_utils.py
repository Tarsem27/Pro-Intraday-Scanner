
# ==============================
# file: data_utils.py
# ==============================
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Yahoo Finance (via yfinance) intraday limits — request at most this much history
# in one call so we do not rely on retries or partial failures. (Approximate; see yfinance docs.)
_YAHOO_MAX_PERIOD_BY_INTERVAL: Dict[str, str] = {
    "1m": "7d",
    "2m": "59d",
    "3m": "59d",
    "5m": "59d",
    "15m": "59d",
    "30m": "59d",
    "60m": "730d",
    "90m": "730d",
    "1h": "730d",
}


def _period_to_days_approx(period: str) -> float:
    p = (period or "1mo").strip().lower()
    if p in ("ytd", "max"):
        return 3650.0
    if p.endswith("d"):
        return float(p[:-1] or 1)
    if p.endswith("mo"):
        return float(p[:-2] or 1) * 30.0
    if p.endswith("y"):
        return float(p[:-1] or 1) * 365.0
    return 30.0


def max_yahoo_period_for_interval(interval: str) -> str:
    """Longest history Yahoo reliably serves for this bar size (one-shot request)."""
    return _YAHOO_MAX_PERIOD_BY_INTERVAL.get(interval, "2y")


def clamp_yahoo_period(period: str, interval: str) -> str:
    """Use the shorter of the requested window and Yahoo's cap for this interval."""
    cap = max_yahoo_period_for_interval(interval)
    if _period_to_days_approx(period) <= _period_to_days_approx(cap):
        return period
    return cap


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Build higher timeframe bars from intraday OHLCV (one local transform, no extra Yahoo calls)."""
    if df.empty:
        return df
    need = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    if not need:
        return pd.DataFrame(index=df.index)
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    sub = df[need].copy()
    out = sub.resample(rule).agg({k: agg[k] for k in need if k in agg}).dropna(how="all")
    return out


def _coerce_numeric_series(df: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")

    data = df[column]
    if isinstance(data, pd.DataFrame):
        if data.empty:
            return pd.Series(default, index=df.index, dtype="float64")
        data = data.bfill(axis=1).iloc[:, 0]

    return pd.to_numeric(data, errors="coerce")


def _normalize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index.copy())
    for column in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if column in df.columns:
            out[column] = _coerce_numeric_series(df, column, default=0.0 if column == "Volume" else np.nan)
    return out


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
    high = _coerce_numeric_series(df, "High")
    low = _coerce_numeric_series(df, "Low")
    close = _coerce_numeric_series(df, "Close")
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    out = _normalize_price_frame(df)
    high = _coerce_numeric_series(out, "High")
    low = _coerce_numeric_series(out, "Low")
    close = _coerce_numeric_series(out, "Close")
    vol = _coerce_numeric_series(out, "Volume", default=0.0).fillna(0)
    typical = (high + low + close) / 3
    out["VWAP"] = (typical * vol).cumsum() / vol.cumsum().replace(0, np.nan)
    return out


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = _normalize_price_frame(df)
    if "Volume" not in out.columns:
        out["Volume"] = 0
    out = add_vwap(out)
    close = _coerce_numeric_series(out, "Close")
    high = _coerce_numeric_series(out, "High")
    low = _coerce_numeric_series(out, "Low")
    volume = _coerce_numeric_series(out, "Volume", default=0.0)
    out["EMA9"] = ema(close, 9)
    out["EMA20"] = ema(close, 20)
    out["SMA50"] = sma(close, 50)
    out["RSI14"] = rsi(close, 14)
    out["ATR14"] = atr(out, 14)
    out["MACD"], out["MACD_SIGNAL"], out["MACD_HIST"] = macd(close)
    roll = volume.rolling(20, min_periods=5).mean()
    roll = roll.replace(0, np.nan)
    rv = volume / roll
    rv = rv.replace([np.inf, -np.inf], np.nan)
    rv = rv.ffill(limit=15).bfill(limit=15).fillna(1.0)
    rv = rv.clip(lower=0.01, upper=100.0)
    if len(rv) >= 2 and float(volume.iloc[-1] or 0) <= 0:
        rv = rv.copy()
        pv = rv.iloc[-2]
        rv.iloc[-1] = float(pv) if pd.notna(pv) and float(pv) > 0 else 1.0
    out["RollingVol20"] = roll
    out["RelVol"] = rv
    out["SpreadProxyPct"] = (high - low).rolling(3).mean() / close.replace(0, np.nan) * 100
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
    """One Yahoo request per call — period is clamped to provider limits (no retry loop)."""
    period = clamp_yahoo_period(period, interval)
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
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    try:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.rename(columns=str.title)
        keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
        if not keep:
            return pd.DataFrame()
        cleaned = _normalize_price_frame(df[keep])
        return cleaned.dropna(how="all")
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


