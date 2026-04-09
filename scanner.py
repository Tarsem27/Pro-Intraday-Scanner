# ==============================
# file: scanner.py
# ==============================
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from data_utils import add_indicators, download_history, normalize_score
from market_intel import get_symbol_intel


READINESS_LOOKBACK_HOURS = 48
RSI_NEUTRAL_LOW = 45
RSI_NEUTRAL_HIGH = 55
MIN_SETUP_RELVOL = 0.9
MIN_TRIGGER_RELVOL = 1.3
MIN_VWAP_DISTANCE_PCT = 0.15
MIN_QUALITY_SCORE = 4
BREAKOUT_BUFFER_PCT = 0.0007
STOP_BUFFER_PCT = 0.001
DEFAULT_SCAN_MODE = "Exploratory"
DEFAULT_HISTORY_MODE = "Exploratory"
SCANNER_MODES = {
    "Exploratory": {
        "rsi_neutral_low": RSI_NEUTRAL_LOW,
        "rsi_neutral_high": RSI_NEUTRAL_HIGH,
        "block_neutral_rsi": False,
        "min_setup_relvol": 0.45,
        "min_trigger_relvol": 0.85,
        "min_vwap_distance_pct": 0.012,
        "min_quality_score": 0,
        "breakout_buffer_pct": 0.0002,
        "close_strength_min": 0.38,
        "quality_relvol_threshold": 0.85,
        "quality_vwap_distance_pct": 0.04,
        "hard_block_choppy": False,
        "hard_block_high_event": False,
    },
    "Balanced": {
        "rsi_neutral_low": 48,
        "rsi_neutral_high": 52,
        "block_neutral_rsi": True,
        "min_setup_relvol": 0.65,
        "min_trigger_relvol": 1.05,
        "min_vwap_distance_pct": 0.05,
        "min_quality_score": 2,
        "breakout_buffer_pct": 0.00045,
        "close_strength_min": 0.5,
        "quality_relvol_threshold": 1.0,
        "quality_vwap_distance_pct": 0.1,
        "hard_block_choppy": False,
        "hard_block_high_event": False,
    },
    "Strict": {
        "rsi_neutral_low": RSI_NEUTRAL_LOW,
        "rsi_neutral_high": RSI_NEUTRAL_HIGH,
        "block_neutral_rsi": True,
        "min_setup_relvol": MIN_SETUP_RELVOL,
        "min_trigger_relvol": MIN_TRIGGER_RELVOL,
        "min_vwap_distance_pct": MIN_VWAP_DISTANCE_PCT,
        "min_quality_score": MIN_QUALITY_SCORE,
        "breakout_buffer_pct": BREAKOUT_BUFFER_PCT,
        "close_strength_min": 0.6,
        "quality_relvol_threshold": 1.2,
        "quality_vwap_distance_pct": 0.2,
        "hard_block_choppy": False,
        "hard_block_high_event": True,
    },
}


def _get_mode_config(mode: Optional[str]) -> Dict:
    return SCANNER_MODES.get(mode or DEFAULT_SCAN_MODE, SCANNER_MODES[DEFAULT_SCAN_MODE])


def support_resistance(df: pd.DataFrame) -> Dict:
    recent = df.tail(30)
    if recent.empty:
        return {"support": np.nan, "resistance": np.nan, "prev_high": np.nan, "prev_low": np.nan}
    prev = df.iloc[:-1] if len(df) > 1 else df
    return {
        "support": round(float(recent["Low"].min()), 4),
        "resistance": round(float(recent["High"].max()), 4),
        "prev_high": round(float(prev["High"].max()), 4),
        "prev_low": round(float(prev["Low"].min()), 4),
    }


def mtf_confirmation(symbol: str, include_prepost: bool) -> Dict:
    tf5 = download_history(symbol, period="1d", interval="5m", include_prepost=include_prepost)
    tf15 = download_history(symbol, period="5d", interval="15m", include_prepost=include_prepost)
    tf60 = download_history(symbol, period="1mo", interval="60m", include_prepost=include_prepost)

    def bias(df: pd.DataFrame) -> int:
        if df.empty or len(df) < 30:
            return 0
        x = add_indicators(df)
        last = x.iloc[-1]
        bull = bool(last["Close"] > last["VWAP"] and last["EMA9"] > last["EMA20"])
        bear = bool(last["Close"] < last["VWAP"] and last["EMA9"] < last["EMA20"])
        if bull:
            return 1
        if bear:
            return -1
        return 0

    b5 = bias(tf5)
    b15 = bias(tf15)
    b60 = bias(tf60)
    if b5 == b15 == b60 == 1:
        state = "BULLISH ALIGNED"
    elif b5 == b15 == b60 == -1:
        state = "BEARISH ALIGNED"
    else:
        state = "MIXED"

    return {"bias_5m": b5, "bias_15m": b15, "bias_60m": b60, "state": state}


def _effective_relvol(last: pd.Series, prev: pd.Series) -> float:
    """Last-bar volume is often 0 outside RTH (stale Yahoo quote); impute for gating and triggers."""
    vol = float(last.get("Volume", 0) or 0)
    r = last.get("RelVol", np.nan)
    try:
        rf = float(r) if pd.notna(r) else np.nan
    except (TypeError, ValueError):
        rf = np.nan
    if vol > 0 and pd.notna(rf) and rf > 0:
        return rf
    pr = prev.get("RelVol", np.nan)
    try:
        pf = float(pr) if pd.notna(pr) else np.nan
    except (TypeError, ValueError):
        pf = np.nan
    if pd.notna(pf) and pf > 0:
        return pf
    return 1.0


def build_trade_plan(signal: str, price: float, atr_value: float, levels: Dict) -> Dict:
    atr_value = 0 if pd.isna(atr_value) else float(atr_value)
    support = levels.get("support", np.nan)
    resistance = levels.get("resistance", np.nan)
    structure_buffer = max(price * STOP_BUFFER_PCT, atr_value * 0.15)

    if signal == "LONG":
        entry = price
        if pd.notna(support) and support < entry:
            stop = support - structure_buffer
        else:
            stop = price - max(atr_value, structure_buffer)
        risk = max(entry - stop, 1e-6)
        target1 = entry + 2 * risk
        target2 = resistance if pd.notna(resistance) and resistance > entry else entry + 3 * risk
    else:
        entry = price
        if pd.notna(resistance) and resistance > entry:
            stop = resistance + structure_buffer
        else:
            stop = price + max(atr_value, structure_buffer)
        risk = max(stop - entry, 1e-6)
        target1 = entry - 2 * risk
        target2 = support if pd.notna(support) and support < entry else entry - 3 * risk

    rr1 = abs((target1 - entry) / risk) if risk else np.nan
    rr2 = abs((target2 - entry) / risk) if risk else np.nan
    return {
        "entry": round(entry, 4),
        "stop": round(stop, 4),
        "target1": round(target1, 4),
        "target2": round(target2, 4),
        "rr1": round(rr1, 2) if pd.notna(rr1) else np.nan,
        "rr2": round(rr2, 2) if pd.notna(rr2) else np.nan,
    }


def entry_trigger(signal: str, df: pd.DataFrame, levels: Dict, mode: str = DEFAULT_SCAN_MODE) -> Dict:
    if df.empty or len(df) < 3:
        return {"ready": False, "trigger_text": "insufficient trigger data"}

    mode_config = _get_mode_config(mode)
    last = df.iloc[-1]
    prev = df.iloc[-2]
    relvol = _effective_relvol(last, prev)
    close_price = float(last["Close"])
    open_price = float(last["Open"])
    high_price = float(last["High"])
    low_price = float(last["Low"])
    atr_value = float(last["ATR14"]) if pd.notna(last.get("ATR14")) else 0.0
    candle_range = max(high_price - low_price, 1e-6)
    close_strength = (close_price - low_price) / candle_range
    breakdown_strength = (high_price - close_price) / candle_range
    breakout_buffer = max(close_price * mode_config["breakout_buffer_pct"], atr_value * 0.1)
    prev_high = levels.get("prev_high", np.nan)
    prev_low = levels.get("prev_low", np.nan)

    if signal == "LONG":
        threshold = max(float(prev["High"]), float(prev_high) if pd.notna(prev_high) else float(prev["High"])) + breakout_buffer
        ready = bool(
            last["Close"] > last["VWAP"]
            and close_price >= threshold
            and high_price >= threshold
            and relvol >= mode_config["min_trigger_relvol"]
            and close_price > open_price
            and close_strength >= mode_config["close_strength_min"]
        )
        text = f"LONG ready above {threshold:.4f}" if ready else "long not ready"
    else:
        threshold = min(float(prev["Low"]), float(prev_low) if pd.notna(prev_low) else float(prev["Low"])) - breakout_buffer
        ready = bool(
            last["Close"] < last["VWAP"]
            and close_price <= threshold
            and low_price <= threshold
            and relvol >= mode_config["min_trigger_relvol"]
            and close_price < open_price
            and breakdown_strength >= mode_config["close_strength_min"]
        )
        text = f"SHORT ready below {threshold:.4f}" if ready else "short not ready"

    return {"ready": ready, "trigger_text": text}


def _quality_score(
    signal: str,
    *,
    above_vwap: bool,
    ema_bull: bool,
    ema_bear: bool,
    mtf_state: str,
    relvol: float,
    vwap_distance_pct: float,
    spread_proxy: float,
    liquidity_label: str,
    event_risk: str,
    mode: str = DEFAULT_SCAN_MODE,
) -> int:
    mode_config = _get_mode_config(mode)
    score = 0
    if signal == "LONG":
        score += int(above_vwap)
        score += int(ema_bull)
        score += int(mtf_state == "BULLISH ALIGNED")
    else:
        score += int(not above_vwap)
        score += int(ema_bear)
        score += int(mtf_state == "BEARISH ALIGNED")
    score += int(relvol >= mode_config["quality_relvol_threshold"])
    score += int(pd.notna(vwap_distance_pct) and vwap_distance_pct >= mode_config["quality_vwap_distance_pct"])
    score += int(pd.notna(spread_proxy) and spread_proxy <= 1.2)
    score += int(liquidity_label == "HIGH")
    score -= int(event_risk == "MEDIUM")
    return max(score, 0)


def _history_period_for_interval(interval: str) -> str:
    if interval == "1m":
        return "7d"
    if interval in {"2m", "5m", "15m", "30m"}:
        return "2mo"
    if interval == "60m":
        return "3mo"
    return "6mo"


def analyze_symbol(symbol: str, period: str, interval: str, include_prepost: bool, regime: Dict, mode: str = DEFAULT_SCAN_MODE) -> Dict:
    raw = download_history(symbol, period=period, interval=interval, include_prepost=include_prepost)
    if raw.empty or len(raw) < 50:
        return {"symbol": symbol, "status": "insufficient_data"}

    mode_config = _get_mode_config(mode)
    df = add_indicators(raw.dropna(subset=["Open", "High", "Low", "Close"]))
    if df.empty or len(df) < 50:
        return {"symbol": symbol, "status": "insufficient_data"}

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    price = float(latest["Close"])
    session_open = float(df["Open"].iloc[0])
    session_high = float(df["High"].max())
    session_low = float(df["Low"].min())
    change_pct = ((price - session_open) / session_open * 100) if session_open else np.nan
    relvol = _effective_relvol(latest, prev)
    rsi_now = float(latest["RSI14"]) if pd.notna(latest["RSI14"]) else np.nan
    atr_value = float(latest["ATR14"]) if pd.notna(latest["ATR14"]) else np.nan
    atr_pct = (atr_value / price * 100) if price else np.nan
    above_vwap = bool(price > float(latest["VWAP"])) if pd.notna(latest["VWAP"]) else False
    vwap_distance_pct = abs(price - float(latest["VWAP"])) / price * 100 if price and pd.notna(latest["VWAP"]) else np.nan
    ema_bull = bool(latest["EMA9"] > latest["EMA20"] > latest["SMA50"]) if pd.notna(latest["SMA50"]) else False
    ema_bear = bool(latest["EMA9"] < latest["EMA20"] < latest["SMA50"]) if pd.notna(latest["SMA50"]) else False
    macd_up = bool(latest["MACD_HIST"] > prev["MACD_HIST"] and latest["MACD_HIST"] > 0) if pd.notna(latest["MACD_HIST"]) and pd.notna(prev["MACD_HIST"]) else False
    macd_down = bool(latest["MACD_HIST"] < prev["MACD_HIST"] and latest["MACD_HIST"] < 0) if pd.notna(latest["MACD_HIST"]) and pd.notna(prev["MACD_HIST"]) else False
    dist_high_pct = ((session_high - price) / price * 100) if price else np.nan
    dist_low_pct = ((price - session_low) / price * 100) if price else np.nan
    near_high = dist_high_pct <= 0.8 if pd.notna(dist_high_pct) else False
    near_low = dist_low_pct <= 0.8 if pd.notna(dist_low_pct) else False
    spread_proxy = float(latest["SpreadProxyPct"]) if pd.notna(latest["SpreadProxyPct"]) else np.nan
    levels = support_resistance(df)
    mtf = mtf_confirmation(symbol, include_prepost=include_prepost)
    intel = get_symbol_intel(symbol, df=df, price_hint=price)
    blocked_status = None
    if mode_config["hard_block_choppy"] and regime["overall"] == "CHOPPY":
        blocked_status = "regime_blocked"
    elif mode_config["hard_block_high_event"] and intel["event_risk"] == "HIGH":
        blocked_status = "event_blocked"
    elif relvol < mode_config["min_setup_relvol"]:
        blocked_status = "low_volume"
    elif (
        mode_config.get("block_neutral_rsi", True)
        and pd.notna(rsi_now)
        and mode_config["rsi_neutral_low"] < rsi_now < mode_config["rsi_neutral_high"]
    ):
        blocked_status = "neutral_rsi"
    elif pd.notna(vwap_distance_pct) and vwap_distance_pct < mode_config["min_vwap_distance_pct"]:
        blocked_status = "vwap_indecision"

    long_score = 0.0
    short_score = 0.0
    long_score += 14 if above_vwap else 0
    short_score += 14 if not above_vwap else 0
    long_score += 14 if ema_bull else 0
    short_score += 14 if ema_bear else 0
    long_score += 10 if near_high else 0
    short_score += 10 if near_low else 0
    long_score += 12 if macd_up else 0
    short_score += 12 if macd_down else 0
    long_score += 10 * normalize_score(relvol, 0.8, 3.0)
    short_score += 10 * normalize_score(relvol, 0.8, 3.0)
    long_score += 8 * normalize_score(abs(change_pct), 0.2, 4.0)
    short_score += 8 * normalize_score(abs(change_pct), 0.2, 4.0)
    long_score += 8 * normalize_score(atr_pct, 0.3, 2.5)
    short_score += 8 * normalize_score(atr_pct, 0.3, 2.5)

    if pd.notna(rsi_now):
        if 52 <= rsi_now <= 74:
            long_score += 8
        elif rsi_now > 80:
            long_score -= 4
        if 26 <= rsi_now <= 48:
            short_score += 8
        elif rsi_now < 20:
            short_score -= 4

    if mtf["state"] == "BULLISH ALIGNED":
        long_score += 12
    elif mtf["state"] == "BEARISH ALIGNED":
        short_score += 12

    if regime["overall"] == "RISK-ON TREND UP":
        long_score += 8
        short_score -= 4
    elif regime["overall"] == "RISK-OFF TREND DOWN":
        short_score += 8
        long_score -= 4

    if pd.notna(spread_proxy) and spread_proxy > 1.8:
        long_score -= 8
        short_score -= 8

    if intel["news_sentiment"] == "BULLISH":
        long_score += 6
        short_score -= 2
    elif intel["news_sentiment"] == "BEARISH":
        short_score += 6
        long_score -= 2

    if intel["option_bias"] == "BULLISH":
        long_score += 6
    elif intel["option_bias"] == "BEARISH":
        short_score += 6

    if intel["event_risk"] == "HIGH":
        long_score -= 6
        short_score -= 6
    elif intel["event_risk"] == "MEDIUM":
        long_score -= 2
        short_score -= 2

    if intel["liquidity_label"] == "HIGH":
        long_score += 2
        short_score += 2
    elif intel["liquidity_label"] == "LOW":
        long_score -= 6
        short_score -= 6

    long_score = max(0.0, min(100.0, round(long_score, 2)))
    short_score = max(0.0, min(100.0, round(short_score, 2)))
    signal = "LONG" if long_score >= short_score else "SHORT"
    conviction = max(long_score, short_score)
    quality_score = _quality_score(
        signal,
        above_vwap=above_vwap,
        ema_bull=ema_bull,
        ema_bear=ema_bear,
        mtf_state=mtf["state"],
        relvol=relvol,
        vwap_distance_pct=vwap_distance_pct,
        spread_proxy=spread_proxy,
        liquidity_label=intel["liquidity_label"],
        event_risk=intel["event_risk"],
        mode=mode,
    )
    if blocked_status is None and quality_score < mode_config["min_quality_score"]:
        blocked_status = "low_quality"
    trigger = entry_trigger(signal, df, levels, mode=mode)
    plan = build_trade_plan(signal, price, atr_value, levels)

    reasons = []
    if signal == "LONG":
        if above_vwap:
            reasons.append("price above VWAP")
        if ema_bull:
            reasons.append("EMA stack bullish")
        if near_high:
            reasons.append("holding near session high")
        if macd_up:
            reasons.append("MACD improving")
        if mtf["state"] == "BULLISH ALIGNED":
            reasons.append("multi-timeframe bullish")
    else:
        if not above_vwap:
            reasons.append("price below VWAP")
        if ema_bear:
            reasons.append("EMA stack bearish")
        if near_low:
            reasons.append("holding near session low")
        if macd_down:
            reasons.append("MACD weakening")
        if mtf["state"] == "BEARISH ALIGNED":
            reasons.append("multi-timeframe bearish")

    if relvol >= 1.2:
        reasons.append(f"rel vol {relvol:.2f}x")
    if pd.notna(rsi_now):
        reasons.append(f"RSI {rsi_now:.1f}")
    if pd.notna(vwap_distance_pct):
        reasons.append(f"VWAP gap {vwap_distance_pct:.2f}%")
    if pd.notna(spread_proxy) and spread_proxy <= 1.8:
        reasons.append(f"tradability ok ({spread_proxy:.2f}% proxy)")
    elif pd.notna(spread_proxy):
        reasons.append(f"spread risk high ({spread_proxy:.2f}% proxy)")
    if intel["news_sentiment"] in {"BULLISH", "BEARISH"}:
        reasons.append(f"news tone {intel['news_sentiment'].lower()}")
    if intel["option_bias"] in {"BULLISH", "BEARISH"}:
        reasons.append(f"options flow {intel['option_bias'].lower()}")
    if intel["event_risk"] in {"HIGH", "MEDIUM"}:
        reasons.append(intel["event_summary"])
    if blocked_status:
        reasons.append(f"blocked: {blocked_status.replace('_', ' ')}")

    return {
        "symbol": symbol,
        "mode": mode,
        "scanner_status": "ok" if blocked_status is None else blocked_status,
        "scanner_qualified": blocked_status is None,
        "status": "ok" if blocked_status is None else blocked_status,
        "qualified": blocked_status is None,
        "price": round(price, 4),
        "change_pct": round(change_pct, 2) if pd.notna(change_pct) else np.nan,
        "volume": int(latest["Volume"]) if pd.notna(latest["Volume"]) else 0,
        "relvol": round(relvol, 2),
        "rsi": round(rsi_now, 2) if pd.notna(rsi_now) else np.nan,
        "atr": round(atr_value, 4) if pd.notna(atr_value) else np.nan,
        "atr_pct": round(atr_pct, 2) if pd.notna(atr_pct) else np.nan,
        "spread_proxy_pct": round(spread_proxy, 2) if pd.notna(spread_proxy) else np.nan,
        "vwap_distance_pct": round(vwap_distance_pct, 3) if pd.notna(vwap_distance_pct) else np.nan,
        "vwap": round(float(latest["VWAP"]), 4) if pd.notna(latest["VWAP"]) else np.nan,
        "session_high": round(session_high, 4),
        "session_low": round(session_low, 4),
        "support": levels["support"],
        "resistance": levels["resistance"],
        "prev_high": levels["prev_high"],
        "prev_low": levels["prev_low"],
        "dist_high_pct": round(dist_high_pct, 2) if pd.notna(dist_high_pct) else np.nan,
        "dist_low_pct": round(dist_low_pct, 2) if pd.notna(dist_low_pct) else np.nan,
        "long_score": long_score,
        "short_score": short_score,
        "quality_score": quality_score,
        "signal": signal,
        "conviction": conviction,
        "mtf_state": mtf["state"],
        "regime": regime["overall"],
        "trigger_ready": trigger["ready"],
        "trigger_text": trigger["trigger_text"],
        "entry": plan["entry"],
        "stop": plan["stop"],
        "target1": plan["target1"],
        "target2": plan["target2"],
        "rr1": plan["rr1"],
        "rr2": plan["rr2"],
        "reasons": "; ".join(reasons[:6]),
        "catalyst": intel["headline"] if intel["headline"] else intel["event_summary"],
        "news_sentiment": intel["news_sentiment"],
        "news_score": intel["news_score"],
        "news_count": intel["news_count"],
        "headline": intel["headline"],
        "headline_brief": intel["headline_brief"],
        "option_expiry": intel["option_expiry"],
        "option_bias": intel["option_bias"],
        "put_call_ratio": intel["put_call_ratio"],
        "call_volume": intel["call_volume"],
        "put_volume": intel["put_volume"],
        "atm_iv": intel["atm_iv"],
        "event_name": intel["event_name"],
        "event_date": intel["event_date"],
        "event_days": intel["event_days"],
        "event_risk": intel["event_risk"],
        "event_summary": intel["event_summary"],
        "spread_estimate_bps": intel["spread_estimate_bps"],
        "spread_source": intel["spread_source"],
        "liquidity_score": intel["liquidity_score"],
        "liquidity_label": intel["liquidity_label"],
        "dollar_volume_median": intel["dollar_volume_median"],
        "chart": df.tail(160).copy(),
    }


def scan_symbols(
    symbols: List[str],
    period: str,
    interval: str,
    include_prepost: bool,
    pause_s: float,
    regime: Dict,
    mode: str = DEFAULT_SCAN_MODE,
) -> pd.DataFrame:
    rows = []
    progress = st.progress(0, text="Scanning symbols...")
    total = max(len(symbols), 1)
    for idx, symbol in enumerate(symbols, start=1):
        result = analyze_symbol(symbol, period, interval, include_prepost, regime, mode=mode)
        if result.get("symbol"):
            rows.append(result)
        progress.progress(idx / total, text=f"Scanning {idx}/{total}: {symbol}")
        if pause_s > 0:
            time.sleep(pause_s)
    progress.empty()
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if "qualified" not in frame.columns:
        frame["qualified"] = frame["status"].eq("ok")
    return frame.sort_values(
        ["qualified", "trigger_ready", "quality_score", "conviction", "relvol", "atr_pct"],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)


def backtest_symbol(symbol: str, include_prepost: bool) -> Dict:
    df = download_history(symbol, period="5d", interval="15m", include_prepost=include_prepost)
    if df.empty or len(df) < 80:
        return {"symbol": symbol, "trades": 0, "win_rate": np.nan, "avg_r": np.nan}

    df = add_indicators(df)
    levels_window = 20
    trades = []

    for i in range(max(levels_window, 25), len(df) - 6):
        cur = df.iloc[i]
        window = df.iloc[i - levels_window:i]
        resistance = float(window["High"].max())
        support = float(window["Low"].min())
        price = float(cur["Close"])
        atr_value = float(cur["ATR14"]) if pd.notna(cur["ATR14"]) else 0.0
        if atr_value <= 0:
            continue

        long_ready = bool(price > cur["VWAP"] and price >= resistance and (cur["RelVol"] if pd.notna(cur["RelVol"]) else 0) >= 1.0)
        short_ready = bool(price < cur["VWAP"] and price <= support and (cur["RelVol"] if pd.notna(cur["RelVol"]) else 0) >= 1.0)
        if not long_ready and not short_ready:
            continue

        signal = "LONG" if long_ready else "SHORT"
        entry = price
        future = df.iloc[i + 1:i + 7]
        if signal == "LONG":
            stop = entry - atr_value
            target = entry + 2 * atr_value
            stopped = (future["Low"] <= stop).any()
            targeted = (future["High"] >= target).any()
        else:
            stop = entry + atr_value
            target = entry - 2 * atr_value
            stopped = (future["High"] >= stop).any()
            targeted = (future["Low"] <= target).any()

        if targeted and not stopped:
            trades.append(2.0)
        elif stopped and not targeted:
            trades.append(-1.0)
        else:
            exit_price = float(future["Close"].iloc[-1]) if not future.empty else entry
            r = (exit_price - entry) / atr_value if signal == "LONG" else (entry - exit_price) / atr_value
            trades.append(round(r, 2))

    if not trades:
        return {"symbol": symbol, "trades": 0, "win_rate": np.nan, "avg_r": np.nan}

    wins = sum(1 for t in trades if t > 0)
    return {
        "symbol": symbol,
        "trades": len(trades),
        "win_rate": round(wins / len(trades) * 100, 1),
        "avg_r": round(float(np.mean(trades)), 2),
    }


@st.cache_data(ttl=300, show_spinner=False)
def get_readiness_timeline(
    symbol: str,
    interval: str,
    include_prepost: bool,
    lookback_hours: int = READINESS_LOOKBACK_HOURS,
    mode: str = DEFAULT_HISTORY_MODE,
) -> pd.DataFrame:
    history_period = _history_period_for_interval(interval)
    raw = download_history(symbol, period=history_period, interval=interval, include_prepost=include_prepost)
    if raw.empty or len(raw) < 32:
        return pd.DataFrame()

    df = add_indicators(raw.dropna(subset=["Open", "High", "Low", "Close"])).copy()
    if df.empty or len(df) < 32:
        return pd.DataFrame()

    cutoff = pd.Timestamp(df.index[-1]) - pd.Timedelta(hours=lookback_hours)
    rows = []
    for i in range(30, len(df)):
        window = df.iloc[: i + 1]
        levels = support_resistance(window)
        long_trigger = entry_trigger("LONG", window, levels, mode=mode)
        short_trigger = entry_trigger("SHORT", window, levels, mode=mode)
        rows.append(
            {
                "timestamp": pd.Timestamp(window.index[-1]),
                "long_ready": bool(long_trigger["ready"]),
                "short_ready": bool(short_trigger["ready"]),
            }
        )

    timeline = pd.DataFrame(rows)
    if timeline.empty:
        return timeline

    recent_positions = timeline.index[timeline["timestamp"] >= cutoff].tolist()
    if recent_positions:
        start_pos = max(0, recent_positions[0] - 1)
        timeline = timeline.iloc[start_pos:].reset_index(drop=True)
    return timeline


def summarize_recent_readiness(timeline: pd.DataFrame, side: str) -> Dict:
    if timeline.empty:
        return {
            "current_ready": False,
            "current_duration": None,
            "last_started": None,
            "last_ended": None,
            "last_duration": None,
            "events": [],
        }

    column = "long_ready" if side == "LONG" else "short_ready"
    active_start = None
    previous_ready = False
    events = []

    for _, row in timeline.iterrows():
        ts = pd.Timestamp(row["timestamp"])
        ready = bool(row[column])
        if ready and not previous_ready:
            active_start = ts
        elif not ready and previous_ready and active_start is not None:
            duration = ts - active_start
            events.append(
                {
                    "side": side,
                    "started": active_start,
                    "ended": ts,
                    "duration": duration,
                    "status": "completed",
                }
            )
            active_start = None
        previous_ready = ready

    last_ts = pd.Timestamp(timeline["timestamp"].iloc[-1])
    current_duration = None
    if previous_ready and active_start is not None:
        current_duration = last_ts - active_start
        events.append(
            {
                "side": side,
                "started": active_start,
                "ended": None,
                "duration": current_duration,
                "status": "active",
            }
        )

    completed_events = [event for event in events if event["status"] == "completed"]
    last_completed = completed_events[-1] if completed_events else None

    return {
        "current_ready": previous_ready,
        "current_duration": current_duration,
        "last_started": last_completed["started"] if last_completed else (active_start if previous_ready else None),
        "last_ended": last_completed["ended"] if last_completed else None,
        "last_duration": last_completed["duration"] if last_completed else current_duration,
        "events": events[-5:],
    }


@st.cache_data(ttl=300, show_spinner=False)
def get_recent_ready_assets(
    symbols: List[str],
    interval: str,
    include_prepost: bool,
    lookback_hours: int = READINESS_LOOKBACK_HOURS,
    mode: str = DEFAULT_HISTORY_MODE,
) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        timeline = get_readiness_timeline(
            symbol,
            interval=interval,
            include_prepost=include_prepost,
            lookback_hours=lookback_hours,
            mode=mode,
        )
        if timeline.empty:
            continue

        long_summary = summarize_recent_readiness(timeline, "LONG")
        short_summary = summarize_recent_readiness(timeline, "SHORT")
        long_starts = int((timeline["long_ready"] & ~timeline["long_ready"].shift(fill_value=False)).sum())
        short_starts = int((timeline["short_ready"] & ~timeline["short_ready"].shift(fill_value=False)).sum())

        candidates = []
        if long_summary["last_started"] is not None:
            candidates.append(("LONG", pd.Timestamp(long_summary["last_started"])))
        if short_summary["last_started"] is not None:
            candidates.append(("SHORT", pd.Timestamp(short_summary["last_started"])))

        if not candidates and long_starts == 0 and short_starts == 0:
            continue

        last_side, last_ready_at = max(candidates, key=lambda item: item[1]) if candidates else ("N/A", pd.NaT)
        current_ready_side = "LONG" if long_summary["current_ready"] else ("SHORT" if short_summary["current_ready"] else "NO")

        rows.append(
            {
                "symbol": symbol,
                "currently_ready": current_ready_side != "NO",
                "current_ready_side": current_ready_side,
                "last_ready_side": last_side,
                "last_ready_at": last_ready_at,
                "long_ready_events": long_starts,
                "short_ready_events": short_starts,
                "total_ready_events": long_starts + short_starts,
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["currently_ready", "last_ready_at", "total_ready_events"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _evaluate_readiness_trade(signal: str, future_df: pd.DataFrame, entry: float, stop: float, target1: float) -> Dict:
    if future_df.empty:
        return {
            "exit_price": entry,
            "exit_timestamp": None,
            "actual_pnl": 0.0,
            "actual_pnl_pct": 0.0,
            "actual_r": 0.0,
            "outcome_status": "no_follow_through",
        }

    risk = max(abs(entry - stop), 1e-6)
    exit_price = float(future_df["Close"].iloc[-1])
    exit_timestamp = pd.Timestamp(future_df.index[-1])
    outcome_status = "data_end_exit"

    for idx, row in future_df.iterrows():
        high = float(row["High"])
        low = float(row["Low"])
        bar_ts = pd.Timestamp(idx)

        if signal == "LONG":
            stop_hit = low <= stop
            target_hit = high >= target1
            if stop_hit and target_hit:
                exit_price = stop
                exit_timestamp = bar_ts
                outcome_status = "stop_hit_same_bar"
                break
            if stop_hit:
                exit_price = stop
                exit_timestamp = bar_ts
                outcome_status = "stop_hit"
                break
            if target_hit:
                exit_price = target1
                exit_timestamp = bar_ts
                outcome_status = "target1_hit"
                break
        else:
            stop_hit = high >= stop
            target_hit = low <= target1
            if stop_hit and target_hit:
                exit_price = stop
                exit_timestamp = bar_ts
                outcome_status = "stop_hit_same_bar"
                break
            if stop_hit:
                exit_price = stop
                exit_timestamp = bar_ts
                outcome_status = "stop_hit"
                break
            if target_hit:
                exit_price = target1
                exit_timestamp = bar_ts
                outcome_status = "target1_hit"
                break

    actual_pnl = exit_price - entry if signal == "LONG" else entry - exit_price
    return {
        "exit_price": round(exit_price, 4),
        "exit_timestamp": exit_timestamp,
        "actual_pnl": round(actual_pnl, 4),
        "actual_pnl_pct": round(actual_pnl / entry * 100, 2) if entry else np.nan,
        "actual_r": round(actual_pnl / risk, 2) if risk else np.nan,
        "outcome_status": outcome_status,
    }


@st.cache_data(ttl=300, show_spinner=False)
def get_readiness_trade_audit(
    symbol: str,
    interval: str,
    include_prepost: bool,
    lookback_hours: int = READINESS_LOOKBACK_HOURS,
    mode: str = DEFAULT_HISTORY_MODE,
) -> pd.DataFrame:
    history_period = _history_period_for_interval(interval)
    raw = download_history(symbol, period=history_period, interval=interval, include_prepost=include_prepost)
    if raw.empty or len(raw) < 32:
        return pd.DataFrame()

    df = add_indicators(raw.dropna(subset=["Open", "High", "Low", "Close"])).copy()
    if df.empty or len(df) < 32:
        return pd.DataFrame()

    last_ts = pd.Timestamp(df.index[-1])
    cutoff = last_ts - pd.Timedelta(hours=lookback_hours)

    rows = []
    for i in range(30, len(df)):
        window = df.iloc[: i + 1]
        levels = support_resistance(window)
        last = window.iloc[-1]
        price = float(last["Close"])
        atr_value = float(last["ATR14"]) if pd.notna(last["ATR14"]) else np.nan
        long_trigger = entry_trigger("LONG", window, levels, mode=mode)
        short_trigger = entry_trigger("SHORT", window, levels, mode=mode)
        long_plan = build_trade_plan("LONG", price, atr_value, levels)
        short_plan = build_trade_plan("SHORT", price, atr_value, levels)
        rows.append(
            {
                "timestamp": pd.Timestamp(window.index[-1]),
                "long_ready": bool(long_trigger["ready"]),
                "short_ready": bool(short_trigger["ready"]),
                "long_entry": long_plan["entry"],
                "long_stop": long_plan["stop"],
                "long_target1": long_plan["target1"],
                "long_rr1": long_plan["rr1"],
                "short_entry": short_plan["entry"],
                "short_stop": short_plan["stop"],
                "short_target1": short_plan["target1"],
                "short_rr1": short_plan["rr1"],
            }
        )

    timeline = pd.DataFrame(rows)
    if timeline.empty:
        return timeline

    audits = []
    for side in ["LONG", "SHORT"]:
        ready_col = "long_ready" if side == "LONG" else "short_ready"
        entry_col = "long_entry" if side == "LONG" else "short_entry"
        stop_col = "long_stop" if side == "LONG" else "short_stop"
        target_col = "long_target1" if side == "LONG" else "short_target1"
        rr_col = "long_rr1" if side == "LONG" else "short_rr1"

        active_start_idx = None
        previous_ready = False
        for idx, row in timeline.iterrows():
            ready = bool(row[ready_col])
            if ready and not previous_ready:
                active_start_idx = idx
            elif not ready and previous_ready and active_start_idx is not None:
                start_row = timeline.iloc[active_start_idx]
                end_row = timeline.iloc[idx]
                start_ts = pd.Timestamp(start_row["timestamp"])
                end_ts = pd.Timestamp(end_row["timestamp"])
                if end_ts >= cutoff:
                    future_df = df[df.index > start_ts]
                    entry = float(start_row[entry_col])
                    stop = float(start_row[stop_col])
                    target1 = float(start_row[target_col])
                    evaluation = _evaluate_readiness_trade(side, future_df, entry, stop, target1)
                    if evaluation["outcome_status"] != "no_follow_through":
                        audits.append(
                            {
                                "side": side,
                                "started": start_ts,
                                "ended": evaluation["exit_timestamp"],
                                "readiness_ended": end_ts,
                                "window_duration": end_ts - start_ts,
                                "entry": round(entry, 4),
                                "stop": round(stop, 4),
                                "target1": round(target1, 4),
                                "suggested_profit": round(abs(target1 - entry), 4),
                                "suggested_profit_pct": round(abs(target1 - entry) / entry * 100, 2) if entry else np.nan,
                                "suggested_loss": round(abs(entry - stop), 4),
                                "suggested_loss_pct": round(abs(entry - stop) / entry * 100, 2) if entry else np.nan,
                                "rr1": start_row[rr_col],
                                **evaluation,
                            }
                        )
                active_start_idx = None
            previous_ready = ready

        if previous_ready and active_start_idx is not None:
            start_row = timeline.iloc[active_start_idx]
            start_ts = pd.Timestamp(start_row["timestamp"])
            if last_ts >= cutoff:
                future_df = df[df.index > start_ts]
                entry = float(start_row[entry_col])
                stop = float(start_row[stop_col])
                target1 = float(start_row[target_col])
                evaluation = _evaluate_readiness_trade(side, future_df, entry, stop, target1)
                if evaluation["outcome_status"] != "no_follow_through":
                    audits.append(
                        {
                            "side": side,
                            "started": start_ts,
                            "ended": evaluation["exit_timestamp"],
                            "readiness_ended": last_ts,
                            "window_duration": last_ts - start_ts,
                            "entry": round(entry, 4),
                            "stop": round(stop, 4),
                            "target1": round(target1, 4),
                            "suggested_profit": round(abs(target1 - entry), 4),
                            "suggested_profit_pct": round(abs(target1 - entry) / entry * 100, 2) if entry else np.nan,
                            "suggested_loss": round(abs(entry - stop), 4),
                            "suggested_loss_pct": round(abs(entry - stop) / entry * 100, 2) if entry else np.nan,
                            "rr1": start_row[rr_col],
                            **evaluation,
                        }
                    )

    if not audits:
        return pd.DataFrame()

    return pd.DataFrame(audits).sort_values("started", ascending=False).reset_index(drop=True)

