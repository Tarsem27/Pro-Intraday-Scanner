# ==============================
# file: scanner.py
# ==============================
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from data_utils import add_indicators, download_history, normalize_score
from market_intel import get_symbol_intel


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


def build_trade_plan(signal: str, price: float, atr_value: float, levels: Dict) -> Dict:
    atr_value = 0 if pd.isna(atr_value) else float(atr_value)
    support = levels.get("support", np.nan)
    resistance = levels.get("resistance", np.nan)

    if signal == "LONG":
        entry = price
        stop = max(price - atr_value, support) if pd.notna(support) else price - atr_value
        risk = max(entry - stop, 1e-6)
        target1 = entry + 2 * risk
        target2 = resistance if pd.notna(resistance) and resistance > entry else entry + 3 * risk
    else:
        entry = price
        stop = min(price + atr_value, resistance) if pd.notna(resistance) else price + atr_value
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


def entry_trigger(signal: str, df: pd.DataFrame, levels: Dict) -> Dict:
    if df.empty or len(df) < 3:
        return {"ready": False, "trigger_text": "insufficient trigger data"}

    last = df.iloc[-1]
    prev = df.iloc[-2]
    relvol = float(last.get("RelVol", 0) or 0)
    resistance = levels.get("resistance", np.nan)
    support = levels.get("support", np.nan)

    if signal == "LONG":
        threshold = max(prev["High"], resistance * 0.998 if pd.notna(resistance) else prev["High"])
        ready = bool(last["Close"] > last["VWAP"] and last["Close"] >= threshold and relvol >= 1.0)
        text = f"LONG ready above {threshold:.4f}" if ready else "long not ready"
    else:
        threshold = min(prev["Low"], support * 1.002 if pd.notna(support) else prev["Low"])
        ready = bool(last["Close"] < last["VWAP"] and last["Close"] <= threshold and relvol >= 1.0)
        text = f"SHORT ready below {threshold:.4f}" if ready else "short not ready"

    return {"ready": ready, "trigger_text": text}


def analyze_symbol(symbol: str, period: str, interval: str, include_prepost: bool, regime: Dict) -> Dict:
    raw = download_history(symbol, period=period, interval=interval, include_prepost=include_prepost)
    if raw.empty or len(raw) < 50:
        return {"symbol": symbol, "status": "insufficient_data"}

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
    relvol = float(latest["RelVol"]) if pd.notna(latest["RelVol"]) else 0.0
    rsi_now = float(latest["RSI14"]) if pd.notna(latest["RSI14"]) else np.nan
    atr_value = float(latest["ATR14"]) if pd.notna(latest["ATR14"]) else np.nan
    atr_pct = (atr_value / price * 100) if price else np.nan
    above_vwap = bool(price > float(latest["VWAP"])) if pd.notna(latest["VWAP"]) else False
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
    elif regime["overall"] == "CHOPPY":
        long_score -= 5
        short_score -= 5

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
    trigger = entry_trigger(signal, df, levels)
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

    return {
        "symbol": symbol,
        "status": "ok",
        "price": round(price, 4),
        "change_pct": round(change_pct, 2) if pd.notna(change_pct) else np.nan,
        "volume": int(latest["Volume"]) if pd.notna(latest["Volume"]) else 0,
        "relvol": round(relvol, 2),
        "rsi": round(rsi_now, 2) if pd.notna(rsi_now) else np.nan,
        "atr": round(atr_value, 4) if pd.notna(atr_value) else np.nan,
        "atr_pct": round(atr_pct, 2) if pd.notna(atr_pct) else np.nan,
        "spread_proxy_pct": round(spread_proxy, 2) if pd.notna(spread_proxy) else np.nan,
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


def scan_symbols(symbols: List[str], period: str, interval: str, include_prepost: bool, pause_s: float, regime: Dict) -> pd.DataFrame:
    rows = []
    progress = st.progress(0, text="Scanning symbols...")
    total = max(len(symbols), 1)
    for idx, symbol in enumerate(symbols, start=1):
        result = analyze_symbol(symbol, period, interval, include_prepost, regime)
        if result.get("status") == "ok":
            rows.append(result)
        progress.progress(idx / total, text=f"Scanning {idx}/{total}: {symbol}")
        if pause_s > 0:
            time.sleep(pause_s)
    progress.empty()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["trigger_ready", "conviction", "relvol", "atr_pct"], ascending=[False, False, False, False]).reset_index(drop=True)


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
def get_readiness_timeline(symbol: str, interval: str, include_prepost: bool, lookback_hours: int = 48) -> pd.DataFrame:
    history_period = "5d" if interval in {"1m", "2m", "5m", "15m", "30m"} else "1mo"
    raw = download_history(symbol, period=history_period, interval=interval, include_prepost=include_prepost)
    if raw.empty or len(raw) < 35:
        return pd.DataFrame()

    df = add_indicators(raw.dropna(subset=["Open", "High", "Low", "Close"])).copy()
    if df.empty or len(df) < 35:
        return pd.DataFrame()

    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)

    cutoff = df.index[-1] - pd.Timedelta(hours=lookback_hours)
    rows = []
    for i in range(30, len(df)):
        window = df.iloc[: i + 1]
        levels = support_resistance(window)
        long_trigger = entry_trigger("LONG", window, levels)
        short_trigger = entry_trigger("SHORT", window, levels)
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

