from datetime import datetime
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


POSITIVE_NEWS_WORDS = {
    "beat", "beats", "surge", "surges", "jump", "jumps", "bullish", "upgrade",
    "upgrades", "buyback", "record", "growth", "strong", "wins", "win",
    "partnership", "expands", "expansion", "raises", "raised", "profit",
}
NEGATIVE_NEWS_WORDS = {
    "miss", "misses", "drop", "drops", "plunge", "plunges", "bearish", "downgrade",
    "downgrades", "lawsuit", "probe", "cut", "cuts", "warning", "weak",
    "loss", "decline", "declines", "falls", "fall", "delay", "delays",
}


def _clean_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        for item in value.to_numpy().flatten().tolist():
            ts = _clean_timestamp(item)
            if ts is not None:
                return ts
        return None
    if isinstance(value, pd.Series):
        for item in value.tolist():
            ts = _clean_timestamp(item)
            if ts is not None:
                return ts
        return None
    if isinstance(value, (list, tuple, set)):
        for item in value:
            ts = _clean_timestamp(item)
            if ts is not None:
                return ts
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    if isinstance(ts, pd.DatetimeIndex):
        return _clean_timestamp(ts[0] if len(ts) else None)
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_localize(None)
    return pd.Timestamp(ts)


def _safe_float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def _label_from_score(score: float, bullish_cutoff: float = 0.35, bearish_cutoff: float = -0.35) -> str:
    if pd.isna(score):
        return "UNKNOWN"
    if score >= bullish_cutoff:
        return "BULLISH"
    if score <= bearish_cutoff:
        return "BEARISH"
    return "NEUTRAL"


def _score_headline(title: str) -> float:
    words = {token.strip(".,:;!?()[]{}'\"").lower() for token in title.split()}
    positive_hits = len(words & POSITIVE_NEWS_WORDS)
    negative_hits = len(words & NEGATIVE_NEWS_WORDS)
    return float(positive_hits - negative_hits)


def _news_items_to_rows(news_items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in news_items or []:
        content = item.get("content", item) if isinstance(item, dict) else {}
        title = str(content.get("title") or item.get("title") or "").strip()
        publisher = str(content.get("provider", {}).get("displayName") or content.get("publisher") or item.get("publisher") or "Unknown")
        published_raw = (
            content.get("pubDate")
            or content.get("publishedAt")
            or item.get("providerPublishTime")
            or item.get("published")
        )
        published = _clean_timestamp(published_raw)
        link = (
            content.get("canonicalUrl", {}).get("url")
            or content.get("clickThroughUrl", {}).get("url")
            or item.get("link")
            or ""
        )
        if title:
            rows.append(
                {
                    "title": title,
                    "publisher": publisher,
                    "published": published,
                    "link": link,
                    "sentiment_score": _score_headline(title),
                }
            )
    return rows


@st.cache_data(ttl=300, show_spinner=False)
def fetch_news_snapshot(symbol: str) -> Dict[str, Any]:
    try:
        ticker = yf.Ticker(symbol)
        raw_items = getattr(ticker, "news", None) or []
        rows = _news_items_to_rows(raw_items)
        if not rows:
            return {
                "news_count": 0,
                "news_score": 0.0,
                "news_sentiment": "UNKNOWN",
                "headline": "No recent headline available",
                "headline_brief": "",
                "news_titles": [],
            }

        news_df = pd.DataFrame(rows).sort_values("published", ascending=False, na_position="last")
        top_rows = news_df.head(3).copy()
        avg_score = float(top_rows["sentiment_score"].mean()) if not top_rows.empty else 0.0
        top_headline = str(top_rows.iloc[0]["title"])
        top_titles = top_rows["title"].tolist()
        brief = " | ".join(top_titles)
        return {
            "news_count": int(len(news_df)),
            "news_score": round(avg_score, 2),
            "news_sentiment": _label_from_score(avg_score),
            "headline": top_headline,
            "headline_brief": brief,
            "news_titles": top_titles,
        }
    except Exception:
        return {
            "news_count": 0,
            "news_score": 0.0,
            "news_sentiment": "UNKNOWN",
            "headline": "News feed unavailable",
            "headline_brief": "",
            "news_titles": [],
        }


def _extract_event_candidates(calendar_obj: Any, info_obj: Dict[str, Any] | None) -> List[tuple[str, pd.Timestamp]]:
    events: List[tuple[str, pd.Timestamp]] = []
    calendar_map: Dict[str, Any] = {}

    if isinstance(calendar_obj, pd.DataFrame):
        for column in calendar_obj.columns:
            calendar_map[str(column)] = calendar_obj[column]
    elif isinstance(calendar_obj, dict):
        calendar_map = dict(calendar_obj)

    for key, raw_value in calendar_map.items():
        ts = _clean_timestamp(raw_value)
        if ts is not None:
            events.append((str(key), ts))

    info_obj = info_obj or {}
    extra_keys = {
        "earningsDate": "Earnings Date",
        "exDividendDate": "Ex-Dividend Date",
        "dividendDate": "Dividend Date",
    }
    for raw_key, label in extra_keys.items():
        ts = _clean_timestamp(info_obj.get(raw_key))
        if ts is not None:
            events.append((label, ts))

    return events


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_event_snapshot(symbol: str) -> Dict[str, Any]:
    try:
        ticker = yf.Ticker(symbol)
        calendar_obj = getattr(ticker, "calendar", None)
        info_obj = {}
        try:
            info_obj = ticker.info or {}
        except Exception:
            info_obj = {}

        events = _extract_event_candidates(calendar_obj, info_obj)
        today = pd.Timestamp(datetime.utcnow().date())
        future_events = sorted((name, ts) for name, ts in events if ts >= today - pd.Timedelta(days=1))

        if not future_events:
            return {
                "event_name": "No scheduled event found",
                "event_date": None,
                "event_days": np.nan,
                "event_risk": "LOW",
                "event_summary": "No near-dated earnings/dividend event found",
            }

        event_name, event_ts = future_events[0]
        days = int((event_ts.normalize() - today).days)
        if days <= 1:
            risk = "HIGH"
        elif days <= 5:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        if days < 0:
            summary = f"{event_name} just passed"
        elif days == 0:
            summary = f"{event_name} is today"
        elif days == 1:
            summary = f"{event_name} is tomorrow"
        else:
            summary = f"{event_name} in {days} days"

        return {
            "event_name": event_name,
            "event_date": event_ts.strftime("%Y-%m-%d"),
            "event_days": days,
            "event_risk": risk,
            "event_summary": summary,
        }
    except Exception:
        return {
            "event_name": "Calendar unavailable",
            "event_date": None,
            "event_days": np.nan,
            "event_risk": "UNKNOWN",
            "event_summary": "Event calendar unavailable",
        }


def _nearest_strike_iv(chain_df: pd.DataFrame, price: float) -> float:
    if chain_df.empty or "strike" not in chain_df.columns or "impliedVolatility" not in chain_df.columns or pd.isna(price):
        return np.nan
    nearest_idx = (chain_df["strike"] - price).abs().idxmin()
    if pd.isna(nearest_idx):
        return np.nan
    return _safe_float(chain_df.loc[nearest_idx, "impliedVolatility"])


@st.cache_data(ttl=600, show_spinner=False)
def fetch_option_snapshot(symbol: str, price_hint: float) -> Dict[str, Any]:
    try:
        ticker = yf.Ticker(symbol)
        expiries = list(getattr(ticker, "options", []) or [])
        if not expiries:
            return {
                "option_expiry": "N/A",
                "option_bias": "NO OPTIONS",
                "put_call_ratio": np.nan,
                "call_volume": 0,
                "put_volume": 0,
                "atm_iv": np.nan,
                "option_summary": "No option chain available",
            }

        expiry = expiries[0]
        chain = ticker.option_chain(expiry)
        calls = getattr(chain, "calls", pd.DataFrame()).copy()
        puts = getattr(chain, "puts", pd.DataFrame()).copy()

        call_volume = int(calls.get("volume", pd.Series(dtype=float)).fillna(0).sum()) if not calls.empty else 0
        put_volume = int(puts.get("volume", pd.Series(dtype=float)).fillna(0).sum()) if not puts.empty else 0
        call_oi = float(calls.get("openInterest", pd.Series(dtype=float)).fillna(0).sum()) if not calls.empty else 0.0
        put_oi = float(puts.get("openInterest", pd.Series(dtype=float)).fillna(0).sum()) if not puts.empty else 0.0
        put_call_ratio = put_oi / call_oi if call_oi > 0 else np.nan
        call_iv = _nearest_strike_iv(calls, price_hint)
        put_iv = _nearest_strike_iv(puts, price_hint)
        atm_iv = np.nanmean([call_iv, put_iv]) if pd.notna(call_iv) or pd.notna(put_iv) else np.nan

        if pd.isna(put_call_ratio):
            bias = "MIXED"
        elif put_call_ratio <= 0.75:
            bias = "BULLISH"
        elif put_call_ratio >= 1.30:
            bias = "BEARISH"
        else:
            bias = "MIXED"

        summary = f"{expiry} PCR {put_call_ratio:.2f}" if pd.notna(put_call_ratio) else f"{expiry} PCR unavailable"
        return {
            "option_expiry": expiry,
            "option_bias": bias,
            "put_call_ratio": round(put_call_ratio, 2) if pd.notna(put_call_ratio) else np.nan,
            "call_volume": call_volume,
            "put_volume": put_volume,
            "atm_iv": round(float(atm_iv), 3) if pd.notna(atm_iv) else np.nan,
            "option_summary": summary,
        }
    except Exception:
        return {
            "option_expiry": "N/A",
            "option_bias": "UNAVAILABLE",
            "put_call_ratio": np.nan,
            "call_volume": 0,
            "put_volume": 0,
            "atm_iv": np.nan,
            "option_summary": "Option chain unavailable",
        }


def build_liquidity_snapshot(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "spread_estimate_bps": np.nan,
            "spread_source": "unavailable",
            "liquidity_score": 0.0,
            "liquidity_label": "UNKNOWN",
            "dollar_volume_median": np.nan,
        }

    recent = df.tail(20).copy()
    recent["DollarVolume"] = recent["Close"].fillna(0) * recent["Volume"].fillna(0)
    spread_estimate_pct = float(recent["SpreadProxyPct"].iloc[-1]) if "SpreadProxyPct" in recent.columns and pd.notna(recent["SpreadProxyPct"].iloc[-1]) else np.nan
    spread_estimate_bps = spread_estimate_pct * 100 if pd.notna(spread_estimate_pct) else np.nan
    median_dollar_volume = float(recent["DollarVolume"].median()) if not recent.empty else np.nan
    bar_range_pct = ((recent["High"] - recent["Low"]) / recent["Close"].replace(0, np.nan) * 100).replace([np.inf, -np.inf], np.nan)
    median_bar_range = float(bar_range_pct.median()) if not bar_range_pct.empty else np.nan

    liquidity_score = 50.0
    if pd.notna(median_dollar_volume):
        if median_dollar_volume >= 50_000_000:
            liquidity_score += 25
        elif median_dollar_volume >= 10_000_000:
            liquidity_score += 15
        elif median_dollar_volume < 2_000_000:
            liquidity_score -= 15

    if pd.notna(spread_estimate_bps):
        if spread_estimate_bps <= 25:
            liquidity_score += 15
        elif spread_estimate_bps <= 50:
            liquidity_score += 5
        elif spread_estimate_bps > 120:
            liquidity_score -= 20

    if pd.notna(median_bar_range):
        if median_bar_range > 2.5:
            liquidity_score -= 10
        elif median_bar_range < 0.6:
            liquidity_score += 5

    liquidity_score = max(0.0, min(100.0, round(liquidity_score, 1)))
    if liquidity_score >= 75:
        liquidity_label = "HIGH"
    elif liquidity_score >= 50:
        liquidity_label = "MEDIUM"
    else:
        liquidity_label = "LOW"

    return {
        "spread_estimate_bps": round(spread_estimate_bps, 1) if pd.notna(spread_estimate_bps) else np.nan,
        "spread_source": "Bar-range proxy",
        "liquidity_score": liquidity_score,
        "liquidity_label": liquidity_label,
        "dollar_volume_median": round(median_dollar_volume, 0) if pd.notna(median_dollar_volume) else np.nan,
    }


def get_symbol_intel(symbol: str, df: pd.DataFrame, price_hint: float) -> Dict[str, Any]:
    news = fetch_news_snapshot(symbol)
    event = fetch_event_snapshot(symbol)
    options = fetch_option_snapshot(symbol, price_hint=price_hint)
    liquidity = build_liquidity_snapshot(df)
    return {**news, **event, **options, **liquidity}
