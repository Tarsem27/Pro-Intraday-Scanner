# ==============================
# file: app.py
# ==============================
from datetime import datetime
import time
from typing import List
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

from config import DEFAULT_UNIVERSES, DEFAULT_CUSTOM_SYMBOLS
from data_utils import detect_market_regime, parse_symbol_input
from execution import build_order_ticket, get_broker_adapter, ticket_to_frame
from notifications import (
    get_telegram_bot_token,
    get_telegram_default_chat_id,
    send_telegram_message,
    send_ready_signal_notifications,
    telegram_alerts_enabled,
)
from scanner import (
    DEFAULT_HISTORY_MODE,
    DEFAULT_SCAN_MODE,
    READINESS_LOOKBACK_HOURS,
    SCANNER_MODES,
    backtest_symbol,
    get_recent_ready_assets,
    get_readiness_timeline,
    get_readiness_trade_audit,
    scan_symbols,
    summarize_recent_readiness,
)
from ui_components import render_market_intel, render_metric_card, render_setup_detail

st.set_page_config(page_title="Pro Intraday Scanner", layout="wide")

MELBOURNE_TZ = ZoneInfo("Australia/Melbourne")

RESULT_DEFAULTS = {
    "scanner_status": "unknown",
    "scanner_qualified": False,
    "status": "unknown",
    "qualified": True,
    "accepted": True,
    "acceptance_status": "ok",
    "signal": "N/A",
    "conviction": pd.NA,
    "price": pd.NA,
    "relvol": pd.NA,
    "rsi": pd.NA,
    "spread_proxy_pct": pd.NA,
    "vwap_distance_pct": pd.NA,
    "trigger_ready": False,
    "trigger_text": "No trigger data",
    "reasons": "No detailed data",
    "news_sentiment": "UNKNOWN",
    "option_bias": "UNAVAILABLE",
    "event_risk": "UNKNOWN",
    "liquidity_label": "UNKNOWN",
    "news_count": 0,
    "headline": "No headline available",
    "event_summary": "No event data",
    "spread_estimate_bps": pd.NA,
    "spread_source": "unavailable",
    "liquidity_score": pd.NA,
    "option_expiry": "N/A",
    "put_call_ratio": pd.NA,
    "call_volume": 0,
    "put_volume": 0,
    "atm_iv": pd.NA,
    "quality_score": 0,
}

GLOSSARY_ITEMS = [
    ("Signal", "The scanner's directional idea. `LONG` means the setup favors buying for a move up. `SHORT` means it favors selling for a move down."),
    ("Conviction", "A 0-100 score built from trend, momentum, volume, regime, news, options, and liquidity checks. Higher does not mean guaranteed; it only means more conditions aligned."),
    ("Trigger-ready", "Whether price has already reached the scanner's minimum confirmation level. `READY` means the setup is active now. `WAIT` means the idea exists but confirmation has not happened yet."),
    ("VWAP", "Volume Weighted Average Price. It shows the average traded price weighted by volume. Many intraday traders treat price above VWAP as stronger intraday behavior and below VWAP as weaker behavior."),
    ("EMA9 / EMA20", "Fast exponential moving averages. They react quickly to recent price changes. When EMA9 is above EMA20, short-term momentum is often stronger."),
    ("SMA50", "A slower simple moving average. It helps show the broader intraday trend context compared with the faster EMAs."),
    ("RSI", "Relative Strength Index. A momentum oscillator from 0 to 100. Very high RSI can mean strong momentum or overextension. Very low RSI can mean weakness or oversold conditions."),
    ("ATR %", "Average True Range as a percentage of price. It estimates how much the asset typically moves. Higher ATR usually means wider swings and potentially wider stops."),
    ("RelVol", "Relative volume. A reading of `2.0` means recent volume is about 2x the rolling average. Bigger relative volume can make a breakout more trustworthy."),
    ("Spread Proxy %", "A rough tradability estimate based on recent bar range, not a real broker bid/ask spread. Lower is generally easier to trade. It is a proxy, not an execution quote."),
    ("Support / Resistance", "Areas where price recently found buyers or sellers. Support is a floor-like area; resistance is a ceiling-like area."),
    ("Prev High / Prev Low", "Reference points from prior bars. Traders watch them because breakouts and breakdowns often happen around these levels."),
    ("MTF State", "Multi-timeframe state. The scanner checks more than one timeframe and labels them aligned bullish, aligned bearish, or mixed."),
    ("Entry", "The price area the plan assumes for getting in. It is not a promise that the market will fill there."),
    ("Stop", "The exit level used if the trade is wrong. It exists to define risk before entering."),
    ("Target 1 / Target 2", "Planned profit areas. These are scenarios, not guarantees."),
    ("R:R", "Risk-to-reward. If a trade risks $1 to try to make $2, the R:R is 2.0."),
    ("News Sentiment", "A simple headline-tone read from recent public headlines. It is only a lightweight context signal, not deep news analysis."),
    ("Options Bias", "A rough read of options positioning from put/call open interest or volume. It can hint at bullish or bearish positioning, but it is not a standalone trade signal."),
    ("Event Risk", "A warning for scheduled catalysts like earnings or dividend dates. High event risk means the asset may gap or move sharply around the event."),
    ("Liquidity Label", "A rough quality label for how tradable the instrument looks based on recent dollar volume and spread proxy. Higher liquidity is generally easier for beginners."),
    ("Options Opportunity Board", "A ranked view of underlyings whose option-chain context looks interesting. This is not the same as scanning individual option contracts."),
]


def render_beginner_guide():
    with st.expander("Beginner Guide: terms, example, assumptions"):
        tab1, tab2, tab3, tab4 = st.tabs(["Quick start", "Glossary", "Worked example", "Assumptions"])

        with tab1:
            st.markdown(
                """
                **What this app does**

                This scanner does not predict the future. It ranks symbols based on a checklist of conditions that intraday traders often watch: trend, momentum, volume, structure, market regime, catalyst risk, and tradability.

                **A simple way to use it**

                1. Start with `Market regime`. If the market is choppy, be more selective.
                2. Look for higher `Conviction` and decide whether you only want `READY` setups.
                3. Open one symbol in `Setup inspection`.
                4. Read the `Reasoning`, `Trade plan`, `News / Catalyst`, `Options Snapshot`, and `Event / Liquidity` sections together.
                5. Before entering anything, ask: where is my entry, where is my stop, what invalidates the idea, and is the instrument liquid enough for me?

                **What a beginner should focus on first**

                Watch `Signal`, `Trigger-ready`, `VWAP`, `RelVol`, `Stop`, `Target 1`, and `Event risk` before worrying about every advanced field.

                **What the new universes mean**

                `Penny / Low Price Stocks` and `Small / Mid Caps` can move harder than mega caps, but they can also be less liquid and less forgiving. Beginners should size smaller and pay extra attention to spread and event risk.
                """
            )

        with tab2:
            for term, explanation in GLOSSARY_ITEMS:
                st.markdown(f"**{term}**")
                st.write(explanation)

        with tab3:
            st.markdown(
                """
                **Worked example: AAPL intraday long**

                Imagine AAPL is trading at `210.40`.

                Example readings:
                - `Signal = LONG`
                - `Conviction = 74`
                - `Trigger-ready = READY`
                - `VWAP = 209.95`
                - `EMA9 > EMA20 > SMA50`
                - `RelVol = 1.9`
                - `RSI = 63`
                - `Entry = 210.40`
                - `Stop = 209.70`
                - `Target 1 = 211.80`
                - `Target 2 = 212.50`
                - `Event risk = LOW`
                - `Liquidity label = HIGH`

                **How a beginner should read that**

                Price is above VWAP, short-term moving averages are stacked bullish, and volume is stronger than usual. That means buyers are in control for now. The setup is `READY`, so the scanner believes price has already met its basic confirmation.

                The trade plan says you would be wrong if price falls to around `209.70`, so that becomes the invalidation point. If you do not know where you are wrong, you should not take the trade. If price reaches `211.80`, that is the first planned profit zone.

                **What this does not mean**

                It does not mean AAPL must go up. It only means several conditions are aligned at the same time. A beginner should still check the chart, verify the live spread, keep size small, and use the stop level as a hard risk boundary.

                **About options**

                This app currently scans the underlying asset first, then uses the option chain as context. It does not yet tell you which exact call or put contract to buy.
                """
            )

        with tab4:
            st.markdown(
                """
                **Scanner assumptions**

                - The scanner assumes recent price behavior can provide useful intraday context.
                - It assumes higher volume and aligned timeframes make setups more meaningful.
                - It assumes the planned `Entry`, `Stop`, and `Targets` are guides, not guaranteed fills.
                - It assumes public news, options, and calendar data are helpful context, but not perfect.
                - It assumes the `Spread Proxy %` is only an estimate unless a real broker spread feed is connected.

                **Important beginner cautions**

                - `Conviction` is not certainty.
                - `READY` does not mean safe.
                - `LONG` and `SHORT` are ideas, not instructions.
                - High `Event risk` can cause fast moves and slippage.
                - Low-liquidity instruments can move erratically and fill badly.
                - Penny stocks can behave far worse than large caps when spreads widen.
                - A good setup with bad risk sizing can still be a bad trade.

                **Good first-demo-account habits**

                - Trade only very liquid names at first.
                - Avoid trading right before earnings or major events.
                - Risk a tiny amount and practice following stops.
                - Review whether the setup failed because the idea was wrong or because the execution was poor.
                """
            )


def render_htf_ltf_workflow_note():
    with st.expander("HTF / LTF workflow (common SMC-style teaching)"):
        st.markdown(
            """
            Many educators describe a **3-step** rhythm: establish bias on a higher timeframe, confirm on a lower timeframe, then execute with defined risk.

            **1. Range (higher timeframe)**  
            Use a slower chart (for example **15m**) to see who is in control, where structure breaks, and where imbalances (*fair value gaps*) may attract price. This app’s **multi-timeframe state**, **VWAP**, and **support/resistance** are in that spirit, but it does **not** auto-detect BOS, CHoCH, or FVG.

            **2. Change (lower timeframe)**  
            Drop to a faster chart (for example **1m** or **5m**) for confirmation and to avoid chop. Here, **trigger-ready**, **relative volume**, and **quality score** play a similar confirmation role. **1m** history from the data provider is short—if sections below look empty, try **5m** or **15m**.

            **3. Execution**  
            Use the printed **entry**, **stop**, **target**, and **R:R**. Treat **high-impact news** as a reason to stand aside. Expectancy comes from repeating the process over many trades, not from any single alert.

            **Psychology**  
            Keep per-trade expectations modest; edge shows up over a **sample of trades**, not in every signal.
            """
        )


def render_best_profit_scan_table(scan_df: pd.DataFrame):
    st.subheader("Best profit setups (this scan)")
    st.caption(
        "Highest-ranked names from the latest run: trigger-ready first, then quality, conviction, and planned R:R (`rr1`). "
        "This reflects the scanner’s *current* edge estimate — not audited past P/L."
    )
    if scan_df.empty:
        st.caption("Run a scan to populate this table.")
        return
    ok = scan_df[scan_df["status"].eq("ok")].copy() if "status" in scan_df.columns else scan_df.copy()
    if ok.empty:
        st.info("No passing setups in the last scan. Try Exploratory mode or a different universe.")
        return
    ok["rr1_num"] = pd.to_numeric(ok["rr1"], errors="coerce")
    ok = ok.sort_values(
        ["trigger_ready", "quality_score", "conviction", "rr1_num"],
        ascending=[False, False, False, False],
    )
    show_cols = [
        "symbol",
        "signal",
        "trigger_ready",
        "quality_score",
        "conviction",
        "rr1",
        "entry",
        "stop",
        "target1",
        "change_pct",
    ]
    show_cols = [c for c in show_cols if c in ok.columns]
    st.dataframe(ok[show_cols].head(12), use_container_width=True, hide_index=True)


def _format_duration(delta: pd.Timedelta | None) -> str:
    if delta is None or pd.isna(delta):
        return "N/A"
    total_seconds = int(max(delta.total_seconds(), 0))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _format_melbourne_time(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ts = ts.tz_convert(MELBOURNE_TZ)
    return ts.strftime("%Y-%m-%d %H:%M:%S %Z")


def _readiness_lookback_label() -> str:
    h = READINESS_LOOKBACK_HOURS
    return f"{h} hours" if h != 1 else "1 hour"


def _history_interval_candidates(interval: str) -> List[str]:
    fallback_map = {
        "1m": ["1m", "5m", "15m", "30m"],
        "2m": ["2m", "5m", "15m", "30m"],
        "5m": ["5m", "15m", "30m"],
        "15m": ["15m", "30m", "5m"],
        "30m": ["30m", "15m", "5m"],
    }
    return fallback_map.get(interval, [interval, "15m", "30m"])


def _history_mode_candidates(history_mode: str) -> List[str]:
    ordered = [history_mode, "Balanced", "Exploratory", "Strict"]
    seen = set()
    return [mode for mode in ordered if not (mode in seen or seen.add(mode))]


def _resolve_signal_timing_data(symbol: str, interval: str, include_prepost: bool, history_mode: str):
    attempted = []
    for candidate_interval in _history_interval_candidates(interval):
        for candidate_mode in _history_mode_candidates(history_mode):
            timeline = get_readiness_timeline(
                symbol,
                interval=candidate_interval,
                include_prepost=include_prepost,
                lookback_hours=READINESS_LOOKBACK_HOURS,
                mode=candidate_mode,
            )
            attempted.append(f"{candidate_interval}/{candidate_mode}")
            if timeline.empty:
                continue
            audit_df = get_readiness_trade_audit(
                symbol,
                interval=candidate_interval,
                include_prepost=include_prepost,
                lookback_hours=READINESS_LOOKBACK_HOURS,
                mode=candidate_mode,
            )
            return {
                "timeline": timeline,
                "audit_df": audit_df,
                "used_interval": candidate_interval,
                "used_mode": candidate_mode,
                "fallback_used": candidate_interval != interval or candidate_mode != history_mode,
                "attempted": attempted,
            }

    return {
        "timeline": pd.DataFrame(),
        "audit_df": pd.DataFrame(),
        "used_interval": interval,
        "used_mode": history_mode,
        "fallback_used": False,
        "attempted": attempted,
    }


def _resolve_recent_ready_assets(symbols: List[str], interval: str, include_prepost: bool, history_mode: str):
    attempted = []
    for candidate_interval in _history_interval_candidates(interval):
        for candidate_mode in _history_mode_candidates(history_mode):
            ready_assets = get_recent_ready_assets(
                symbols,
                interval=candidate_interval,
                include_prepost=include_prepost,
                lookback_hours=READINESS_LOOKBACK_HOURS,
                mode=candidate_mode,
            )
            attempted.append(f"{candidate_interval}/{candidate_mode}")
            if ready_assets.empty:
                continue
            return {
                "ready_assets": ready_assets,
                "used_interval": candidate_interval,
                "used_mode": candidate_mode,
                "fallback_used": candidate_interval != interval or candidate_mode != history_mode,
                "attempted": attempted,
            }

    return {
        "ready_assets": pd.DataFrame(),
        "used_interval": interval,
        "used_mode": history_mode,
        "fallback_used": False,
        "attempted": attempted,
    }


@st.cache_data(ttl=300, show_spinner=False)
def get_portfolio_trade_audit(symbols: List[str], interval: str, include_prepost: bool, history_mode: str) -> pd.DataFrame:
    """Merge per-symbol readiness audits. Tries fallback bar sizes/modes when the primary history is empty (same idea as signal timing)."""
    audit_frames = []
    for symbol in symbols:
        symbol_audit = pd.DataFrame()
        used_interval = interval
        used_mode = history_mode
        for candidate_interval in _history_interval_candidates(interval):
            for candidate_mode in _history_mode_candidates(history_mode):
                symbol_audit = get_readiness_trade_audit(
                    symbol,
                    interval=candidate_interval,
                    include_prepost=include_prepost,
                    lookback_hours=READINESS_LOOKBACK_HOURS,
                    mode=candidate_mode,
                )
                if not symbol_audit.empty:
                    used_interval, used_mode = candidate_interval, candidate_mode
                    break
            if not symbol_audit.empty:
                break
        if symbol_audit.empty:
            continue
        enriched = symbol_audit.copy()
        enriched["symbol"] = symbol
        enriched["audit_interval"] = used_interval
        enriched["audit_mode"] = used_mode
        audit_frames.append(enriched)

    if not audit_frames:
        return pd.DataFrame()

    return pd.concat(audit_frames, ignore_index=True)


def render_profitable_traits_section(all_results: pd.DataFrame, interval: str, include_prepost: bool, history_mode: str):
    render_best_profit_scan_table(all_results)

    st.subheader("Profitable Traits")
    st.caption(
        "Tracks simulated outcomes for past `READY` windows (stop vs Target 1 vs end of history) across scanned symbols, "
        "using the same readiness audit as Signal timing. Empty results usually mean sparse data at your bar size — try **5m**/**15m** or **Exploratory** history mode."
    )

    if all_results.empty:
        st.caption("Run a scan first to build profitability stats.")
        return

    audit_df = get_portfolio_trade_audit(all_results["symbol"].dropna().tolist(), interval=interval, include_prepost=include_prepost, history_mode=history_mode)
    if audit_df.empty:
        st.info(
            "No audited trades were available for these symbols. "
            "The engine needs enough bars and at least one closed `READY` window with forward bars. "
            "Try **5m** or **15m** interval, **Exploratory** history mode, or fewer symbols if downloads are failing."
        )
        return

    if "audit_interval" in audit_df.columns and (audit_df["audit_interval"] != interval).any():
        alt = audit_df["audit_interval"].mode()
        st.caption(
            f"Some audits used fallback bar sizes (for example `{alt.iloc[0] if len(alt) else '?'}`) because the primary interval produced no scored history."
        )

    completed = audit_df[audit_df["outcome_status"] != "no_follow_through"].copy()
    if completed.empty:
        st.info(
            "Audits were found, but every row was still `no_follow_through` (no forward bars after the signal). "
            "Switch to **5m** or **15m** in the sidebar so the provider returns enough history to score outcomes."
        )
        return

    total_trades = len(completed)
    win_rate = (completed["actual_pnl"] > 0).mean() * 100 if total_trades else 0.0
    avg_r = float(completed["actual_r"].mean()) if completed["actual_r"].notna().any() else np.nan
    net_pnl = float(completed["actual_pnl"].sum())

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        render_metric_card("Audited trades", str(total_trades))
    with p2:
        render_metric_card("Win rate", f"{win_rate:.1f}%")
    with p3:
        render_metric_card("Avg R", "N/A" if pd.isna(avg_r) else f"{avg_r:.2f}")
    with p4:
        render_metric_card("Net P/L", f"{net_pnl:.4f}")

    symbol_stats = (
        completed.groupby("symbol", dropna=False)
        .agg(
            trades=("symbol", "size"),
            win_rate=("actual_pnl", lambda s: round((s > 0).mean() * 100, 1)),
            avg_r=("actual_r", "mean"),
            net_pnl=("actual_pnl", "sum"),
        )
        .reset_index()
        .sort_values(["net_pnl", "avg_r", "trades"], ascending=[False, False, False])
    )
    symbol_stats["avg_r"] = symbol_stats["avg_r"].round(2)
    symbol_stats["net_pnl"] = symbol_stats["net_pnl"].round(4)

    side_stats = (
        completed.groupby("side", dropna=False)
        .agg(
            trades=("side", "size"),
            win_rate=("actual_pnl", lambda s: round((s > 0).mean() * 100, 1)),
            avg_r=("actual_r", "mean"),
            net_pnl=("actual_pnl", "sum"),
        )
        .reset_index()
        .sort_values("net_pnl", ascending=False)
    )
    side_stats["avg_r"] = side_stats["avg_r"].round(2)
    side_stats["net_pnl"] = side_stats["net_pnl"].round(4)

    recent_trades = completed.sort_values("started", ascending=False).head(10).copy()
    recent_trades["started"] = recent_trades["started"].apply(_format_melbourne_time)
    recent_trades["ended"] = recent_trades["ended"].apply(_format_melbourne_time)
    recent_trades["result"] = np.where(recent_trades["actual_pnl"] > 0, "Win", "Loss")

    left, right = st.columns(2)
    with left:
        st.markdown("**Top symbols lately**")
        st.dataframe(symbol_stats.head(10), use_container_width=True, hide_index=True)
    with right:
        st.markdown("**Long vs Short performance**")
        st.dataframe(side_stats, use_container_width=True, hide_index=True)

    st.markdown("**Recent audited trades**")
    st.dataframe(
        recent_trades[["symbol", "started", "side", "result", "outcome_status", "actual_r", "actual_pnl", "ended"]],
        use_container_width=True,
        hide_index=True,
    )


def render_signal_timing_section(symbol: str, interval: str, include_prepost: bool, history_mode: str):
    lookback_lbl = _readiness_lookback_label()
    resolved = _resolve_signal_timing_data(symbol, interval=interval, include_prepost=include_prepost, history_mode=history_mode)
    timeline = resolved["timeline"]
    audit_df = resolved["audit_df"]
    st.subheader("Signal timing")
    st.caption(
        f"Shows when this symbol was `READY` to buy or sell over the last {lookback_lbl} "
        f"based on recent market bars using `{history_mode}` history rules. All times below are in Melbourne time."
    )

    if resolved["fallback_used"]:
        st.caption(
            f"Auto-fallback used `interval={resolved['used_interval']}` and `history mode={resolved['used_mode']}` "
            f"because the requested history did not have enough usable data."
        )

    if timeline.empty:
        st.info(f"Not enough recent bar data to build a readiness history for the last {lookback_lbl} for this symbol.")
        return

    summaries = {side: summarize_recent_readiness(timeline, side) for side in ["LONG", "SHORT"]}
    cols = st.columns(2)
    for idx, side in enumerate(["LONG", "SHORT"]):
        summary = summaries[side]
        last_started = summary["last_started"]
        last_ended = summary["last_ended"]
        last_duration = summary["last_duration"]

        if summary["current_ready"]:
            status_text = f"READY now for {_format_duration(summary['current_duration'])}"
            window_text = (
                f"Current ready window started: {_format_melbourne_time(last_started)}"
                if last_started is not None
                else "Current ready window started: N/A"
            )
        elif last_started is not None and last_ended is not None:
            status_text = "Not ready now"
            window_text = (
                f"Last ready window: {_format_melbourne_time(last_started)} "
                f"to {_format_melbourne_time(last_ended)}"
            )
        else:
            status_text = f"No ready signal seen in the last {lookback_lbl}"
            window_text = "Last ready window: N/A"

        with cols[idx]:
            with st.container(border=True):
                st.markdown(f"**{side} readiness**")
                st.write(status_text)
                st.caption(window_text)
                st.caption(f"Most recent duration: {_format_duration(last_duration)}")

    event_rows = []
    for side in ["LONG", "SHORT"]:
        for event in summaries[side]["events"]:
            event_rows.append(
                {
                    "side": side,
                    "started": _format_melbourne_time(event["started"]),
                    "ended": _format_melbourne_time(event["ended"]) if event["ended"] is not None else "Still active",
                    "duration": _format_duration(event["duration"]),
                    "status": event["status"],
                }
            )

    if event_rows:
        event_df = pd.DataFrame(event_rows).sort_values("started", ascending=False).head(6)
        st.dataframe(event_df, use_container_width=True, hide_index=True)

    st.markdown("**Recent signal audit**")
    st.caption("Assumption: entry at the ready bar, planned profit uses `Target 1`, planned risk uses `Stop`, and the audit keeps the trade alive until `Target 1`, `Stop`, or the end of the available bar history.")

    if audit_df.empty:
        st.info(f"No recent ready events with enough follow-through data were found for audit over the last {lookback_lbl}.")
        return

    recent_audit = audit_df.head(10).copy()
    completed_audit = recent_audit[recent_audit["outcome_status"] != "no_follow_through"].copy()
    win_count = int((completed_audit["actual_pnl"] > 0).sum()) if not completed_audit.empty else 0
    loss_count = int((completed_audit["actual_pnl"] < 0).sum()) if not completed_audit.empty else 0
    flat_count = int((completed_audit["actual_pnl"] == 0).sum()) if not completed_audit.empty else 0
    total_pnl = float(completed_audit["actual_pnl"].sum()) if not completed_audit.empty else 0.0

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        render_metric_card("Audited signals", str(len(recent_audit)))
    with a2:
        render_metric_card("Wins / Losses", f"{win_count} / {loss_count}")
    with a3:
        render_metric_card("Flat exits", str(flat_count))
    with a4:
        render_metric_card("Net actual P/L", f"{total_pnl:.4f}")

    display_audit = recent_audit.copy()
    display_audit["started"] = display_audit["started"].apply(_format_melbourne_time)
    display_audit["ended"] = display_audit["ended"].apply(_format_melbourne_time)
    if "readiness_ended" in display_audit.columns:
        display_audit["readiness_ended"] = display_audit["readiness_ended"].apply(_format_melbourne_time)
    display_audit["window_duration"] = display_audit["window_duration"].apply(_format_duration)
    display_audit["result"] = np.where(display_audit["actual_pnl"] > 0, "Win", np.where(display_audit["actual_pnl"] < 0, "Loss", "Flat"))
    display_audit = display_audit.rename(columns={"ended": "trade_exited"})
    audit_cols = [
        "started",
        "side",
        "entry",
        "stop",
        "target1",
        "suggested_profit",
        "suggested_loss",
        "trade_exited",
        "exit_price",
        "actual_pnl",
        "actual_r",
        "result",
        "outcome_status",
        "window_duration",
    ]
    st.dataframe(display_audit[audit_cols], use_container_width=True, hide_index=True)


def render_recent_ready_assets_section(symbols: List[str], interval: str, include_prepost: bool, history_mode: str) -> pd.DataFrame:
    lookback_lbl = _readiness_lookback_label()
    resolved = _resolve_recent_ready_assets(symbols, interval=interval, include_prepost=include_prepost, history_mode=history_mode)
    ready_assets = resolved["ready_assets"]
    st.subheader(f"Assets with ready signals in last {lookback_lbl}")
    st.caption(
        f"This helps you find which scanned symbols had at least one `READY` event recently, "
        f"even if they are not ready right now. The list currently uses `{history_mode}` history rules."
    )

    if resolved["fallback_used"]:
        st.caption(
            f"Auto-fallback used `interval={resolved['used_interval']}` and `history mode={resolved['used_mode']}` "
            f"to find usable recent-ready history."
        )

    if interval == "1m":
        st.caption(
            f"Note: `1m` history is usually limited by the data provider, so the full last-{lookback_lbl} window may not be available at that interval."
        )

    if ready_assets.empty:
        st.info(f"No scanned symbols showed a recorded `READY` event in the last {lookback_lbl} at the current interval.")
        st.markdown(
            """
**This is usually not a single “broken feed” — it is often a mix of data limits and strict rules:**

1. **Source (Yahoo Finance via `yfinance`)** — Free/delayed data, occasional gaps, and **1m** history is shorter and patchier than **5m/15m**. Symbols can fail the minimum bar count (`35+` bars) and be skipped quietly.

2. **What counts as `READY` here** — The history list only marks bars that pass the **full trigger** (break of prior high/low, VWAP side, relative volume, directional close, etc.) on **that bar interval**, inside the **last 48 hours**. The main scan can still show a symbol as interesting without any bar in that window counting as `READY`.

3. **Time and liquidity** — Outside regular session, on very quiet days, or with a small symbol list, it is normal to see **zero** qualifying bars in 48 hours.

**Things to try:** use **5m** or **15m**, scan more **liquid** names (mega caps / ETFs), run during **US cash session**, and confirm **Exploratory** history mode. Fallbacks were tried across: `"""
            + ", ".join(resolved.get("attempted", [])[:12])
            + ("…" if len(resolved.get("attempted", [])) > 12 else "")
            + "`."
        )
        return ready_assets

    display = ready_assets.copy()
    display["last_ready_at"] = display["last_ready_at"].apply(_format_melbourne_time)
    st.dataframe(display, use_container_width=True, hide_index=True)
    return ready_assets


def render_options_opportunity_board(view: pd.DataFrame):
    st.subheader("Options opportunity board")
    st.caption("Ranks underlying assets whose option-chain context looks most interesting. This is still an underlying scanner, not a contract picker.")

    if view.empty:
        st.info("No scanned assets available for options review.")
        return

    option_view = view.copy()
    option_view = option_view[option_view["option_bias"].isin(["BULLISH", "BEARISH", "MIXED"])].copy()
    if option_view.empty:
        st.info("No option-enabled assets were found in the current filtered results.")
        return

    option_view["contract_interest"] = (
        option_view["call_volume"].fillna(0)
        + option_view["put_volume"].fillna(0)
        + option_view["news_count"].fillna(0) * 100
    )
    option_view = option_view.sort_values(
        ["contract_interest", "atm_iv", "conviction"],
        ascending=[False, False, False],
    ).head(12)

    summary = option_view.copy()
    summary["expected_move_hint_pct"] = (summary["atm_iv"].astype(float) * 100 / 16).round(2)
    option_cols = [
        "symbol",
        "signal",
        "option_bias",
        "put_call_ratio",
        "atm_iv",
        "expected_move_hint_pct",
        "call_volume",
        "put_volume",
        "event_risk",
        "conviction",
        "headline",
    ]
    st.dataframe(summary[option_cols], use_container_width=True, hide_index=True)

    bull_count = int((option_view["option_bias"] == "BULLISH").sum())
    bear_count = int((option_view["option_bias"] == "BEARISH").sum())
    avg_iv = float(option_view["atm_iv"].dropna().mean()) if option_view["atm_iv"].notna().any() else np.nan
    highest_interest = option_view.iloc[0]["symbol"] if not option_view.empty else "N/A"

    o1, o2, o3, o4 = st.columns(4)
    with o1:
        render_metric_card("Option names", str(len(option_view)))
    with o2:
        render_metric_card("Bull / Bear bias", f"{bull_count} / {bear_count}")
    with o3:
        render_metric_card("Avg ATM IV", "N/A" if pd.isna(avg_iv) else f"{avg_iv:.2f}")
    with o4:
        render_metric_card("Top options name", str(highest_interest))


def render_telegram_diagnostics():
    bot_token = get_telegram_bot_token()
    chat_id = get_telegram_default_chat_id()
    test_status = st.session_state.telegram_test_status

    with st.expander("Telegram diagnostics"):
        d1, d2, d3 = st.columns(3)
        with d1:
            render_metric_card("Alerts enabled", "YES" if telegram_alerts_enabled() else "NO")
        with d2:
            render_metric_card("Bot token found", "YES" if bot_token else "NO")
        with d3:
            render_metric_card("Chat ID found", "YES" if chat_id else "NO")

        if bot_token:
            masked_token = f"{bot_token[:8]}...{bot_token[-4:]}" if len(bot_token) > 12 else "loaded"
            st.caption(f"Loaded bot token: {masked_token}")
        else:
            st.warning("No `TELEGRAM_BOT_TOKEN` found in local environment or Streamlit secrets.")

        if chat_id:
            st.caption(f"Loaded chat ID: {chat_id}")
        else:
            st.warning("No `TELEGRAM_CHAT_ID` found in local environment or Streamlit secrets.")

        st.caption("If you only added GitHub repository secrets, a local Streamlit app cannot read them. Local runs need Windows environment variables or `.streamlit/secrets.toml`.")

        if test_status is not None:
            if test_status.get("ok"):
                st.success("Last Telegram ping succeeded.")
                if test_status.get("response"):
                    st.code(str(test_status["response"]), language="json")
            else:
                st.error(f"Last Telegram ping failed: {test_status.get('error', 'unknown_error')}")

        if st.session_state.telegram_last_status:
            st.markdown("**Last alert send attempt**")
            st.dataframe(pd.DataFrame(st.session_state.telegram_last_status), use_container_width=True, hide_index=True)

if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "last_scan" not in st.session_state:
    st.session_state.last_scan = pd.DataFrame()
if "last_all_scan" not in st.session_state:
    st.session_state.last_all_scan = pd.DataFrame()
if "scanned_at" not in st.session_state:
    st.session_state.scanned_at = "Not run"
if "staged_order" not in st.session_state:
    st.session_state.staged_order = None
if "telegram_notified_signals" not in st.session_state:
    st.session_state.telegram_notified_signals = {}
if "telegram_last_status" not in st.session_state:
    st.session_state.telegram_last_status = []
if "telegram_test_status" not in st.session_state:
    st.session_state.telegram_test_status = None

st.sidebar.title("Scanner Setup")
universe_name = st.sidebar.selectbox("Choose universe", list(DEFAULT_UNIVERSES.keys()) + ["Custom"])
if universe_name == "Custom":
    custom_symbols = st.sidebar.text_area(
        "Paste symbols (comma separated)",
        value=DEFAULT_CUSTOM_SYMBOLS,
        height=140,
    )
    symbols = parse_symbol_input(custom_symbols)
else:
    symbols = DEFAULT_UNIVERSES[universe_name]

period = st.sidebar.selectbox("Lookback period", ["1d", "5d", "1mo"], index=0)
interval = st.sidebar.selectbox("Bar interval", ["1m", "2m", "5m", "15m", "30m"], index=2)
mode_options = list(SCANNER_MODES.keys())
default_scan_mode_index = mode_options.index(DEFAULT_SCAN_MODE) if DEFAULT_SCAN_MODE in mode_options else 0
default_history_mode_index = mode_options.index(DEFAULT_HISTORY_MODE) if DEFAULT_HISTORY_MODE in mode_options else 0
scan_mode = st.sidebar.selectbox("Scanner mode", mode_options, index=default_scan_mode_index)
history_mode = st.sidebar.selectbox("History mode", mode_options, index=default_history_mode_index)
include_prepost = st.sidebar.checkbox("Include pre/post market", value=True)
scan_pause = st.sidebar.slider("Pause between requests (seconds)", 0.0, 0.5, 0.0, 0.05)
max_symbols = st.sidebar.slider("Max symbols to scan", 5, 250, min(50, max(5, len(symbols))))
auto_refresh = st.sidebar.checkbox("Auto refresh every 2 min", value=False)
run_scan = st.sidebar.button("Run Pro Scan", type="primary", use_container_width=True)
test_telegram_ping = st.sidebar.button("Send Telegram test ping", use_container_width=True)

st.title("Pro Intraday Market Scanner")
st.caption("Ranks symbols for manual intraday decisions using regime, structure, momentum, multi-timeframe alignment, triggers and risk planning.")
st.caption("All displayed times use Australia/Melbourne.")
st.caption(f"Live scan mode: `{scan_mode}`. History mode: `{history_mode}`.")

with st.expander("How to use this tool"):
    st.write(
        """
        1. Check the market regime first. If it is choppy, respect it.
        2. Run a scan and focus on trigger-ready setups.
        3. Inspect support, resistance, entry, stop and targets.
        4. Use watchlist to keep only the names worth stalking.
        5. Backtest before trusting any idea too much.
        """
    )
    st.caption("Telegram alerts: store `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, and optionally `TELEGRAM_ALERTS_ENABLED=true` in environment variables or Streamlit secrets.")

render_beginner_guide()
render_htf_ltf_workflow_note()
render_telegram_diagnostics()

regime = detect_market_regime()

if test_telegram_ping:
    bot_token = get_telegram_bot_token()
    chat_id = get_telegram_default_chat_id()
    ping_message = f"Ping from Pro Intraday Scanner at {pd.Timestamp.now(tz=MELBOURNE_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')}"
    st.session_state.telegram_test_status = send_telegram_message(
        bot_token=bot_token,
        chat_id=chat_id,
        message=ping_message,
    )

if run_scan or auto_refresh:
    shortlist = symbols[:max_symbols]
    all_results = scan_symbols(
        symbols=shortlist,
        period=period,
        interval=interval,
        include_prepost=include_prepost,
        pause_s=scan_pause,
        regime=regime,
        mode=scan_mode,
    )
    scan_timestamp = pd.Timestamp.now(tz=MELBOURNE_TZ)
    st.session_state.last_all_scan = all_results
    st.session_state.scanned_at = scan_timestamp.strftime("%Y-%m-%d %H:%M:%S %Z")
    results = all_results.copy()
    st.session_state.last_scan = results
    if telegram_alerts_enabled():
        bot_token = get_telegram_bot_token()
        chat_id = get_telegram_default_chat_id()
        ready_alerts = results[results["trigger_ready"]].copy() if not results.empty else pd.DataFrame()
        st.session_state.telegram_last_status = send_ready_signal_notifications(
            ready_df=ready_alerts,
            bot_token=bot_token,
            chat_id=chat_id,
            scan_time_label=st.session_state.scanned_at,
            notified_cache=st.session_state.telegram_notified_signals,
        )

all_results = st.session_state.last_all_scan
if not all_results.empty:
    all_results = all_results.copy()
    for column, default_value in RESULT_DEFAULTS.items():
        if column not in all_results.columns:
            all_results[column] = default_value
    all_results["accepted"] = True
    all_results["acceptance_status"] = "ok"
    all_results["qualified"] = True
    all_results["status"] = all_results.get("scanner_status", all_results["status"])
    st.session_state.last_all_scan = all_results
results = all_results.copy() if not all_results.empty else pd.DataFrame()
st.session_state.last_scan = results

r1, r2, r3, r4, r5 = st.columns(5)
with r1:
    render_metric_card("Market regime", regime["overall"])
with r2:
    render_metric_card("SPY", f"{regime['spy']['state']} ({regime['spy']['change_pct']}%)")
with r3:
    render_metric_card("QQQ", f"{regime['qqq']['state']} ({regime['qqq']['change_pct']}%)")
with r4:
    render_metric_card("VIX", str(regime["vix"]))
with r5:
    render_metric_card("Last scan", st.session_state.scanned_at)

m1, m2, m3, m4 = st.columns(4)
with m1:
    render_metric_card("Universe size", str(min(len(symbols), max_symbols)))
with m2:
    render_metric_card("Ranked symbols", str(len(results)))
with m3:
    ready_count = int(results["trigger_ready"].sum()) if not results.empty else 0
    render_metric_card("Trigger-ready", str(ready_count))
with m4:
    render_metric_card("Watchlist", str(len(st.session_state.watchlist)))

if all_results.empty:
    st.warning("No scan results yet. Either the market is dead, the data provider returned nothing, or the chosen symbols are not moving.")
    if auto_refresh:
        time.sleep(120)
        st.rerun()
    st.stop()

if results.empty:
    st.warning("No symbols were returned in the latest scan.")

st.markdown("---")

summary_left, summary_right = st.columns([1.35, 1])

with summary_left:
    st.subheader("Live opportunities")
    alerts = results[results["trigger_ready"]].head(6)
    if not alerts.empty:
        for _, row in alerts.iterrows():
            badge_signal = "🟢 LONG" if row["signal"] == "LONG" else "🔴 SHORT"
            badge_ready = "⚡ READY" if row["trigger_ready"] else "⏳ WAIT"
            st.markdown(
                f"**{row['symbol']}**  \
{badge_signal} | {badge_ready} | Conviction **{row['conviction']:.1f}** | RelVol **{row['relvol']}** | Move **{row['change_pct']}%**  \
{row['trigger_text']}"
            )
    else:
        st.info("No trigger-ready setups right now.")

with summary_right:
    st.subheader("Market notes")
    with st.container(border=True):
        st.markdown(
            f"**Regime:** {regime['overall']}  \
**SPY:** {regime['spy']['state']} ({regime['spy']['change_pct']}%)  \
**QQQ:** {regime['qqq']['state']} ({regime['qqq']['change_pct']}%)  \
**VIX:** {regime['vix']}"
        )
        st.caption("If the market is mixed or choppy, the scanner will naturally produce weaker setups.")
        st.caption("News, options, and event risk below are public-data enrichments. Spread is still a labeled proxy unless you wire in a broker feed.")
        if telegram_alerts_enabled():
            if not get_telegram_bot_token():
                st.warning("Telegram alerts are enabled, but no bot token was found in environment/secrets.")
            elif not get_telegram_default_chat_id():
                st.warning("Telegram alerts are enabled, but no chat ID was found in environment/secrets.")
            elif st.session_state.telegram_last_status:
                sent_ok = sum(1 for item in st.session_state.telegram_last_status if item.get("ok"))
                sent_fail = sum(1 for item in st.session_state.telegram_last_status if not item.get("ok"))
                st.caption(f"Telegram alert run: {sent_ok} sent, {sent_fail} failed.")
        else:
            st.caption("Telegram alerts are off unless `TELEGRAM_ALERTS_ENABLED=true` is set in environment/secrets.")
        if st.session_state.telegram_test_status is not None:
            if st.session_state.telegram_test_status.get("ok"):
                st.success("Telegram ping sent successfully.")
            else:
                st.error(f"Telegram ping failed: {st.session_state.telegram_test_status.get('error', 'unknown_error')}")
                st.caption("If you only added GitHub repository secrets, a local Streamlit app cannot read them. Local runs need environment variables or `.streamlit/secrets.toml`.")

st.markdown("---")
candidate_view = results.copy()

summary_cols = [
    "symbol", "status", "signal", "trigger_ready", "quality_score", "conviction", "price", "change_pct", "relvol", "rsi",
    "atr_pct", "spread_proxy_pct", "news_sentiment", "option_bias", "event_risk",
    "liquidity_label", "mtf_state", "entry", "stop", "target1", "rr1", "reasons"
]

main_left, main_right = st.columns([1.35, 1])

with main_left:
    st.subheader("Ranked candidates")
    st.caption("Everything returned by the scan is shown here. Nothing is filtered out in the app, so this section is purely ranking-based.")
    if candidate_view.empty:
        st.info("No ranked candidates are available.")
    else:
        display_candidates = candidate_view.copy()
        display_candidates["status"] = display_candidates["status"].str.replace("_", " ").str.title()

        def row_style(row):
            signal_bg = "#e8f5e9" if row["signal"] == "LONG" else "#ffebee"
            trigger_bg = "#fff8e1" if row["trigger_ready"] else ""
            return [
                "background-color: " + signal_bg if col == "signal"
                else ("background-color: " + trigger_bg if col == "trigger_ready"
                else "")
                for col in row.index
            ]

        styled_view = display_candidates[summary_cols].style.apply(row_style, axis=1)
        st.dataframe(styled_view, use_container_width=True, hide_index=True)
        render_options_opportunity_board(candidate_view)

    render_profitable_traits_section(all_results, interval=interval, include_prepost=include_prepost, history_mode=history_mode)

with main_right:
    st.subheader("Watchlist")
    watchlist_source = candidate_view["symbol"].tolist()
    add_symbol = st.selectbox("Add symbol to watchlist", [""] + watchlist_source)
    wc1, wc2 = st.columns([1, 1])
    with wc1:
        if st.button("Add to watchlist") and add_symbol:
            if add_symbol not in st.session_state.watchlist:
                st.session_state.watchlist.append(add_symbol)
    with wc2:
        if st.button("Clear watchlist"):
            st.session_state.watchlist = []

    with st.container(border=True):
        if st.session_state.watchlist:
            st.write(", ".join(st.session_state.watchlist))
        else:
            st.caption("No symbols in watchlist yet.")

    st.subheader("Top 5 now")
    top_now = candidate_view.head(5)
    if top_now.empty:
        st.caption("No current candidates to rank right now.")
    else:
        for i, row in top_now.iterrows():
            signal_badge = "🟢 LONG" if row["signal"] == "LONG" else "🔴 SHORT"
            trigger_badge = "⚡ READY" if row["trigger_ready"] else "⏳ WAIT"
            with st.container(border=True):
                st.markdown(
                    f"**#{i+1} {row['symbol']}** — {signal_badge} | {trigger_badge}  \
Quality: **{row['quality_score']}** | Conviction: **{row['conviction']:.1f}** | Move: **{row['change_pct']}%** | RelVol: **{row['relvol']}** | RSI: **{row['rsi']}** | RR1: **{row['rr1']}**"
                )
                st.caption(row["reasons"])

st.markdown("---")
st.subheader("Setup inspection")
st.caption("Selected symbol with chart, trade plan, rejection reason if any, and recent readiness history.")
inspect_pool = all_results[all_results["chart"].notna()].copy() if "chart" in all_results.columns else all_results.copy()
if inspect_pool.empty:
    st.info("No chart-capable symbols are available from the latest scan.")
    if auto_refresh:
        time.sleep(120)
        st.rerun()
    st.stop()
recent_ready_assets = render_recent_ready_assets_section(
    inspect_pool["symbol"].tolist(),
    interval=interval,
    include_prepost=include_prepost,
    history_mode=history_mode,
)
inspect_symbols = inspect_pool["symbol"].tolist()
default_symbol = recent_ready_assets.iloc[0]["symbol"] if not recent_ready_assets.empty else inspect_symbols[0]
default_index = inspect_symbols.index(default_symbol) if default_symbol in inspect_symbols else 0
selected_symbol = st.selectbox("Choose symbol", inspect_symbols, index=default_index, key="detail_symbol")
chart_mode = st.radio(
    "Chart type",
    ["Candlestick", "Line + Indicators", "Close Only"],
    horizontal=True,
    key="chart_mode",
)
show_volume = st.checkbox("Show volume", value=True, key="show_volume")
show_vwap = st.checkbox("Show VWAP", value=True, key="show_vwap")
show_ema9 = st.checkbox("Show EMA9", value=True, key="show_ema9")
show_ema20 = st.checkbox("Show EMA20", value=True, key="show_ema20")
show_sma50 = st.checkbox("Show SMA50", value=True, key="show_sma50")
recent_bars = st.slider("Recent bars to display", min_value=30, max_value=160, value=80, step=10, key="recent_bars")
selected = inspect_pool.loc[inspect_pool["symbol"] == selected_symbol].iloc[0]
render_setup_detail(
    selected,
    chart_mode=chart_mode,
    show_volume=show_volume,
    recent_bars=recent_bars,
    indicator_flags={
        "VWAP": show_vwap,
        "EMA9": show_ema9,
        "EMA20": show_ema20,
        "SMA50": show_sma50,
    },
)
render_market_intel(selected)
render_signal_timing_section(selected_symbol, interval=interval, include_prepost=include_prepost, history_mode=history_mode)

st.markdown("---")
st.subheader("Execution ticket")
st.caption("This stages a ticket for your Plus500 demo workflow. It does not auto-send orders to the retail Plus500 platform from this app.")

execution_mode = st.selectbox(
    "Execution mode",
    ["Plus500 manual confirmation", "Direct API placeholder"],
    index=0,
    key="execution_mode",
)
ex1, ex2, ex3, ex4 = st.columns(4)
with ex1:
    order_type = st.selectbox("Order type", ["Market", "Limit"], index=0, key="order_type")
with ex2:
    time_in_force = st.selectbox("Time in force", ["DAY", "GTC"], index=0, key="time_in_force")
with ex3:
    quantity = st.number_input(
        "Units / contracts",
        min_value=1.0,
        value=1.0,
        step=1.0,
        key="order_quantity",
        help="Match this carefully to how Plus500 sizes the selected instrument.",
    )
with ex4:
    limit_default = float(selected["entry"]) if pd.notna(selected["entry"]) else float(selected["price"])
    limit_price = st.number_input(
        "Limit price",
        value=limit_default,
        step=0.01,
        format="%.4f",
        key="limit_price",
        disabled=order_type != "Limit",
    )

order_notes = st.text_input(
    "Execution notes",
    value=f"{selected['signal']} setup from scanner",
    key="execution_notes",
)
confirm_manual = st.checkbox(
    "I understand I still need to confirm the order manually inside Plus500.",
    value=True,
    key="confirm_manual_execution",
)

preview_ticket = build_order_ticket(
    selected=selected,
    broker_mode=execution_mode,
    quantity=quantity,
    order_type=order_type,
    time_in_force=time_in_force,
    limit_price=limit_price if order_type == "Limit" else None,
    notes=order_notes,
)

px1, px2, px3, px4 = st.columns(4)
with px1:
    render_metric_card("Side", preview_ticket.side)
with px2:
    render_metric_card("Qty", str(int(preview_ticket.quantity) if float(preview_ticket.quantity).is_integer() else preview_ticket.quantity))
with px3:
    render_metric_card("Risk / unit", "N/A" if preview_ticket.risk_per_unit is None else f"{preview_ticket.risk_per_unit:.4f}")
with px4:
    render_metric_card("Total est. risk", "N/A" if preview_ticket.estimated_total_risk is None else f"{preview_ticket.estimated_total_risk:.2f}")

action_col1, action_col2 = st.columns([1, 1])
with action_col1:
    if st.button("Stage order ticket", use_container_width=True, disabled=not confirm_manual):
        adapter = get_broker_adapter(execution_mode)
        st.session_state.staged_order = adapter.stage_order(preview_ticket)
with action_col2:
    if st.button("Clear staged ticket", use_container_width=True):
        st.session_state.staged_order = None

if st.session_state.staged_order:
    staged_order = st.session_state.staged_order
    st.info(f"Ticket status: {staged_order['status']}")
    st.dataframe(ticket_to_frame(staged_order), use_container_width=True, hide_index=True)
    st.code(pd.Series(staged_order).to_json(indent=2), language="json")
    for instruction in staged_order.get("instructions", []):
        st.caption(instruction)
    st.warning("Check the live Plus500 bid/ask, spread, instrument mapping, and sizing before you submit anything in the demo platform.")

st.markdown("---")
st.subheader("Quick backtest")
backtest_symbols = st.multiselect(
    "Choose symbols to backtest",
    options=inspect_pool["symbol"].tolist(),
    default=inspect_pool["symbol"].head(3).tolist() if len(inspect_pool) >= 3 else inspect_pool["symbol"].tolist(),
)
if st.button("Run quick backtest"):
    bt_rows = [backtest_symbol(sym, include_prepost=include_prepost) for sym in backtest_symbols]
    bt_df = pd.DataFrame(bt_rows)
    st.dataframe(bt_df, use_container_width=True, hide_index=True)

export_df = all_results.drop(columns=["chart"], errors="ignore")
csv_bytes = export_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download results as CSV",
    data=csv_bytes,
    file_name=f"pro_intraday_scan_{datetime.now(MELBOURNE_TZ).strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv",
)

with st.expander("Future upgrades"):
    st.write(
        "This build now includes a public-data news snapshot, option chain snapshot, event calendar filter, a labeled spread/liquidity proxy, and manual-confirm execution tickets. The next upgrade would be swapping those providers to real-time news and a supported broker or futures API."
    )

if auto_refresh:
    time.sleep(120)
    st.rerun()
