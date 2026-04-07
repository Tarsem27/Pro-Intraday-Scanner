# ==============================
# file: app.py
# ==============================
from datetime import datetime
import time

import pandas as pd
import streamlit as st

from config import DEFAULT_UNIVERSES, DEFAULT_CUSTOM_SYMBOLS
from data_utils import detect_market_regime, parse_symbol_input
from execution import build_order_ticket, get_broker_adapter, ticket_to_frame
from scanner import backtest_symbol, get_readiness_timeline, scan_symbols, summarize_recent_readiness
from ui_components import render_market_intel, render_metric_card, render_setup_detail

st.set_page_config(page_title="Pro Intraday Scanner", layout="wide")

RESULT_DEFAULTS = {
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
                - A good setup with bad risk sizing can still be a bad trade.

                **Good first-demo-account habits**

                - Trade only very liquid names at first.
                - Avoid trading right before earnings or major events.
                - Risk a tiny amount and practice following stops.
                - Review whether the setup failed because the idea was wrong or because the execution was poor.
                """
            )


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


def render_signal_timing_section(symbol: str, interval: str, include_prepost: bool):
    timeline = get_readiness_timeline(symbol, interval=interval, include_prepost=include_prepost, lookback_hours=48)
    st.subheader("Signal timing")
    st.caption("Shows when this symbol was `READY` to buy or sell over the last 48 hours based on recent market bars.")

    if timeline.empty:
        st.info("Not enough recent bar data to build a 48-hour readiness history for this symbol.")
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
                f"Current ready window started: {pd.Timestamp(last_started).strftime('%Y-%m-%d %H:%M:%S')}"
                if last_started is not None
                else "Current ready window started: N/A"
            )
        elif last_started is not None and last_ended is not None:
            status_text = "Not ready now"
            window_text = (
                f"Last ready window: {pd.Timestamp(last_started).strftime('%Y-%m-%d %H:%M:%S')} "
                f"to {pd.Timestamp(last_ended).strftime('%Y-%m-%d %H:%M:%S')}"
            )
        else:
            status_text = "No ready signal seen in the last 48h"
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
                    "started": pd.Timestamp(event["started"]).strftime("%Y-%m-%d %H:%M:%S"),
                    "ended": pd.Timestamp(event["ended"]).strftime("%Y-%m-%d %H:%M:%S") if event["ended"] is not None else "Still active",
                    "duration": _format_duration(event["duration"]),
                    "status": event["status"],
                }
            )

    if event_rows:
        event_df = pd.DataFrame(event_rows).sort_values("started", ascending=False).head(6)
        st.dataframe(event_df, use_container_width=True, hide_index=True)

if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "last_scan" not in st.session_state:
    st.session_state.last_scan = pd.DataFrame()
if "scanned_at" not in st.session_state:
    st.session_state.scanned_at = "Not run"
if "staged_order" not in st.session_state:
    st.session_state.staged_order = None

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
include_prepost = st.sidebar.checkbox("Include pre/post market", value=True)
min_conviction = st.sidebar.slider("Minimum conviction", 0, 100, 45)
min_relvol = st.sidebar.slider("Minimum relative volume", 0.0, 5.0, 0.5, 0.1)
min_abs_change = st.sidebar.slider("Minimum absolute % move", 0.0, 10.0, 0.2, 0.1)
max_spread_proxy = st.sidebar.slider("Max spread proxy %", 0.1, 5.0, 2.0, 0.1)
only_trigger_ready = st.sidebar.checkbox("Only show trigger-ready setups", value=False)
event_risk_filter = st.sidebar.selectbox("Event risk filter", ["All", "Hide HIGH", "Only HIGH"], index=0)
news_filter = st.sidebar.selectbox("News tone filter", ["All", "Bullish only", "Bearish only", "Neutral only"], index=0)
options_filter = st.sidebar.selectbox("Options flow filter", ["All", "Bullish only", "Bearish only"], index=0)
scan_pause = st.sidebar.slider("Pause between requests (seconds)", 0.0, 0.5, 0.0, 0.05)
max_symbols = st.sidebar.slider("Max symbols to scan", 5, 250, min(50, max(5, len(symbols))))
auto_refresh = st.sidebar.checkbox("Auto refresh every 2 min", value=False)
run_scan = st.sidebar.button("Run Pro Scan", type="primary", use_container_width=True)

st.title("Pro Intraday Market Scanner")
st.caption("Ranks symbols for manual intraday decisions using regime, structure, momentum, multi-timeframe alignment, triggers and risk planning.")

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

render_beginner_guide()

regime = detect_market_regime()

if run_scan or auto_refresh:
    shortlist = symbols[:max_symbols]
    raw_results = scan_symbols(
        symbols=shortlist,
        period=period,
        interval=interval,
        include_prepost=include_prepost,
        pause_s=scan_pause,
        regime=regime,
    )
    results = raw_results.copy()
    scan_timestamp = pd.Timestamp.now()
    if not results.empty:
        results = results[
            (results["conviction"] >= min_conviction)
            & (results["relvol"] >= min_relvol)
            & (results["change_pct"].abs() >= min_abs_change)
            & (results["spread_proxy_pct"] <= max_spread_proxy)
        ].reset_index(drop=True)
        if only_trigger_ready:
            results = results[results["trigger_ready"]].reset_index(drop=True)
    st.session_state.last_scan = results
    st.session_state.scanned_at = scan_timestamp.strftime("%Y-%m-%d %H:%M:%S")

results = st.session_state.last_scan
if not results.empty:
    results = results.copy()
    for column, default_value in RESULT_DEFAULTS.items():
        if column not in results.columns:
            results[column] = default_value
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
    render_metric_card("Qualified setups", str(len(results)))
with m3:
    ready_count = int(results["trigger_ready"].sum()) if not results.empty else 0
    render_metric_card("Trigger-ready", str(ready_count))
with m4:
    render_metric_card("Watchlist", str(len(st.session_state.watchlist)))

if results.empty:
    st.warning("No setups yet. Either the market is dead, the filters are strict, or the chosen symbols are not moving.")
    if auto_refresh:
        time.sleep(120)
        st.rerun()
    st.stop()

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

st.markdown("---")
show_direction = st.radio("Show setups", ["All", "LONG only", "SHORT only"], horizontal=True)
view = results.copy()
if show_direction == "LONG only":
    view = view[view["signal"] == "LONG"].reset_index(drop=True)
elif show_direction == "SHORT only":
    view = view[view["signal"] == "SHORT"].reset_index(drop=True)

if event_risk_filter == "Hide HIGH":
    view = view[view["event_risk"] != "HIGH"].reset_index(drop=True)
elif event_risk_filter == "Only HIGH":
    view = view[view["event_risk"] == "HIGH"].reset_index(drop=True)

if news_filter == "Bullish only":
    view = view[view["news_sentiment"] == "BULLISH"].reset_index(drop=True)
elif news_filter == "Bearish only":
    view = view[view["news_sentiment"] == "BEARISH"].reset_index(drop=True)
elif news_filter == "Neutral only":
    view = view[view["news_sentiment"] == "NEUTRAL"].reset_index(drop=True)

if options_filter == "Bullish only":
    view = view[view["option_bias"] == "BULLISH"].reset_index(drop=True)
elif options_filter == "Bearish only":
    view = view[view["option_bias"] == "BEARISH"].reset_index(drop=True)

if view.empty:
    st.warning("Your current directional/news/event/options filters removed every setup. Loosen one of the filters and rerun.")
    if auto_refresh:
        time.sleep(120)
        st.rerun()
    st.stop()

summary_cols = [
    "symbol", "signal", "trigger_ready", "conviction", "price", "change_pct", "relvol", "rsi",
    "atr_pct", "spread_proxy_pct", "news_sentiment", "option_bias", "event_risk",
    "liquidity_label", "mtf_state", "entry", "stop", "target1", "rr1", "reasons"
]

main_left, main_right = st.columns([1.35, 1])

with main_left:
    st.subheader("Ranked candidates")
    def row_style(row):
        signal_bg = '#e8f5e9' if row['signal'] == 'LONG' else '#ffebee'
        trigger_bg = '#fff8e1' if row['trigger_ready'] else ''
        return ['background-color: ' + signal_bg if col == 'signal' else ('background-color: ' + trigger_bg if col == 'trigger_ready' else '') for col in row.index]

    styled_view = view[summary_cols].style.apply(row_style, axis=1)
    st.dataframe(styled_view, use_container_width=True, hide_index=True)

with main_right:
    st.subheader("Watchlist")
    add_symbol = st.selectbox("Add symbol to watchlist", [""] + view["symbol"].tolist())
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
    for i, row in view.head(5).iterrows():
        signal_badge = "🟢 LONG" if row["signal"] == "LONG" else "🔴 SHORT"
        trigger_badge = "⚡ READY" if row["trigger_ready"] else "⏳ WAIT"
        with st.container(border=True):
            st.markdown(
                f"**#{i+1} {row['symbol']}** — {signal_badge} | {trigger_badge}  \
Conviction: **{row['conviction']:.1f}** | Move: **{row['change_pct']}%** | RelVol: **{row['relvol']}** | RSI: **{row['rsi']}** | RR1: **{row['rr1']}**"
            )
            st.caption(row["reasons"])

st.markdown("---")
st.subheader("Setup inspection")
st.caption("Selected trade setup with cleaner summary cards, chart levels and context.")
selected_symbol = st.selectbox("Choose symbol", view["symbol"].tolist(), key="detail_symbol")
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
selected = view.loc[view["symbol"] == selected_symbol].iloc[0]
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
render_signal_timing_section(selected_symbol, interval=interval, include_prepost=include_prepost)

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
    options=view["symbol"].tolist(),
    default=view["symbol"].head(3).tolist() if len(view) >= 3 else view["symbol"].tolist(),
)
if st.button("Run quick backtest"):
    bt_rows = [backtest_symbol(sym, include_prepost=include_prepost) for sym in backtest_symbols]
    bt_df = pd.DataFrame(bt_rows)
    st.dataframe(bt_df, use_container_width=True, hide_index=True)

export_df = view.drop(columns=["chart"], errors="ignore")
csv_bytes = export_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download results as CSV",
    data=csv_bytes,
    file_name=f"pro_intraday_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv",
)

with st.expander("Future upgrades"):
    st.write(
        "This build now includes a public-data news snapshot, option chain snapshot, event calendar filter, a labeled spread/liquidity proxy, and manual-confirm execution tickets. The next upgrade would be swapping those providers to real-time news and a supported broker or futures API."
    )

if auto_refresh:
    time.sleep(120)
    st.rerun()
