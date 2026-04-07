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
from scanner import scan_symbols, backtest_symbol
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

regime = detect_market_regime()

if run_scan or auto_refresh:
    shortlist = symbols[:max_symbols]
    results = scan_symbols(
        symbols=shortlist,
        period=period,
        interval=interval,
        include_prepost=include_prepost,
        pause_s=scan_pause,
        regime=regime,
    )
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
    st.session_state.scanned_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
