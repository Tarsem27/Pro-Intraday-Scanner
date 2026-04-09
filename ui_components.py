
# ==============================
# file: ui_components.py
# ==============================
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


def render_metric_card(label: str, value: str):
    st.markdown(
        f"""
        <div style="padding:10px 14px;border:1px solid #333;border-radius:14px;">
            <div style="font-size:0.85rem;opacity:0.75;">{label}</div>
            <div style="font-size:1.25rem;font-weight:700;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_price_chart(
    chart_df: pd.DataFrame,
    chart_mode: str = "Candlestick",
    show_volume: bool = True,
    indicator_flags: dict | None = None,
    levels: dict | None = None,
):
    if indicator_flags is None:
        indicator_flags = {"VWAP": True, "EMA9": True, "EMA20": True, "SMA50": True}
    if levels is None:
        levels = {}

    if show_volume:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.05)
    else:
        fig = make_subplots(rows=1, cols=1)

    price_row = 1
    vol_row = 2

    if chart_mode == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=chart_df.index,
                open=chart_df["Open"],
                high=chart_df["High"],
                low=chart_df["Low"],
                close=chart_df["Close"],
                name="Candles",
            ),
            row=price_row,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=chart_df.index,
                y=chart_df["Close"],
                mode="lines",
                name="Close",
            ),
            row=price_row,
            col=1,
        )

    if chart_mode != "Close Only":
        for col in ["VWAP", "EMA9", "EMA20", "SMA50"]:
            if col in chart_df.columns and indicator_flags.get(col, True):
                fig.add_trace(
                    go.Scatter(
                        x=chart_df.index,
                        y=chart_df[col],
                        mode="lines",
                        name=col,
                    ),
                    row=price_row,
                    col=1,
                )

    for level_name, level_value in levels.items():
        if pd.notna(level_value):
            fig.add_hline(
                y=float(level_value),
                line_dash="dot",
                annotation_text=level_name,
                annotation_position="right",
                row=price_row,
                col=1,
            )

    if show_volume and "Volume" in chart_df.columns:
        fig.add_trace(
            go.Bar(
                x=chart_df.index,
                y=chart_df["Volume"],
                name="Volume",
                opacity=0.45,
            ),
            row=vol_row,
            col=1,
        )

    fig.update_layout(
        height=560 if show_volume else 450,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def render_setup_detail(
    selected,
    chart_mode: str = "Candlestick",
    show_volume: bool = True,
    recent_bars: int = 80,
    indicator_flags: dict | None = None,
):
    chart_df = selected["chart"].copy().tail(recent_bars)
    level_lines = {
        "Support": selected.get("support"),
        "Resistance": selected.get("resistance"),
        "Prev High": selected.get("prev_high"),
        "Prev Low": selected.get("prev_low"),
        "Entry": selected.get("entry"),
        "Stop": selected.get("stop"),
        "Target 1": selected.get("target1"),
    }

    top1, top2, top3 = st.columns(3)
    with top1:
        with st.container(border=True):
            st.markdown(
                f"""**Setup summary**  
Symbol: **{selected['symbol']}**  
Signal: **{selected['signal']}**  
Conviction: **{selected['conviction']:.1f}**  
Trigger: **{'READY' if selected['trigger_ready'] else 'WAIT'}**  
MTF: **{selected['mtf_state']}**"""
            )
    with top2:
        with st.container(border=True):
            st.markdown(
                f"""**Trade plan**  
Entry: **{selected['entry']}**  
Stop: **{selected['stop']}**  
Target 1: **{selected['target1']}**  
Target 2: **{selected['target2']}**  
R:R to T1: **{selected['rr1']}**"""
            )
    with top3:
        with st.container(border=True):
            st.markdown(
                f"""**Levels**  
Support: **{selected['support']}**  
Resistance: **{selected['resistance']}**  
Prev High: **{selected['prev_high']}**  
Prev Low: **{selected['prev_low']}**  
Regime: **{selected['regime']}**"""
            )

    fig = build_price_chart(
        chart_df,
        chart_mode=chart_mode,
        show_volume=show_volume,
        indicator_flags=indicator_flags,
        levels=level_lines,
    )
    st.plotly_chart(fig, use_container_width=True)

    lower_left, lower_right = st.columns([1.15, 1])
    with lower_left:
        with st.container(border=True):
            st.markdown("**Reasoning**")
            st.write(selected["reasons"])
            st.caption(f"Trigger: {selected['trigger_text']}")
            td = selected.get("trigger_detail") or ""
            if td:
                with st.expander("Why WAIT / what READY needs (this bar)", expanded=False):
                    st.write(td)
            st.caption(f"Catalyst: {selected['catalyst']}")
    with lower_right:
        with st.container(border=True):
            st.markdown("**Context**")
            st.write(f"Price: {selected['price']} | Change: {selected['change_pct']}% | RelVol: {selected['relvol']} | RSI: {selected['rsi']}")
            st.write(f"ATR%: {selected['atr_pct']} | Spread proxy: {selected['spread_proxy_pct']}%")


def render_market_intel(selected):
    intel1, intel2, intel3 = st.columns(3)

    with intel1:
        with st.container(border=True):
            st.markdown("**News / Catalyst**")
            st.write(f"Tone: {selected.get('news_sentiment', 'UNKNOWN')} | Stories: {selected.get('news_count', 0)}")
            st.caption(selected.get("headline", "No headline available"))

    with intel2:
        with st.container(border=True):
            st.markdown("**Options Snapshot**")
            st.write(
                f"Bias: {selected.get('option_bias', 'N/A')} | Expiry: {selected.get('option_expiry', 'N/A')}"
            )
            st.write(
                f"PCR: {selected.get('put_call_ratio', 'N/A')} | ATM IV: {selected.get('atm_iv', 'N/A')}"
            )
            st.caption(
                f"Call vol: {selected.get('call_volume', 0)} | Put vol: {selected.get('put_volume', 0)}"
            )

    with intel3:
        with st.container(border=True):
            st.markdown("**Event / Liquidity**")
            st.write(
                f"Event risk: {selected.get('event_risk', 'UNKNOWN')} | {selected.get('event_summary', 'No event data')}"
            )
            st.write(
                f"Liquidity: {selected.get('liquidity_label', 'UNKNOWN')} ({selected.get('liquidity_score', 'N/A')})"
            )
            st.caption(
                f"Spread est: {selected.get('spread_estimate_bps', 'N/A')} bps via {selected.get('spread_source', 'N/A')}"
            )


def render_top_ideas(view):
    st.subheader("Top 5 now")
    for i, row in view.head(5).iterrows():
        st.markdown(
            f"**#{i+1} {row['symbol']} — {row['signal']} {'READY' if row['trigger_ready'] else 'WAIT'}**  \n"
            f"Conviction: {row['conviction']:.1f} | Move: {row['change_pct']}% | RelVol: {row['relvol']} | RSI: {row['rsi']} | RR1: {row['rr1']}  \n"
            f"Why: {row['reasons']}"
        )
