import json
import os
from typing import Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st


def get_telegram_bot_token() -> str:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if token:
        return token
    try:
        return str(st.secrets.get("TELEGRAM_BOT_TOKEN", "")).strip()
    except Exception:
        return ""


def get_telegram_default_chat_id() -> str:
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if chat_id:
        return chat_id
    try:
        return str(st.secrets.get("TELEGRAM_CHAT_ID", "")).strip()
    except Exception:
        return ""


def send_telegram_message(bot_token: str, chat_id: str, message: str) -> Dict:
    if not bot_token or not chat_id or not message.strip():
        return {"ok": False, "error": "missing_bot_token_or_chat_id"}

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = json.dumps(
        {
            "chat_id": chat_id,
            "text": message,
            "disable_web_page_preview": True,
        }
    ).encode("utf-8")
    request = Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=10) as response:
            body = response.read().decode("utf-8")
        return {"ok": True, "response": body}
    except HTTPError as exc:
        return {"ok": False, "error": f"http_{exc.code}"}
    except URLError as exc:
        return {"ok": False, "error": f"url_error_{exc.reason}"}
    except Exception as exc:
        return {"ok": False, "error": f"unexpected_{type(exc).__name__}"}


def build_ready_signal_message(row, scan_time_label: str) -> str:
    signal_emoji = "🟢" if row["signal"] == "LONG" else "🔴"
    headline = row.get("headline", "No recent headline")
    return (
        f"{signal_emoji} Ready Signal\n"
        f"Symbol: {row['symbol']}\n"
        f"Signal: {row['signal']}\n"
        f"Conviction: {row['conviction']:.1f}\n"
        f"Entry: {row['entry']}\n"
        f"Stop: {row['stop']}\n"
        f"Target 1: {row['target1']}\n"
        f"R:R to T1: {row['rr1']}\n"
        f"Trigger: {row['trigger_text']}\n"
        f"News: {headline}\n"
        f"Scanned: {scan_time_label}"
    )


def send_ready_signal_notifications(
    ready_df,
    bot_token: str,
    chat_id: str,
    scan_time_label: str,
    notified_cache: Dict[str, bool],
) -> List[Dict]:
    outcomes: List[Dict] = []
    if ready_df.empty or not bot_token or not chat_id:
        return outcomes

    active_keys = set()
    for _, row in ready_df.iterrows():
        dedupe_key = f"{row['symbol']}|{row['signal']}|{row['trigger_text']}"
        active_keys.add(dedupe_key)
        if notified_cache.get(dedupe_key):
            continue

        message = build_ready_signal_message(row, scan_time_label=scan_time_label)
        result = send_telegram_message(bot_token, chat_id, message)
        if result.get("ok"):
            notified_cache[dedupe_key] = True
        outcomes.append({"symbol": row["symbol"], "signal": row["signal"], **result})

    stale_keys = [key for key in notified_cache.keys() if key not in active_keys]
    for key in stale_keys:
        notified_cache.pop(key, None)

    return outcomes
