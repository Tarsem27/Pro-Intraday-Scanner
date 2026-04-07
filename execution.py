from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict
import uuid
from zoneinfo import ZoneInfo

import pandas as pd

MELBOURNE_TZ = ZoneInfo("Australia/Melbourne")


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None

@dataclass
class OrderTicket:
    ticket_id: str
    created_at: str
    broker_mode: str
    symbol: str
    side: str
    order_type: str
    time_in_force: str
    quantity: float
    reference_price: float | None
    limit_price: float | None
    stop_loss: float | None
    take_profit: float | None
    signal: str
    conviction: float | None
    notes: str
    risk_per_unit: float | None
    estimated_total_risk: float | None


def build_order_ticket(
    selected,
    broker_mode: str,
    quantity: float,
    order_type: str,
    time_in_force: str,
    limit_price: float | None = None,
    notes: str = "",
) -> OrderTicket:
    side = "BUY" if selected["signal"] == "LONG" else "SELL"
    reference_price = _safe_float(selected.get("price"))
    planned_entry = _safe_float(selected.get("entry"))
    stop_loss = _safe_float(selected.get("stop"))
    take_profit = _safe_float(selected.get("target1"))

    normalized_order_type = order_type.upper()
    if normalized_order_type == "LIMIT":
        chosen_limit = limit_price if limit_price is not None else planned_entry
    else:
        chosen_limit = None

    entry_for_risk = chosen_limit if chosen_limit is not None else planned_entry
    risk_per_unit = None
    if entry_for_risk is not None and stop_loss is not None:
        risk_per_unit = abs(entry_for_risk - stop_loss)

    estimated_total_risk = risk_per_unit * quantity if risk_per_unit is not None else None

    return OrderTicket(
        ticket_id=str(uuid.uuid4())[:8],
        created_at=datetime.now(MELBOURNE_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"),
        broker_mode=broker_mode,
        symbol=str(selected["symbol"]),
        side=side,
        order_type=normalized_order_type,
        time_in_force=time_in_force.upper(),
        quantity=float(quantity),
        reference_price=reference_price,
        limit_price=chosen_limit,
        stop_loss=stop_loss,
        take_profit=take_profit,
        signal=str(selected["signal"]),
        conviction=_safe_float(selected.get("conviction")),
        notes=notes.strip(),
        risk_per_unit=risk_per_unit,
        estimated_total_risk=estimated_total_risk,
    )


class ManualConfirmationBroker:
    name = "plus500_manual_confirmation"

    def stage_order(self, ticket: OrderTicket) -> Dict[str, Any]:
        payload = asdict(ticket)
        payload["status"] = "STAGED_FOR_MANUAL_CONFIRMATION"
        payload["instructions"] = [
            "Open the matching instrument in your Plus500 demo account.",
            "Verify quantity, direction, and whether Plus500 expects units, lots, or contracts for that instrument.",
            "Set stop-loss and take-profit manually after reviewing the staged ticket.",
            "Only submit if the live Plus500 bid/ask and spread still match your plan.",
        ]
        return payload


class Plus500DirectPlaceholderBroker:
    name = "plus500_direct_placeholder"

    def stage_order(self, ticket: OrderTicket) -> Dict[str, Any]:
        payload = asdict(ticket)
        payload["status"] = "DIRECT_API_NOT_CONFIGURED"
        payload["instructions"] = [
            "No direct retail Plus500 API is configured in this app.",
            "Use manual confirmation mode unless you have a supported Plus500 Futures/T4/FIX integration.",
        ]
        return payload


def get_broker_adapter(mode: str):
    if mode == "Plus500 manual confirmation":
        return ManualConfirmationBroker()
    return Plus500DirectPlaceholderBroker()


def ticket_to_frame(ticket_payload: Dict[str, Any]) -> pd.DataFrame:
    keys = [
        "ticket_id",
        "created_at",
        "broker_mode",
        "symbol",
        "side",
        "order_type",
        "time_in_force",
        "quantity",
        "reference_price",
        "limit_price",
        "stop_loss",
        "take_profit",
        "signal",
        "conviction",
        "risk_per_unit",
        "estimated_total_risk",
        "status",
    ]
    row = {key: ticket_payload.get(key) for key in keys}
    return pd.DataFrame([row])
