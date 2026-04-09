
# ==============================
# file: config.py
# ==============================
from typing import Dict, List

# Yahoo Finance: single request per symbol (no sidebar tuning). Intraday must stay within provider limits.
SCAN_PERIOD = "5d"
SCAN_INTERVAL = "5m"
INCLUDE_PREPOST_DEFAULT = True

# Cross-symbol "recent READY" table: each symbol runs a full bar-by-bar timeline (CPU-heavy).
RECENT_READY_MAX_SYMBOLS = 12

DEFAULT_UNIVERSES: Dict[str, List[str]] = {
    "US Mega Caps": [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD", "NFLX", "PLTR",
        "AVGO", "SMCI", "MU", "CRM", "ORCL", "INTC", "QCOM", "UBER", "SHOP", "COIN",
    ],
    "US Momentum / High Beta": [
        "TSLA", "NVDA", "AMD", "PLTR", "SMCI", "COIN", "MSTR", "RIVN", "SOFI", "AFRM",
        "RIOT", "MARA", "IONQ", "ARM", "CAVA", "HIMS", "HOOD", "RBLX", "NET", "SNOW",
        "UPST", "ASTS", "CELH", "RKLB", "TEM",
    ],
    "Small / Mid Caps": [
        "IONQ", "RKLB", "SOFI", "AFRM", "UPST", "RBLX", "HIMS", "PLUG", "RUN", "RIOT",
        "MARA", "LMND", "OPEN", "CHPT", "JOBY", "DNA", "QS", "ACHR", "SOUN", "BBAI",
    ],
    "Penny / Low Price Stocks": [
        "SIRI", "OPEN", "KGC", "PLUG", "MVIS", "RIG", "GRPN", "BB", "TLRY", "MULN",
        "NKLA", "SNDL", "CIFR", "WULF", "CLSK", "MARA", "RIOT", "BNGO", "OCGN", "APLD",
    ],
    "ETFs": ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLE", "SMH", "ARKK", "SOXX"],
    "Indices": ["^GSPC", "^IXIC", "^DJI", "^RUT", "^VIX", "^AXJO"],
    "Forex": ["EURUSD=X", "GBPUSD=X", "AUDUSD=X", "USDJPY=X", "USDCHF=X", "USDCAD=X", "NZDUSD=X"],
    "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "BNB-USD"],
}

DEFAULT_UNIVERSES["ALL"] = list(
    dict.fromkeys(
        symbol
        for universe_name, universe_symbols in DEFAULT_UNIVERSES.items()
        if universe_name != "ALL"
        for symbol in universe_symbols
    )
)

DEFAULT_CUSTOM_SYMBOLS = "TSLA,NVDA,AMD,PLTR,SMCI,COIN,MSTR,RIVN,SOFI,AFRM,RIOT,MARA,IONQ,RKLB,SIRI,PLUG,OPEN,SPY,QQQ,IWM,BTC-USD,ETH-USD"

