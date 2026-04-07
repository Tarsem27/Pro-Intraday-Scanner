
# ==============================
# file: config.py
# ==============================
from typing import Dict, List

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
    "ETFs": ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLE", "SMH", "ARKK", "SOXX"],
    "Indices": ["^GSPC", "^IXIC", "^DJI", "^RUT", "^VIX", "^AXJO"],
    "Forex": ["EURUSD=X", "GBPUSD=X", "AUDUSD=X", "USDJPY=X", "USDCHF=X", "USDCAD=X", "NZDUSD=X"],
    "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "BNB-USD"],
}

DEFAULT_CUSTOM_SYMBOLS = "TSLA,NVDA,AMD,PLTR,SMCI,COIN,MSTR,RIVN,SOFI,AFRM,RIOT,MARA,IONQ,ARM,CAVA,HIMS,HOOD,RBLX,NET,SNOW,SPY,QQQ,IWM,BTC-USD,ETH-USD"

