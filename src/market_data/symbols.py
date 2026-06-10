def canonicalize_symbol(symbol: str) -> str:
    return str(symbol).strip().upper()


def detect_market(symbol: str) -> str:
    return 'us'


def to_yfinance_symbol(symbol: str) -> str:
    return canonicalize_symbol(symbol)
