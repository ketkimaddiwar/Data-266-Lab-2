import re
from typing import Optional, Tuple


def parse_price(raw: str) -> Tuple[Optional[float], Optional[str]]:
    if not raw:
        return None, None

    currency_match = re.match(r"([€$£]|[A-Z]{3})", raw.strip())
    currency = currency_match.group(1) if currency_match else None

    # Replace commas used as thousand separators; keep decimal.
    normalized = raw.replace(",", "")
    price_match = re.search(r"(\d+(\.\d+)?)", normalized)
    if not price_match:
        return None, currency

    try:
        return float(price_match.group(1)), currency
    except ValueError:
        return None, currency


def extract_budget(text: str) -> Optional[float]:
    """Pull a budget like $200 or under 150 from free text."""
    if not text:
        return None
    dollar_match = re.search(r"\$?\s?(\d{2,5})(?:\s?(?:usd|dollars))?", text, re.IGNORECASE)
    if dollar_match:
        try:
            return float(dollar_match.group(1))
        except ValueError:
            return None
    return None
