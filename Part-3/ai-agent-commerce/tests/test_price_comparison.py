import pytest

from agents.price_comparison import PriceComparisonAgent
from models import Product


def make_product(title, price, discount_pct=0.0, source="amazon"):
    return Product(
        title=title,
        link="http://example.com",
        source=source,
        price=price,
        currency="$",
        original_price=price / (1 - discount_pct) if discount_pct else None,
        discount_pct=discount_pct,
        reason="test",
    )


def test_prefers_lower_price():
    items = [
        make_product("Widget A", 30.0, 0.0),
        make_product("Widget A", 20.0, 0.0),
    ]
    result = PriceComparisonAgent().compare(items)
    assert len(result) == 1
    assert result[0].price == 20.0


def test_prefers_higher_discount_when_prices_close():
    items = [
        make_product("Widget B", 25.0, discount_pct=0.10, source="google"),
        make_product("Widget B", 25.0, discount_pct=0.20, source="amazon"),
    ]
    result = PriceComparisonAgent().compare(items)
    assert len(result) == 1
    assert result[0].discount_pct == pytest.approx(0.20)
