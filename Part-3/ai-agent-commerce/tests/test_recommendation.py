from agents.recommendation_engine import RecommendationEngine
from models import Product


def make_product(title, price, rating, sentiment, source="amazon"):
    return Product(
        title=title,
        link="http://example.com",
        source=source,
        price=price,
        currency="$",
        rating=rating,
        sentiment_score=sentiment,
        sentiment_label="positive",
        reason="test",
    )


def test_budget_filter_and_ordering():
    products = [
        make_product("Cheap Good", 20.0, 4.5, 0.9),
        make_product("Over Budget", 150.0, 5.0, 0.95),
        make_product("Mid Priced", 80.0, 4.0, 0.7),
    ]

    rec = RecommendationEngine()
    ranked = rec.recommend("under $100", products, limit=5)

    # Over-budget item should be filtered out
    titles = [p.title for p in ranked]
    assert "Over Budget" not in titles
    # Cheap Good should outrank Mid Priced
    assert titles[0] == "Cheap Good"
