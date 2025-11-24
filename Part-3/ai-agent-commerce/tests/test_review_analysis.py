import os

from agents.review_analysis import ReviewAnalysisAgent
from models import Product


def test_rule_based_with_reviews(monkeypatch):
    os.environ["HF_TOKEN"] = "dummy"
    agent = ReviewAnalysisAgent(hf_token="dummy")
    product = Product(
        title="Test Product",
        link="http://example.com",
        source="amazon",
        price=10.0,
        currency="$",
        rating=4.5,
        reviews_count=1000,
        reason="test",
    )
    analyzed = agent.analyze([product])[0]
    assert analyzed.sentiment_score is not None
    # High review count should keep score near rating/5 = 0.9
    assert analyzed.sentiment_score > 0.8
    assert analyzed.sentiment_label == "positive"


def test_hf_fallback(monkeypatch):
    os.environ["HF_TOKEN"] = "dummy"
    agent = ReviewAnalysisAgent(hf_token="dummy")

    # Mock HF call to avoid network
    monkeypatch.setattr(agent, "_analyze_text", lambda text: 0.8)

    product = Product(
        title="Test Product",
        link="http://example.com",
        source="amazon",
        price=10.0,
        currency="$",
        rating=None,
        reviews_count=None,
        reason="test",
    )
    analyzed = agent.analyze([product])[0]
    assert analyzed.sentiment_score == 0.8
    assert analyzed.sentiment_label == "positive"
