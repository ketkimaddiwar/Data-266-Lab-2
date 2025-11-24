import os
import sys
from typing import List, Optional

from dotenv import load_dotenv

from agents.price_comparison import PriceComparisonAgent
from agents.product_search import ProductSearchAgent
from agents.recommendation_engine import RecommendationEngine
from agents.review_analysis import ReviewAnalysisAgent
from models import Product
from utils import extract_budget


def run(query: str) -> None:
    load_dotenv()

    search_agent = ProductSearchAgent()
    comparison_agent = PriceComparisonAgent()
    review_agent = ReviewAnalysisAgent()
    recommender = RecommendationEngine()

    print("\n[Product Search Agent] Querying Amazon and Google Shopping via SerpAPI...")
    raw_products = search_agent.search(query=query, max_results=10)
    if not raw_products:
        print("No products found for that query.")
        return
    print(f"[Product Search Agent] Completed: retrieved {len(raw_products)} items.")

    raw_source_counts = {}
    for p in raw_products:
        raw_source_counts[p.source] = raw_source_counts.get(p.source, 0) + 1
    print("Raw source counts:", ", ".join(f"{k}={v}" for k, v in raw_source_counts.items()))
    if raw_source_counts.get("google", 0) == 0:
        print("Note: Google Shopping returned no items for this query (could be query phrasing or API plan limits).")

    print(f"[Price Comparison Agent] Found {len(raw_products)} products. De-duping and picking best deals (current + was price where available)...")
    deduped = comparison_agent.compare(raw_products)
    print(f"[Price Comparison Agent] Completed: {len(deduped)} unique deals retained.")

    print("[Review Analysis Agent] Computing sentiment using ratings/reviews and Hugging Face sentiment model...")
    with_sentiment = review_agent.analyze(deduped)
    print("[Review Analysis Agent] Completed sentiment scoring.")

    print("[Recommendation Engine] Scoring with sentiment, rating, price, and budget...")
    recommendations = recommender.recommend(query, with_sentiment, limit=5)

    print("\nSource summary after comparison:")
    source_summary = summarize_sources(with_sentiment)
    for line in source_summary:
        print(f"- {line}")

    budget = extract_budget(query)
    if budget:
        print(f"\nBudget detected: ${budget:.2f}")

    # Per-provider top 5
    print("\nTop 5 per provider:")
    for provider in ("amazon", "google"):
        provider_products = [p for p in with_sentiment if p.source == provider]
        if not provider_products:
            print(f"- {provider}: no items")
            continue
        top_provider = recommender.recommend(query, provider_products, limit=5)
        print(f"- {provider}:")
        print_table(top_provider)

    # Final overall recommendation (single best)
    best_overall = recommender.recommend(query, with_sentiment, limit=1)
    if best_overall:
        best = best_overall[0]
        print("\nFinal recommendation:")
        print_table([best])
        print(f"Justification: {build_justification(best, budget)}")

    print("\nDone.")


def format_product(product: Product) -> str:
    parts: List[str] = [product.title]
    if product.price is not None:
        parts.append(f"Price: {product.currency or '$'}{product.price:.2f}")
        if product.original_price and product.original_price > product.price:
            parts.append(f"(Was {product.currency or '$'}{product.original_price:.2f})")
    if product.rating is not None:
        parts.append(f"Rating: {product.rating:.1f}")
    if product.reviews_count is not None:
        parts.append(f"Reviews: {product.reviews_count}")
    if product.sentiment_score is not None:
        label = product.sentiment_label or "n/a"
        parts.append(f"Sentiment: {label} ({product.sentiment_score:.2f})")
    else:
        parts.append("Sentiment: n/a")
    parts.append(f"Source: {product.source}")
    parts.append(f"Link: {product.link}")
    details = " | ".join(parts)
    justification = f"Why: {product.reason or 'n/a'}"
    return f"{details}\n   {justification}"


def print_table(products: List[Product]) -> None:
    """Render a compact table for a list of products."""
    if not products:
        print("  (no items)")
        return
    headers = ["#", "Title", "Price", "Rating", "Reviews", "Sentiment", "Source"]
    col_widths = [3, 60, 24, 8, 10, 16, 8]
    indent = "  "

    def trunc(text: str, width: int) -> str:
        return text if len(text) <= width else text[: width - 3] + "..."

    header_row = (
        f"{headers[0]:>{col_widths[0]}} "
        f"{headers[1]:<{col_widths[1]}} "
        f"{headers[2]:<{col_widths[2]}} "
        f"{headers[3]:<{col_widths[3]}} "
        f"{headers[4]:<{col_widths[4]}} "
        f"{headers[5]:<{col_widths[5]}} "
        f"{headers[6]:<{col_widths[6]}}"
    )
    separator = "-" * len(header_row)
    print(f"{indent}{header_row}")
    print(f"{indent}{separator}")

    for idx, p in enumerate(products, start=1):
        price = f"{p.currency or '$'}{p.price:.2f}" if p.price is not None else "n/a"
        if p.original_price and p.original_price > (p.price or 0):
            price = f"{price} (was {p.currency or '$'}{p.original_price:.2f})"
        rating = f"{p.rating:.1f}" if p.rating is not None else "n/a"
        reviews = str(p.reviews_count) if p.reviews_count is not None else "n/a"
        sent = "n/a"
        if p.sentiment_score is not None:
            sent = f"{p.sentiment_label or 'n/a'} {p.sentiment_score:.2f}"

        row = (
            f"{idx:>{col_widths[0]}} "
            f"{trunc(p.title, col_widths[1]):<{col_widths[1]}} "
            f"{trunc(price, col_widths[2]):<{col_widths[2]}} "
            f"{rating:<{col_widths[3]}} "
            f"{reviews:<{col_widths[4]}} "
            f"{trunc(sent, col_widths[5]):<{col_widths[5]}} "
            f"{p.source:<{col_widths[6]}}"
        )
        print(f"{indent}{row}")


def build_justification(product: Product, budget: Optional[float]) -> str:
    parts = []
    if product.sentiment_score is not None:
        parts.append(f"high sentiment ({product.sentiment_score:.2f})")
    if product.rating is not None:
        parts.append(f"strong rating ({product.rating:.1f} stars)")
    if product.discount_pct:
        parts.append(f"deal with {product.discount_pct*100:.0f}% off was-price")
    if budget and product.price is not None and product.price <= budget:
        parts.append(f"under budget (${product.price:.2f} vs ${budget:.2f})")
    if not parts:
        return "best available balance of price and quality signals."
    return " | ".join(parts)
def summarize_sources(products: List[Product]) -> List[str]:
    summary = {}
    for p in products:
        entry = summary.setdefault(p.source, {"count": 0, "min_price": None})
        entry["count"] += 1
        if p.price is not None:
            if entry["min_price"] is None or p.price < entry["min_price"]:
                entry["min_price"] = p.price
    lines = []
    for source, data in summary.items():
        min_price = f"${data['min_price']:.2f}" if data["min_price"] is not None else "n/a"
        lines.append(f"{source}: {data['count']} items after dedupe; lowest price {min_price}")
    return lines


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py \"what you are looking for\"")
        sys.exit(1)
    user_query = " ".join(sys.argv[1:]).strip()
    if not user_query:
        print("Please provide a non-empty query in natural language.")
        sys.exit(1)

    try:
        run(user_query)
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}")
        sys.exit(1)
