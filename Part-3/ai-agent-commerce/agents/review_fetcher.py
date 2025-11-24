import os
from typing import List, Optional, Tuple

import requests

from models import Product

SERPAPI_ENDPOINT = "https://serpapi.com/search.json"


class ReviewFetcherAgent:
    def __init__(self, api_key: Optional[str] = None, max_reviews: int = 3):
        self.api_key = api_key or os.getenv("SERPAPI_API_KEY")
        if not self.api_key:
            raise ValueError("SERPAPI_API_KEY is required for review fetching.")
        self.session = requests.Session()
        self.max_reviews = max_reviews

    def fetch(self, products: List[Product]) -> List[Product]:
        for product in products:
            if product.source != "amazon" or not product.asin:
                product.reason += "; reviews_skipped_non_amazon"
                continue
            try:
                reviews, avg_rating = self._fetch_amazon_reviews(product.asin)
                product.review_texts = reviews[: self.max_reviews]
                if avg_rating is not None:
                    product.rating = product.rating or avg_rating
                if product.reviews_count is None:
                    product.reviews_count = len(reviews)
                product.reason += f"; reviews_fetched={len(product.review_texts)}"
            except Exception as exc:  
                product.reason += f"; reviews_failed={exc}"
        return products

    def _fetch_amazon_reviews(self, asin: str) -> Tuple[List[str], Optional[float]]:
        engines = ["amazon_reviews", "amazon_product_reviews"]
        last_error = None
        for engine in engines:
            params = {
                "engine": engine,
                "api_key": self.api_key,
                "amazon_domain": "amazon.com",
                "asin": asin,
            }
            resp = self.session.get(SERPAPI_ENDPOINT, params=params, timeout=30)
            if not resp.ok:
                try:
                    detail = resp.json()
                except Exception:  
                    detail = resp.text
                last_error = ValueError(f"SerpAPI error ({resp.status_code}) for {engine}: {detail}")
                continue
            data = resp.json()
            reviews_payload = data.get("reviews", [])
            texts: List[str] = []
            ratings: List[float] = []
            for review in reviews_payload:
                body = review.get("body")
                title = review.get("title")
                rating_raw = review.get("rating")
                if isinstance(rating_raw, (int, float)):
                    ratings.append(float(rating_raw))
                elif isinstance(rating_raw, str):
                    try:
                        ratings.append(float(rating_raw))
                    except ValueError:
                        pass
                if body:
                    texts.append(body)
                elif title:
                    texts.append(title)
                if len(texts) >= self.max_reviews:
                    break
            avg_rating = sum(ratings) / len(ratings) if ratings else None
            if texts or avg_rating is not None:
                return texts, avg_rating
            last_error = ValueError(f"No reviews returned via {engine}")
        if last_error:
            raise last_error
        raise ValueError("Unknown review fetch failure")
