import os
from typing import List, Optional

import requests

from models import Product
from utils import parse_price


SERPAPI_ENDPOINT = "https://serpapi.com/search.json"


class ProductSearchAgent:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SERPAPI_API_KEY")
        if not self.api_key:
            raise ValueError("SERPAPI_API_KEY is required for product search.")
        self.session = requests.Session()
        loc = os.getenv("GOOGLE_SHOPPING_LOCATION", "")
        self.location = loc.strip("'\"") if loc else None

    def search(self, query: str, max_results: int = 8) -> List[Product]:
        amazon = self._search_amazon(query, max_results=max_results)
        google = self._search_google_shopping(query, max_results=max_results)
        return amazon + google

    def _fetch(self, params: dict) -> dict:
        resp = self.session.get(SERPAPI_ENDPOINT, params=params, timeout=30)
        if not resp.ok:
            detail = ""
            try:
                detail = resp.json()
            except Exception:  
                detail = resp.text
            raise ValueError(f"SerpAPI error ({resp.status_code}) for {params.get('engine')}: {detail}")
        return resp.json()

    def _search_amazon(self, query: str, max_results: int) -> List[Product]:
        params = {
            "engine": "amazon",
            "api_key": self.api_key,
            "k": query,
            "amazon_domain": "amazon.com",
        }
        payload = self._fetch(params)
        items = payload.get("organic_results", [])[:max_results]
        products: List[Product] = []
        for item in items:
            title = item.get("title")
            link = item.get("link") or item.get("product_link")
            asin = item.get("asin")
            raw_price = None
            original_price_raw = None
            price_field = item.get("price")
            if isinstance(price_field, dict):
                raw_price = price_field.get("raw") or price_field.get("displayed_price")
                original_price_raw = (
                    price_field.get("previous_price")
                    or price_field.get("old_price")
                    or price_field.get("before_price")
                    or price_field.get("regular_price")
                )
            elif isinstance(price_field, str):
                raw_price = price_field

            for key in ("old_price", "previous_price", "before_price", "regular_price"):
                top_val = item.get(key)
                if top_val and not original_price_raw:
                    original_price_raw = top_val.get("raw") if isinstance(top_val, dict) else top_val

            price_value, currency = parse_price(raw_price) if raw_price else (None, None)
            orig_value, _ = parse_price(original_price_raw) if original_price_raw else (None, None)
            rating = None
            reviews_count = None
            if isinstance(item.get("rating"), (int, float)):
                rating = float(item["rating"])
            elif isinstance(item.get("rating"), str):
                try:
                    rating = float(item["rating"])
                except ValueError:
                    rating = None
            reviews_count_raw = (
                item.get("ratings_total")
                or item.get("reviews_count")
                or item.get("reviews")
                or item.get("number_of_reviews")
            )
            if isinstance(reviews_count_raw, str):
                reviews_count_raw = reviews_count_raw.replace(",", "")
            try:
                reviews_count = int(reviews_count_raw)
            except (ValueError, TypeError):
                reviews_count = None
            snippet = item.get("snippet")
            if not title or not link:
                continue
            discount_pct = None
            reason = "Found via Amazon SerpAPI search"
            if price_value is not None and orig_value and orig_value > price_value:
                discount_pct = max(0.0, (orig_value - price_value) / orig_value)
                reason += f"; discount_detected old={orig_value} current={price_value}"
            products.append(
                Product(
                    title=title,
                    link=link,
                    source="amazon",
                    price=price_value,
                    currency=currency,
                    original_price=orig_value,
                    discount_pct=discount_pct,
                    rating=rating,
                    reviews_count=reviews_count,
                    snippet=snippet,
                    asin=asin,
                    reason=reason,
                )
            )
        return products

    def _search_google_shopping(self, query: str, max_results: int) -> List[Product]:
        google_query = self._product_query(query)
        params = {
            "engine": "google_shopping",
            "api_key": self.api_key,
            "q": google_query,
            "gl": "us",
            "hl": "en",
            "google_domain": "google.com",
            "num": max_results,
            "device": "desktop",
        }
        if self.location:
            params["location"] = self.location
        payload = self._fetch(params)
        items = payload.get("shopping_results", [])[:max_results]
        if not items:
            items = payload.get("inline_shopping_results", [])[:max_results]
        products = self._parse_google_items(items, reason="Found via Google Shopping raw query")
        if products:
            return products
        if payload.get("error"):
            print(f"Google Shopping error: {payload.get('error')}")
        return []

    def _parse_google_items(self, items: List[dict], reason: str) -> List[Product]:
        products: List[Product] = []
        for item in items:
            title = item.get("title")
            link = item.get("link") or item.get("product_link") or item.get("serpapi_product_api")
            raw_price = item.get("price")
            original_price_raw = item.get("old_price") or item.get("extracted_old_price")
            price_value, currency = parse_price(raw_price) if raw_price else (None, None)
            orig_value, _ = parse_price(original_price_raw) if original_price_raw else (None, None)
            rating = None
            if isinstance(item.get("rating"), (int, float)):
                rating = float(item["rating"])
            elif isinstance(item.get("rating"), str):
                try:
                    rating = float(item["rating"])
                except ValueError:
                    rating = None
            reviews_count_raw = item.get("reviews")
            if isinstance(reviews_count_raw, str):
                reviews_count_raw = reviews_count_raw.replace(",", "")
            try:
                reviews_count = int(reviews_count_raw)
            except (ValueError, TypeError):
                reviews_count = None
            snippet = item.get("snippet") or item.get("description")
            if not title or not link:
                continue
            discount_pct = None
            reason_with_discount = reason
            if price_value is not None and orig_value and orig_value > price_value:
                discount_pct = max(0.0, (orig_value - price_value) / orig_value)
                reason_with_discount += f"; discount_detected old={orig_value} current={price_value}"
            products.append(
                Product(
                    title=title,
                    link=link,
                    source="google",
                    price=price_value,
                    currency=currency,
                    original_price=orig_value,
                    discount_pct=discount_pct,
                    rating=rating,
                    reviews_count=reviews_count,
                    snippet=snippet,
                    reason=reason_with_discount,
                )
            )
        return products

    def _simplify_query(self, query: str) -> str:
        return query

    def _product_query(self, query: str) -> str:
        """Pass the user query through unchanged for Google Shopping."""
        return query.strip()
