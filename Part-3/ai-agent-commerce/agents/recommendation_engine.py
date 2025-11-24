from typing import List, Optional

from models import Product
from utils import extract_budget


class RecommendationEngine:
    def recommend(self, query: str, products: List[Product], limit: int = 5) -> List[Product]:
        budget = extract_budget(query)
        filtered: List[Product] = []
        for product in products:
            if budget and product.price and product.price > budget:
                product.reason += "; filtered_out_over_budget"
                continue
            filtered.append(product)

        ranked = sorted(filtered, key=lambda p: self._score(p, budget), reverse=True)
        return ranked[:limit]

    def _score(self, product: Product, budget: Optional[float]) -> float:
        sentiment = product.sentiment_score if product.sentiment_score is not None else 0.5
        rating = (product.rating / 5.0) if product.rating else 0.0
        if product.price and budget:
            price_affordability = max(0.0, 1.0 - (product.price / budget))
        elif product.price:
            price_affordability = 0.5
        else:
            price_affordability = 0.3

        # Simple weighted blend.
        return 0.5 * sentiment + 0.3 * rating + 0.2 * price_affordability
