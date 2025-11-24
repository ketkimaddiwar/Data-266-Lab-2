import re
from typing import Dict, List

from models import Product


def _canonical_key(title: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9 ]+", "", title.lower())
    tokens = cleaned.split()
    return " ".join(tokens[:8])


class PriceComparisonAgent:
    def _deal_score(self, product: Product) -> float:
        """
        Lower is better. Incorporates price and discount if available.
        """
        if product.price is None:
            return float("inf")
        discount = product.discount_pct or 0.0
        discount = min(max(discount, 0.0), 0.8)  
        return product.price * (1 - 0.5 * discount)

    def compare(self, products: List[Product]) -> List[Product]:
        """
        Deduplicate similar items and keep the best priced option per canonical title.
        If price is missing, keep the first seen. We also consider discount to
        identify stronger deals when prices tie or are close.
        """
        best: Dict[str, Product] = {}
        for product in products:
            key = _canonical_key(product.title)
            if key not in best:
                product.reason += "; initial candidate for comparison"
                best[key] = product
                continue
            current = best[key]
            if product.price is None:
                continue
            current_score = self._deal_score(current)
            new_score = self._deal_score(product)
            replace = False
            if new_score < current_score:
                replace = True
            else:
                if abs(new_score - current_score) <= current_score * 0.02:
                    current_discount = current.discount_pct or 0.0
                    new_discount = product.discount_pct or 0.0
                    if new_discount > current_discount:
                        replace = True
            if replace:
                product.reason += (
                    f"; replaced {current.source} at {current.price} "
                    f"(deal_score {new_score:.2f} vs {current_score:.2f})"
                )
                best[key] = product
        return list(best.values())
