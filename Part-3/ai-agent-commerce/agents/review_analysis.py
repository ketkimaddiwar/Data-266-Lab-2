import os
import math
from typing import List, Optional

import time
from typing import List, Optional

import requests

from models import Product


class ReviewAnalysisAgent:
    def __init__(
        self,
        hf_token: Optional[str] = None,
        model: str = "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    ):
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise ValueError("HF_TOKEN is required for review analysis.")
        # Use accessible, public models on HF router.
        self.model = model
        self.backup_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.session = requests.Session()
        # hf-inference path is required on router.
        self.router_base = "https://router.huggingface.co/hf-inference"

    def analyze(self, products: List[Product]) -> List[Product]:
        for product in products:
            if product.rating is not None and product.reviews_count is not None:
                base = min(max(product.rating / 5.0, 0.0), 1.0)
                weight = min(1.0, math.log10(product.reviews_count + 1) / 3.0)
                score = 0.5 + (base - 0.5) * weight  
                product.sentiment_score = score
                product.sentiment_label = self._label_from_score(score)
                product.reason += (
                    f"; sentiment_from_rating_and_reviews rating={product.rating:.1f} "
                    f"reviews={product.reviews_count} weight={weight:.2f} "
                    f"score={score:.2f} label={product.sentiment_label}"
                )
                continue

            # Otherwise, try HF on the available text signals.
            text = self._build_text(product)
            if text:
                try:
                    hf_score = self._analyze_text(text)
                    product.sentiment_score = hf_score
                    product.sentiment_label = self._label_from_score(hf_score)
                    product.reason += (
                        f"; sentiment_hf_only={hf_score:.2f} label={product.sentiment_label}"
                    )
                    continue
                except Exception as exc:  
                    product.reason += f"; sentiment_hf_failed={exc}"

            if product.rating is not None:
                rating_score = min(max(product.rating / 5.0, 0.0), 1.0)
                product.sentiment_score = rating_score
                product.sentiment_label = self._label_from_score(rating_score)
                product.reason += (
                    f"; sentiment_rating_only={rating_score:.2f} label={product.sentiment_label}"
                )
            else:
                product.sentiment_score = None
                product.sentiment_label = None
                product.reason += "; sentiment_unavailable"
        return products

    def _build_text(self, product: Product) -> Optional[str]:
        """
        Build a text payload for HF sentiment. Even if we lack reviews,
        combine title/snippet with rating context so HF has signal.
        """
        pieces = []
        if product.title:
            pieces.append(product.title)
        if product.snippet:
            pieces.append(product.snippet)
        if product.rating is not None:
            pieces.append(f"User rating {product.rating:.1f} out of 5")
        if not pieces:
            return None
        return ". ".join(pieces)

    def _analyze_text(self, text: str) -> float:
        try:
            return self._call_router(self.model, text, star_model=False)
        except Exception as primary_error: 
            try:
                return self._call_router(self.backup_model, text, star_model=True)
            except Exception as backup_error: 
                raise ValueError(f"HF failed: primary={primary_error}, backup={backup_error}") from backup_error

    def _call_router(self, model: str, text: str, star_model: bool = False) -> float:
        attempts = 2
        last_err = None
        url = f"{self.router_base}/models/{model}"
        payload = {"inputs": self._truncate(text)}
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        params = None

        for attempt in range(attempts):
            resp = self.session.post(url, headers=headers, params=params, json=payload, timeout=45)
            if resp.status_code in {410, 429, 503} and attempt + 1 < attempts:
                time.sleep(5)
                last_err = resp.text
                continue
            if not resp.ok:
                last_err = resp.text
                resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and "error" in data:
                last_err = data["error"]
                if attempt + 1 < attempts:
                    time.sleep(3)
                    continue
                raise ValueError(data["error"])
            if not isinstance(data, list) or not data:
                last_err = "Empty or unexpected HF response"
                continue
            result = data[0]
            if isinstance(result, list):
                result = result[0]
            label = result.get("label")
            score = result.get("score")
            if label is None or score is None:
                last_err = "Missing label/score"
                continue

            if star_model:
                try:
                    stars = int(str(label).split()[0])
                    return min(max((stars - 1) / 4.0, 0.0), 1.0)
                except Exception as exc:  
                    last_err = exc
                    continue

            upper_label = str(label).upper()
            if "POSITIVE" in upper_label or upper_label.endswith("2"):
                return float(score)
            if "NEGATIVE" in upper_label or upper_label.endswith("0"):
                return 1.0 - float(score)
            if "NEUTRAL" in upper_label or upper_label.endswith("1"):
                return 0.5

            last_err = f"Unexpected label: {label}"
        raise ValueError(last_err or "HF sentiment analysis failed")

    def _truncate(self, text: str, max_chars: int = 600) -> str:
        return text if len(text) <= max_chars else text[: max_chars - 3] + "..."

    def _label_from_score(self, score: float) -> str:
        if score >= 0.70:
            return "positive"
        if score <= 0.40:
            return "negative"
        return "neutral"
