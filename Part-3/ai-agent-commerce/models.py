from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Product:
    title: str
    link: str
    source: str  # "amazon" or "google"
    price: Optional[float]
    currency: Optional[str]
    original_price: Optional[float] = None  
    discount_pct: Optional[float] = None 
    rating: Optional[float] = None
    reviews_count: Optional[int] = None
    snippet: Optional[str] = None
    sentiment_score: Optional[float] = None  
    sentiment_label: Optional[str] = None  # "positive" | "neutral" | "negative"
    asin: Optional[str] = None  
    review_texts: List[str] = field(default_factory=list)
    reason: str = field(default_factory=str)  