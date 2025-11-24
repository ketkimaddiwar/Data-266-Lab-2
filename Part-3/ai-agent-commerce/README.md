# Agentic AI E-Commerce Assistant

Autonomous Python agents that search Amazon and Google Shopping (via SerpAPI), compare prices (including “was” prices), compute sentiment (rule-based on ratings/reviews with Hugging Face fallback), and generate personalized recommendations from a single natural-language query.

## Setup
- Python 3.10+ (use `python3` explicitly if `python` is not mapped).
- Create/activate a virtual env:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- Install deps:
  ```bash
  python3 -m pip install -r requirements.txt
  ```
- Add a `.env` (see `.env.example`) with:
  ```bash
  SERPAPI_API_KEY=your_serpapi_key
  HF_TOKEN=your_hugging_face_token
  ```

## Usage
Run a natural language query (no flags):
```bash
python3 main.py "wireless buds under $100 with good positive reviews"
```
If your shell expands `$`, use single quotes or escape it:
```bash
python3 main.py 'wireless buds under $100 with good positive reviews'
# or
python3 main.py "wireless buds under \$100 with good positive reviews"
```

What happens (with per-agent progress logs):
1. **Product Search Agent** hits SerpAPI Amazon + Google Shopping (US/EN), fetching up to 10 results per source.
2. **Price Comparison Agent** deduplicates by canonical title and keeps the best deal using a deal score (current price adjusted by discount_pct when a “was” price is present).
3. **Review Analysis Agent** computes sentiment primarily from rating + reviews (confidence-weighted), it calls Hugging Face router for sentiment analysis. Labels: ≥0.70 positive, ≤0.40 negative, else neutral.
4. **Recommendation Engine** extracts budget from the query, filters out over-budget items, scores by sentiment/rating/affordability, and outputs top 5 per provider plus one final pick with a dynamic justification (sentiment, rating, discount, budget fit).

Outputs are tabular: title, price (and was-price if available), rating, reviews, sentiment (label + score), source, link, and a “Why” trail.

## Scenarios to Validate
- Budget filtering: `python3 main.py "4k monitor under $300 for coding"`
- Specific requirements: `python3 main.py "lightweight gaming mouse with side buttons"`
- Comparative shopping: `python3 main.py "portable bluetooth speaker waterproof good battery life"`

## Notes
- Real external calls (SerpAPI + Hugging Face) require network access and valid tokens.
- Price “history” uses the `old/was` price fields from SerpAPI; we display them and use them to break ties in favor of better deals.
- Sentiment: Hugging Face sentiment on user reviews. Labels: ≥0.70 positive, ≤0.40 negative, else neutral.
- Multiple sources: Amazon + Google Shopping; best deal per canonical title is kept with reasoning in output.
- The urllib3 LibreSSL warning is benign; ignore it or run with `PYTHONWARNINGS="ignore::urllib3.exceptions.NotOpenSSLWarning" python3 main.py "..."`

## Agents and Workflow
- **Product Search Agent** (`agents/product_search.py`): SerpAPI `engine=amazon` and `engine=google_shopping`; normalizes price, was-price, rating, review counts, source.
- **Price Comparison Agent** (`agents/price_comparison.py`): Canonicalizes titles, dedupes across sources, keeps the lowest deal score (price adjusted by discount_pct); reasons note replacements.
- **Review Analysis Agent** (`agents/review_analysis.py`): Hugging Face router (`distilbert-base-uncased-finetuned-sst-2-english`.
- **Recommendation Engine** (`agents/recommendation_engine.py`): Budget extraction, over-budget filtering, scoring by sentiment/rating/affordability, returns top-N and final pick with justification.

## Data Sources / APIs
- SerpAPI Amazon: `engine=amazon`, `k` (query), `amazon_domain=amazon.com`.
- SerpAPI Google Shopping: `engine=google_shopping`, `q` (query), `gl=us`, `hl=en`, `google_domain=google.com`.
- Hugging Face Inference (router): `distilbert-base-uncased-finetuned-sst-2-english`.

## Output Evidence and Reasoning
Each recommendation shows title, price (and was-price), rating, reviews, sentiment label + score, source, link. A “Why” line captures the trail: search source, deal decisions, sentiment basis, budget fit, so each agent’s contribution is visible.

Example (shape varies with live data):
```
Source summary after comparison:
- amazon: 10 items after dedupe; lowest price $8.48
- google: 10 items after dedupe; lowest price $15.99

Top 5 per provider:
  # Title                                   Price (was)   Rating Reviews Sentiment Source
  ...

Final recommendation:
  # Title                                   Price (was)   Rating Reviews Sentiment Source
    Justification: high sentiment | strong rating | deal off was-price | under budget
```

## Tests
Logic-only tests (no network) are included:
```bash
python3 -m pytest
```
They cover price comparison (deal-aware tie-breaking), sentiment analysis, and recommendation ordering/budget filtering.
