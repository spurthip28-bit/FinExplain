# FinExplain 🧠📈
AI-powered stock move explainer.

## What it does
Given a **ticker** (e.g. `AAPL`) and a **date** (e.g. `2025-11-03`), FinExplain:
1. Fetches the price data for that day
2. Calculates the daily % move
3. Fetches (or mocks) news for that day
4. Ranks the news by relevance
5. Generates a plain-English explanation linking the move to the news
6. Returns a confidence score

## Why
Most finance projects stop at “sentiment = positive/negative.”  
FinExplain focuses on **explainability** — “*why did it move?*”

## Features
- Price data via `yfinance`
- Mock news source (can be swapped for NewsAPI / Polygon / FMP)
- Rule-based explanation (no LLM needed)
- Agentic pipeline: MarketAgent, NewsAgent, AnalystAgent
- Streamlit UI to try different tickers/dates
- Batch runner to generate example explanations

## How to run (local / Colab)
```bash
pip install -r requirements.txt
streamlit run app.py
