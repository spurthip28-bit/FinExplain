# FinExplain 🧠📈
FinExplain is an AI-driven financial insight system that explains why a stock moved by integrating market data and financial news through a combination of rule-based analytics and transformer-based natural language reasoning.

It demonstrates how GenAI and analytical frameworks can augment financial analysts — turning raw market data into structured, interpretable insights.



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
- Market + News Fusion: Combines Yahoo Finance price data with daily news headlines to analyze the drivers behind stock movements.
- Agentic Reasoning Pipeline: • MarketAgent – Quantifies the magnitude and direction of a price move.
- NewsAgent – Ranks and classifies headlines using a zero-shot transformer (BART-MNLI) and FinBERT sentiment model.
- AnalystAgent – Synthesizes findings into concise, human-readable explanations with confidence scoring.
- Transformer-based NLP: Zero-shot headline classification into categories like Earnings, Sector, Macro, and Analyst.
- FinBERT sentiment analysis for finance-specific polarity detection.
- Explainability Framework: Structured JSON output including driver attribution, sentiment, and confidence.
- Interactive Dashboard: Streamlit UI to select ticker/date, visualize stock movement, classified headlines, and AI-generated explanations — runnable directly in Google Colab via ngrok.

## How to run (local / Colab)
```bash
pip install -r requirements.txt
streamlit run Finexplainstream.py
