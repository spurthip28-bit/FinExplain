# imports
import streamlit as st
import yfinance as yf
import pandas as pd
from transformers import pipeline


# ML/NLP setup
zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
candidate_labels = ["earnings/results", "sector/industry", "macro/market", "analyst/ratings", "product/company-specific", "other"]

def classify_headline(text):
    out = zero_shot(text, candidate_labels)
    return out["labels"][0], float(out["scores"][0])


def finexplain_run(ticker: str, date_str: str, llm_client=None):
    # price
    move = get_price_move_for(ticker, date_str, days_back=30)

    # news
    articles = get_news_for_ticker_mock(ticker, date_str)
    ranked = rank_news(articles, ticker)

    # structured draft (rule-based)
    structured = generate_explanation_structured(ticker, move, ranked)

    # build LLM prompt from structured
    prompt = llm_prompt_structured(structured)

    # refine with LLM
    if llm_client is not None:
        llm_refined = refine_with_llm(prompt, client=llm_client)
    else:
        llm_refined = structured["explanation"]  # fall back to rule-based

    # agent views (market + news)
    market_view = market_agent(ticker, move)
    news_view = news_agent_transformer(ticker, date_str, ranked)
  
    final_note = llm_refined

    return {
        "ticker": ticker,
        "date": date_str,
        "move": move,
        "market_view": market_view,
        "news_view": news_view,
        "structured": structured,
        "prompt_sent": prompt,
        "final_note": final_note
    }

# STREAMLIT UI
st.title("FinExplain – AI Stock Move Explainer")
ticker = st.sidebar.text_input("Ticker", "AAPL")
date = st.sidebar.date_input("Date")
if st.sidebar.button("Explain"):
    result = finexplain_run(ticker, date.strftime("%Y-%m-%d"))
    st.write(result["final_note"])
    # show news, confidence, etc.
