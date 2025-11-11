import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# ---------------------------------------------------------------------
# OPTIONAL: try to load transformers for zero-shot headline classification
# ---------------------------------------------------------------------
USE_TRANSFORMERS = False
try:
    from transformers import pipeline
    zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    candidate_labels = [
        "earnings/results",
        "analyst/ratings",
        "product/company-specific",
        "sector/industry",
        "macro/market",
        "other",
    ]
    USE_TRANSFORMERS = True
except Exception:
    # we'll just use rule-based fallback
    USE_TRANSFORMERS = False

#DATA
#-------------Fetch Stock prices--------------

def get_price_df(ticker: str, days_back: int = 30) -> pd.DataFrame:
    end = datetime.today().date()
    start = end - timedelta(days=days_back)
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df = df.reset_index()
    return df
#Percentage change in stock price
def get_daily_move(df: pd.DataFrame, date_str: str):
    target = pd.to_datetime(date_str)
    day_row = df[df["Date"] == target]

    if day_row.empty:
        return None

    idx = day_row.index[0]
    if idx == 0:
        return None

    close_today = day_row["Close"].iloc[0]
    close_prev = df.iloc[idx - 1]["Close"]
    pct_change = (close_today - close_prev) / close_prev * 100

    return {
        "date": date_str,
        "close": float(close_today),
        "prev_close": float(close_prev),
        "pct_change": round(float(pct_change), 2),
    }
#Combining fetching stock price and percent change functions into one
def get_price_move_for(ticker: str, date_str: str, days_back: int = 30):
    df = get_price_df(ticker, days_back=days_back)
    return get_daily_move(df, date_str)


#----------Fetch News Data------------
# Mocking news data as we don't have a real API
def get_news_for_ticker_mock(ticker: str, date_str: str):
    return [
        {
            "title": f"{ticker} posts stronger-than-expected results",
            "description": f"Investors reacted to {ticker}'s better performance.",
            "source": {"name": "MockWire"},
            "url": "https://example.com/article1",
            "publishedAt": f"{date_str}T14:00:00Z",
        },
        {
            "title": f"Sector peers rally, lifting {ticker}",
            "description": "Broader sector strength supported the stock.",
            "source": {"name": "MockFinance"},
            "url": "https://example.com/article2",
            "publishedAt": f"{date_str}T09:30:00Z",
        },
    ]

#-------------Score articles based on important sources (Rule-based Modeling)
IMPORTANT_SOURCES = {"Reuters", "Bloomberg", "WSJ", "Financial Times", "CNBC"}

def score_article(article, ticker):
    score = 0
    title = (article.get("title") or "").lower()
    desc = (article.get("description") or "").lower()

    if ticker.lower() in title:
        score += 3
    if ticker.lower() in desc:
        score += 1
    if article.get("source", {}).get("name") in IMPORTANT_SOURCES:
        score += 2
    return score
# sorts the articles by importance
def rank_news(articles, ticker):
    return sorted(articles, key=lambda a: score_article(a, ticker), reverse=True)


# Transformer: zero-shot classifier
def classify_headline_zero_shot(text: str):
    if not USE_TRANSFORMERS:
        return "other", 0.0
    out = zero_shot(text, candidate_labels)
    label = out["labels"][0]
    score = float(out["scores"][0])
    return label, score

# --------Rule based explanation

def generate_explanation_structured(ticker: str, move: dict, ranked_articles: list):
    result = {
        "ticker": ticker,
        "date": move["date"] if move else None,
        "price_move": None,
        "primary_driver": None,
        "articles_used": ranked_articles[:3],
        "confidence": 0.4,
        "explanation": "",
    }

    if move is None:
        result["explanation"] = f"Could not find market data for {ticker} on that date."
        return result

    pct = move["pct_change"]
    direction = "up" if pct > 0 else "down"
    pct_abs = abs(pct)

    result["price_move"] = {"pct_change": pct, "direction": direction}
    result["confidence"] += 0.2  # we have price

    date_str = move["date"]

    if not ranked_articles:
        result["explanation"] = (
            f"On {date_str}, {ticker} {'rose' if pct>0 else 'fell'} {pct_abs:.2f}%. "
            f"No major company-specific headlines were found, so the move may reflect broader market or sector factors."
        )
        return result

    top = ranked_articles[0]
    title = top.get("title", "a news report")
    source = top.get("source", {}).get("name", "a financial outlet")
    result["primary_driver"] = title
    result["confidence"] += 0.2  # we have at least one headline

    if ticker.lower() in (title or "").lower():
        result["confidence"] += 0.2

    result["confidence"] = min(result["confidence"], 1.0)

    explanation = (
        f"On {date_str}, {ticker} {'rose' if pct>0 else 'fell'} {pct_abs:.2f}%. "
        f"The move appears to be linked to '{title}' reported by {source}. "
    )
    if len(ranked_articles) > 1:
        explanation += "Additional coverage on the same day may have reinforced investor sentiment."

    result["explanation"] = explanation
    return result


#-------- Transformer-based News Agent

def news_agent_transformer(ticker: str, date_str: str, ranked_articles: list):
    if not ranked_articles:
        return {
            "has_news": False,
            "summary": "No relevant company-specific headlines found.",
            "drivers": [],
            "articles_used": [],
        }

    enriched = []
    for art in ranked_articles[:5]:
        title = art.get("title", "")
        label, prob = classify_headline_zero_shot(title)
        enriched.append(
            {
                "title": title,
                "source": art.get("source", {}).get("name", "unknown"),
                "publishedAt": art.get("publishedAt", ""),
                "predicted_driver": label,
                "driver_confidence": prob,
            }
        )

    lines = [
        f"- {e['title']} → {e['predicted_driver']} ({e['driver_confidence']:.2f})"
        for e in enriched
    ]
    summary = f"Top headlines for {ticker} on {date_str}:\n" + "\n".join(lines)

    return {
        "has_news": True,
        "summary": summary,
        "drivers": enriched,
        "articles_used": ranked_articles[:5],
    }


#--------- Market Agent

def market_agent(ticker: str, move: dict):
    if move is None:
        return {
            "has_data": False,
            "summary": f"No market data for {ticker}.",
            "impact": "unknown",
        }

    pct = move["pct_change"]
    direction = "up" if pct > 0 else "down"
    magnitude = abs(pct)

    if magnitude >= 5:
        impact = "very_large"
    elif magnitude >= 2:
        impact = "notable"
    else:
        impact = "mild"

    return {
        "has_data": True,
        "ticker": ticker,
        "date": move["date"],
        "direction": direction,
        "pct_change": magnitude,
        "impact": impact,
        "summary": f"{ticker} was {direction} {magnitude:.2f}% on {move['date']} ({impact} move).",
    }


# -------ORCHESTRATOR

def finexplain_run(ticker: str, date_str: str):
    move = get_price_move_for(ticker, date_str, days_back=30)
    articles = get_news_for_ticker_mock(ticker, date_str)
    ranked = rank_news(articles, ticker)
    structured = generate_explanation_structured(ticker, move, ranked)
    market_view = market_agent(ticker, move)
    news_view = news_agent_transformer(ticker, date_str, ranked)

    return {
        "ticker": ticker,
        "date": date_str,
        "move": move,
        "market_view": market_view,
        "news_view": news_view,
        "structured": structured,
        "final_note": structured["explanation"],
    }

# ---------------------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------------------
st.set_page_config(page_title="FinExplain – AI Stock Move Explainer", page_icon="📈", layout="wide")
st.title("📈 FinExplain – AI Stock Move Explainer")

st.caption("Explains daily stock moves using market data + classified news. Transformers loaded: "
           + ("✅ yes" if USE_TRANSFORMERS else "⚠️ no (using rule-based only)"))

col = st.sidebar
ticker = col.text_input("Ticker", "AAPL")
date_in = col.date_input("Date", value=datetime.today())
run_btn = col.button("Explain")

if run_btn:
    date_str = date_in.strftime("%Y-%m-%d")
    result = finexplain_run(ticker.upper(), date_str)

    st.subheader("Final Explanation")
    st.write(result["final_note"])

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Price move**")
        mv = result["move"]
        if mv:
            st.metric(label=f"{ticker.upper()} on {mv['date']}", value=f"{mv['pct_change']}%")
        else:
            st.write("No market data.")

    with c2:
        st.markdown("**Confidence**")
        conf = result["structured"]["confidence"]
        st.progress(min(conf, 1.0))
        st.write(f"{conf:.2f}")

    with c3:
        st.markdown("**Primary driver**")
        st.write(result["structured"].get("primary_driver") or "No strong driver detected")

    st.markdown("---")
    st.markdown("### News considered")
    nv = result["news_view"]
    if nv["has_news"]:
        for art in nv["drivers"]:
            st.markdown(
                f"**{art['title']}**  \n"
                f"Source: {art['source']}  \n"
                f"Driver: `{art['predicted_driver']}` ({art['driver_confidence']:.2f})"
            )
    else:
        st.write("No relevant headlines found.")
else:
    st.info("Enter a ticker and date, then click **Explain**.")
