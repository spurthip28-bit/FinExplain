%%writefile app.py
import streamlit as st
from datetime import date, datetime, timedelta
import yfinance as yf
import pandas as pd

# ============ DATA LAYER ============

def get_price_df(ticker: str, days_back: int = 30) -> pd.DataFrame:
    end = datetime.today().date()
    start = end - timedelta(days=days_back)
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df = df.reset_index()
    return df

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
        "pct_change": round(float(pct_change), 2)
    }

def get_price_move_for(ticker: str, date_str: str, days_back: int = 30):
    df = get_price_df(ticker, days_back=days_back)
    return get_daily_move(df, date_str)

# ============ NEWS (MOCK) ============

def get_news_for_ticker_mock(ticker: str, date_str: str):
    return [
        {
            "title": f"{ticker} posts stronger-than-expected results",
            "description": f"Investors reacted to {ticker}'s better performance.",
            "source": {"name": "MockWire"},
            "url": "https://example.com/article1",
            "publishedAt": f"{date_str}T14:00:00Z"
        },
        {
            "title": f"Sector peers rally, lifting {ticker}",
            "description": "Broader sector strength supported the stock.",
            "source": {"name": "MockFinance"},
            "url": "https://example.com/article2",
            "publishedAt": f"{date_str}T09:30:00Z"
        }
    ]

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

def rank_news(articles, ticker):
    return sorted(articles, key=lambda a: score_article(a, ticker), reverse=True)

# ============ EXPLANATION (STRUCTURED) ============

def generate_explanation_structured(ticker: str, move: dict, ranked_articles: list):
    result = {
        "ticker": ticker,
        "date": move["date"] if move else None,
        "price_move": None,
        "primary_driver": None,
        "articles_used": ranked_articles[:3],
        "confidence": 0.4,
        "explanation": ""
    }

    if move is None:
        result["explanation"] = f"Could not find market data for {ticker} on the selected date."
        return result

    pct = move["pct_change"]
    direction = "up" if pct > 0 else "down"
    pct_abs = abs(pct)

    result["price_move"] = {
        "pct_change": pct,
        "direction": direction
    }
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
    result["confidence"] += 0.2

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

# ============ AGENT VIEWS ============

def market_agent(ticker: str, move: dict):
    if move is None:
        return {
            "has_data": False,
            "summary": f"No market data for {ticker}.",
            "impact": "unknown"
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
        "pct_change": round(magnitude, 2),
        "impact": impact,
        "summary": f"{ticker} was {direction} {magnitude:.2f}% on {move['date']} ({impact} move)."
    }

def news_agent(ticker: str, date_str: str, ranked_articles: list):
    if not ranked_articles:
        return {
            "has_news": False,
            "summary": "No relevant company-specific headlines found.",
            "drivers": [],
            "articles_used": []
        }

    top_articles = ranked_articles[:3]
    summary_lines = [
        f"- {a.get('title')} ({a.get('source', {}).get('name','unknown')})"
        for a in top_articles
    ]
    summary = f"Top headlines for {ticker} on {date_str}:\n" + "\n".join(summary_lines)
    return {
        "has_news": True,
        "summary": summary,
        "drivers": summary_lines,
        "articles_used": top_articles
    }

# ============ ORCHESTRATOR ============

def finexplain_run(ticker: str, date_str: str, llm_client=None):
    move = get_price_move_for(ticker, date_str, days_back=30)
    articles = get_news_for_ticker_mock(ticker, date_str)
    ranked = rank_news(articles, ticker)
    structured = generate_explanation_structured(ticker, move, ranked)
    market_view = market_agent(ticker, move)
    news_view = news_agent(ticker, date_str, ranked)

    # final note = structured explanation (no LLM in this file)
    final_note = structured["explanation"]

    return {
        "ticker": ticker,
        "date": date_str,
        "move": move,
        "market_view": market_view,
        "news_view": news_view,
        "structured": structured,
        "final_note": final_note
    }

# ============ STREAMLIT UI ============

st.set_page_config(page_title="FinExplain – AI Stock Move Explainer", page_icon="📈", layout="wide")
st.title("📈 FinExplain – AI Stock Move Explainer")

col_inputs = st.sidebar
ticker = col_inputs.text_input("Ticker", value="AAPL")
selected_date = col_inputs.date_input("Date", value=date(2025, 11, 3))
run_btn = col_inputs.button("Explain")

if run_btn:
    date_str = selected_date.strftime("%Y-%m-%d")
    result = finexplain_run(ticker.upper(), date_str, llm_client=None)

    st.subheader("Final Explanation")
    st.write(result["final_note"])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Price move**")
        mv = result["move"]
        if mv:
            st.metric(label=f"{ticker.upper()} on {mv['date']}", value=f"{mv['pct_change']}%")
        else:
            st.write("No market data")

    with c2:
        st.markdown("**Confidence**")
        conf = result["structured"]["confidence"]
        st.progress(min(conf, 1.0))
        st.write(f"{conf:.2f}")

    with c3:
        st.markdown("**Primary driver**")
        st.write(result["structured"].get("primary_driver") or "No strong driver")

    st.markdown("---")
    st.markdown("### News considered")
    nv = result["news_view"]
    if nv["has_news"]:
        for art in nv["articles_used"]:
            st.markdown(f"**{art.get('title')}**  \n{art.get('source', {}).get('name','')} — {art.get('publishedAt','')}")
    else:
        st.write("No relevant headlines.")
else:
    st.info("Enter inputs on the left and click **Explain**.")
