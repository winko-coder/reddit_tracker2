# reddit_hype_tracker_filtered.py
# Go Code: Reddit Hype Radar (social + technical filters) -> CSV + Telegram + Chart
# Requirements: praw, textblob, pandas, yfinance, matplotlib, requests, numpy
# Uses environment variables for credentials.

import os
import re
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from collections import Counter, defaultdict
from textblob import TextBlob
import praw

# -------------------------
# Configuration (tweakable)
# -------------------------
SUBREDDITS = [
    "wallstreetbets", "stocks", "investing",
    "pennystocks", "smallstreetbets", "trading", "technology", "artificial",
]
POST_LIMIT = 300            # how many comments/posts per subreddit (per run)
WINDOW_CYCLES = 24          # how many past runs to keep for charts (approx hours)
MIN_MENTIONS_ABS = 15       # absolute mentions in this run to consider
MENTION_GROWTH_FACTOR = 2.5 # current mentions / (recent average + 1) must exceed this
MIN_MARKETCAP = 1_000_000_000  # 1 billion USD (filter out tiny companies)
MIN_AVG_VOLUME = 100_000    # average daily volume filter (may vary by market)
TOP_N_CHART = 6             # how many tickers to plot in combined chart
CSV_FILE = "reddit_trends.csv"
TEMP_FILE = "latest_run.csv"

# technical thresholds
SMA_PERIOD = 50
HIGH_PERIOD = 30

# regex for ticker extraction: 1-5 uppercase letters (filter common words later)
TICKER_REGEX = re.compile(r"\b[A-Z]{1,5}\b")

# common false positives to ignore (extend as needed)
COMMON_WORDS = set([
    "USD","CEO","ETC","ETF","AI","USA","DD","YOLO","AMA","IPO","PM","AM","DDs",
    "BUY","SELL","LONG","SHORT","APR","MAR","JAN","FEB","MAY","JUN","JUL","AUG",
    "SEP","OCT","NOV","DEC","RH","GME","WSB"  # GME included often but keep? adjust
])

# -------------------------
# Environment / Credentials
# -------------------------
# === Reddit API Setup ===
REDDIT_CLIENT_ID=os.getenv("CLIENT_ID"),
REDDIT_CLIENT_SECRET=os.getenv("CLIENT_SECRET"),
REDDIT_USER_AGENT=os.getenv("USER_AGENT")

# === Telegram Setup ===
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

if not (REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET and BOT_TOKEN and CHAT_ID):
    print("ERROR: Missing env vars. Please set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID")
    # Do not exit in Actions (but you may want to raise). We'll stop cleanly.
    # raise SystemExit(1)

# -------------------------
# Helpers
# -------------------------
def send_telegram_message(text):
    """Send simple text message via Telegram (synchronous)."""
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram not configured.")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print("Telegram send error:", e)

def send_telegram_photo(image_bytes, caption=""):
    """Send photo bytes via Telegram."""
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram not configured.")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    files = {"photo": ("chart.png", image_bytes)}
    data = {"chat_id": CHAT_ID, "caption": caption}
    try:
        requests.post(url, data=data, files=files, timeout=20)
    except Exception as e:
        print("Telegram photo error:", e)

def compute_sentiment(text):
    try:
        return TextBlob(text).sentiment.polarity
    except Exception:
        return 0.0

def extract_tickers_from_text(text):
    found = TICKER_REGEX.findall(text)
    # filter out common words and obvious numeric strings
    tickers = [t for t in found if t not in COMMON_WORDS and not t.isdigit() and len(t) <= 5]
    return tickers

# -------------------------
# Reddit data collection
# -------------------------
def collect_reddit_mentions(reddit_client, subreddits, limit=200):
    """Return dict: ticker -> list of sentiments (per mention) and mention count"""
    mentions = defaultdict(list)
    for sub in subreddits:
        try:
            subreddit = reddit_client.subreddit(sub)
            # use newest comments for speed: .comments()
            for comment in subreddit.comments(limit=limit):
                text = (getattr(comment, "body", "") or "") + " " + (getattr(comment, "title", "") or "")
                tickers = extract_tickers_from_text(text)
                if not tickers:
                    continue
                s = compute_sentiment(text)
                for t in tickers:
                    mentions[t].append(s)
        except Exception as e:
            print(f"Warning: subreddit {sub} error: {e}")
    # convert to summary
    summary = {}
    for t, sentiments in mentions.items():
        summary[t] = {
            "mentions": len(sentiments),
            "sentiment_avg": float(np.mean(sentiments)) if sentiments else 0.0
        }
    return summary

# -------------------------
# Market / Technical checks
# -------------------------
def ticker_liquidity_and_info(ticker):
    """Return dict with marketCap, avgVolume, currentPrice, history df, or None on failure"""
    try:
        tk = yf.Ticker(ticker)
        info = tk.info if hasattr(tk, "info") else {}
        # some tickers require suffixes (e.g., .DE) — we try simple one first
        marketCap = info.get("marketCap") or info.get("market_cap") or None
        # avgVolume fallback
        avgVolume = info.get("averageVolume") or info.get("averageVolume10days") or info.get("averageDailyVolume10Day") or info.get("averageVolume10days") or info.get("volume") or 0
        # get history for technicals (60 days)
        hist = tk.history(period="90d", interval="1d")
        if hist is None or hist.empty:
            return None
        current_price = float(hist["Close"].iloc[-1])
        return {
            "marketCap": marketCap,
            "avgVolume": int(avgVolume) if avgVolume is not None else 0,
            "currentPrice": current_price,
            "history": hist
        }
    except Exception as e:
        # print("yfinance error for", ticker, e)
        return None

def technical_signals_from_history(hist_df):
    """Compute sma50 and 30-day high breakout. Returns dict of booleans and values."""
    try:
        close = hist_df["Close"]
        sma50 = close.rolling(window=SMA_PERIOD).mean()
        sma50_last = float(sma50.iloc[-1]) if len(sma50.dropna())>0 else None
        price_last = float(close.iloc[-1])
        high30 = float(close.tail(HIGH_PERIOD).max()) if len(close) >= HIGH_PERIOD else float(close.max())
        # signals
        above_sma50 = (sma50_last is not None) and (price_last > sma50_last)
        breakout_30 = price_last >= high30
        return {
            "price_last": price_last,
            "sma50": sma50_last,
            "above_sma50": above_sma50,
            "high30": high30,
            "breakout_30": breakout_30
        }
    except Exception:
        return {
            "price_last": None,
            "sma50": None,
            "above_sma50": False,
            "high30": None,
            "breakout_30": False
        }

# -------------------------
# Main routine
# -------------------------
def main():
    # init reddit client
    reddit_client = praw.Reddit(
        client_id=os.getenv("CLIENT_ID"),
        client_secret=os.getenv("CLIENT_SECRET"),
        user_agent=os.getenv("USER_AGENT")
    )

    ts_now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print("Run at", ts_now)

    # 1) collect reddit mentions
    reddit_summary = collect_reddit_mentions(reddit_client, SUBREDDITS, limit=POST_LIMIT)
    if not reddit_summary:
        print("No mentions found.")
        return

    # 2) create DataFrame for current run
    rows = []
    for ticker, info in reddit_summary.items():
        rows.append({
            "Timestamp": ts_now,
            "Ticker": ticker,
            "Mentions": info["mentions"],
            "Sentiment": round(info["sentiment_avg"], 4)
        })
    df_run = pd.DataFrame(rows).sort_values("Mentions", ascending=False).reset_index(drop=True)
    if df_run.empty:
        print("No tickers after extraction.")
        return

    # 3) load historical CSV to compute averages for growth factor
    if os.path.exists(CSV_FILE):
        df_hist = pd.read_csv(CSV_FILE)
        # ensure typing
        if "Mentions" in df_hist.columns:
            avg_mentions_hist = df_hist.groupby("Ticker")["Mentions"].rolling(5, min_periods=1).mean().reset_index().groupby("Ticker")["Mentions"].mean()
            # fallback to simple group mean
            avg_mentions_per = df_hist.groupby("Ticker")["Mentions"].mean().to_dict()
        else:
            avg_mentions_per = {}
    else:
        avg_mentions_per = {}

    # 4) evaluate each ticker: liquidity + technical + social filters
    results = []
    for _, r in df_run.iterrows():
        t = r["Ticker"]
        mentions = int(r["Mentions"])
        sentiment = float(r["Sentiment"])

        # quick social filters
        if mentions < MIN_MENTIONS_ABS:
            # too few absolute mentions
            continue

        recent_avg = avg_mentions_per.get(t, 0)
        growth = (mentions) / (recent_avg + 1)
        if growth < MENTION_GROWTH_FACTOR:
            # no strong relative spike
            continue

        # fetch market data / liquidity
        info = ticker_liquidity_and_info(t)
        if info is None:
            # unable to fetch pricing/history
            continue

        marketCap = info.get("marketCap") or 0
        avgVol = info.get("avgVolume") or 0

        # liquidity filter: ensure not a tiny cap
        if marketCap is not None and marketCap < MIN_MARKETCAP:
            # too small
            continue
        if avgVol is not None and avgVol < MIN_AVG_VOLUME:
            # too illiquid
            continue

        # technical signals
        tech = technical_signals_from_history(info["history"])
        # require at least one technical confirmation (above sma50 or breakout)
        if not (tech["above_sma50"] or tech["breakout_30"]):
            continue

        # compute hype strength (mentions * sentiment) - sentiment might be negative, use abs for intensity optionally
        hype_strength = mentions * sentiment

        results.append({
            "Timestamp": ts_now,
            "Ticker": t,
            "Mentions": mentions,
            "Sentiment": sentiment,
            "Price": tech["price_last"],
            "MarketCap": marketCap,
            "AvgVolume": avgVol,
            "Above50DMA": tech["above_sma50"],
            "Breakout30": tech["breakout_30"],
            "HypeStrength": round(hype_strength, 4)
        })

    if not results:
        msg = f"📊 No course-relevant Hypes detected at {ts_now}."
        print(msg)
        send_telegram_message(msg)
        return

    # 5) Convert results to DF and append to CSV
    df_results = pd.DataFrame(results).sort_values("HypeStrength", ascending=False).reset_index(drop=True)
    # Save latest to temp (for GitHub push if desired) and append to main CSV
    df_results.to_csv(TEMP_FILE, index=False)
    if os.path.exists(CSV_FILE):
        df_all = pd.read_csv(CSV_FILE)
        df_combined = pd.concat([df_all, df_results], ignore_index=True)
        df_combined.drop_duplicates(subset=["Timestamp", "Ticker"], inplace=True)
    else:
        df_combined = df_results.copy()
    df_combined.to_csv(CSV_FILE, index=False)

    # 6) Telegram summary (text)
    text = f"🔥 <b>Course-relevant Reddit Hypes</b>\n{ts_now}\n\n"
    for i, row in df_results.head(10).iterrows():
        text += (f"• <b>{row['Ticker']}</b>: {int(row['Mentions'])} mentions, "
                 f"sent {row['Sentiment']:+.2f}, price {row['Price']}, hype {row['HypeStrength']:+.2f}\n")
    text += "\n(Filtered: social spike + liquidity + technical confirm)"
    send_telegram_message(text)

    # 7) Create combined chart (Price + HypeStrength) for top tickers
    try:
        top_for_chart = df_results["Ticker"].head(TOP_N_CHART).tolist()
        plt.figure(figsize=(12,7))
        ax = plt.gca()
        color_iter = iter(plt.cm.tab10.colors)

        for t in top_for_chart:
            # build time series from CSV for this ticker
            hist_t = df_combined[df_combined["Ticker"]==t].sort_values("Timestamp")
            if hist_t.empty:
                continue
            # convert types
            hist_t["Timestamp"] = pd.to_datetime(hist_t["Timestamp"])
            hist_t = hist_t.tail(WINDOW_CYCLES)

            # price plot (left axis)
            price_series = hist_t["Price"].astype(float)
            ax.plot(hist_t["Timestamp"], price_series, label=f"{t} Price", linewidth=2, color=next(color_iter))

            # hype strength on right axis (scale to nicer view)
            ax2 = ax.twinx()
            # scale hype for plot visibility: normalize per ticker
            hype = hist_t["HypeStrength"].astype(float)
            if hype.max() != 0:
                hype_norm = hype / (abs(hype).max()) * (price_series.max() - price_series.min()) * 0.6
            else:
                hype_norm = hype
            ax2.plot(hist_t["Timestamp"], hype_norm, linestyle="--", alpha=0.7, label=f"{t} Hype (scaled)")
            ax2.set_ylabel("Hype (scaled)")

        ax.set_title("Price & HypeStrength (Mentions × Sentiment)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price USD")
        ax.grid(True, linestyle="--", alpha=0.4)
        # combined legend
        lines, labels = ax.get_legend_handles_labels()
        # add second axis legend parts
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper left")
        plt.tight_layout()

        # save to bytes
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        send_telegram_photo(buf, caption="📈 Price & Hype (scaled per ticker)")
        plt.close()
    except Exception as e:
        print("Chart error:", e)

    print("Run finished. Detected:", len(df_results))
    # end main

if __name__ == "__main__":
    main()
