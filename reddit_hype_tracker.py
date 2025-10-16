import praw, re, pandas as pd, requests, os, datetime
from textblob import TextBlob

# === Reddit Credentials aus GitHub Secrets ===
reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    user_agent=os.getenv("USER_AGENT")
)

# === Telegram Bot ===
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# === Parameter ===
SUBREDDITS = ["wallstreetbets", "stocks", "investing"]
POST_LIMIT = 300
HISTORY_FILE = "reddit_trends.csv"
TOP_N = 10


def get_tickers():
    """Sammelt Erwähnungen und Sentiments von Subreddits."""
    data = []
    for sub in SUBREDDITS:
        for comment in reddit.subreddit(sub).comments(limit=POST_LIMIT):
            text = comment.body
            found = re.findall(r"\b[A-Z]{2,5}\b", text)
            for t in found:
                if t not in ["USD","CEO","ETF","AI","USA","GDP","EPS","IPO"]:
                    sentiment = TextBlob(text).sentiment.polarity
                    data.append((t, sentiment))
    return pd.DataFrame(data, columns=["ticker","sentiment"])


def summarize(df):
    """Fasst Erwähnungen und Sentiment je Ticker zusammen."""
    if df.empty:
        return pd.DataFrame(columns=["ticker","mentions","sentiment"])
    grouped = df.groupby("ticker").agg(
        mentions=("ticker","count"),
        sentiment=("sentiment","mean")
    ).reset_index().sort_values(by="mentions", ascending=False)
    return grouped


def load_history():
    """Lädt bisherigen Trendverlauf (falls vorhanden)."""
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    else:
        return pd.DataFrame(columns=["timestamp", "ticker", "mentions"])


def save_cycle(df):
    """Speichert aktuelle Erwähnungen mit Zeitstempel."""
    ts = datetime.datetime.utcnow().isoformat()
    df = df[["ticker", "mentions"]].copy()
    df["timestamp"] = ts
    df.to_csv(HISTORY_FILE, mode="a", header=not os.path.exists(HISTORY_FILE), index=False)


def compute_trends():
    """Analysiert Trends über die Zeit."""
    hist = load_history()
    if hist.empty:
        return pd.DataFrame()
    trend = hist.groupby(["ticker"]).agg(
        avg_mentions=("mentions", "mean"),
        last_mentions=("mentions", "last"),
        count=("mentions", "count")
    ).reset_index()
    trend["trend_strength"] = (trend["last_mentions"] - trend["avg_mentions"]) / (trend["avg_mentions"] + 1)
    trend = trend.sort_values("trend_strength", ascending=False)
    return trend


def send_telegram(msg):
    """Sendet Telegram-Nachricht."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    requests.post(url, data=data)


def main():
    df = get_tickers()
    summary = summarize(df)
    top = summary.head(TOP_N)
    save_cycle(top)
    trend_df = compute_trends().head(5)

    msg = "📈 *Reddit Trending Stocks*\n\n"
    for _, row in top.iterrows():
        msg += f"{row['ticker']}: {row['mentions']} Erwähnungen, Sentiment {row['sentiment']:.2f}\n"

    if not trend_df.empty:
        msg += "\n🔥 *Langfristige Trends:*\n"
        for _, row in trend_df.iterrows():
            msg += f"{row['ticker']}: Trend {row['trend_strength']:+.2f}\n"

    send_telegram(msg)


if __name__ == "__main__":
    main()
