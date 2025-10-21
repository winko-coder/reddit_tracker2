import praw
import pandas as pd
import yfinance as yf
from textblob import TextBlob
from datetime import datetime
from collections import Counter
import os
import requests
import matplotlib.pyplot as plt
import io
import numpy as np

# === Reddit API Setup ===
reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    user_agent=os.getenv("USER_AGENT")
)

# === Telegram Setup ===
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# === Analyse-Parameter ===
SUBREDDITS = ["wallstreetbets", "stocks", "investing"]
LIMIT = 200
TOP_N = 10
CSV_FILE = "reddit_trends.csv"

# === Hilfsfunktionen ===
def extract_tickers(text):
    import re
    return re.findall(r"\b[A-Z]{2,5}\b", text)

def sentiment_score(text):
    return TextBlob(text).sentiment.polarity

def send_telegram_message(text):
    """Schickt Nachricht über Telegram-Bot."""
    if BOT_TOKEN and CHAT_ID:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}
        try:
            requests.post(url, json=payload, timeout=10)
        except Exception as e:
            print(f"⚠️ Telegram-Fehler: {e}")

def send_telegram_photo(image_bytes, caption=""):
    """Sendet ein Bild über Telegram."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    files = {"photo": ("combined_chart.png", image_bytes)}
    data = {"chat_id": CHAT_ID, "caption": caption}
    try:
        requests.post(url, data=data, files=files, timeout=15)
    except Exception as e:
        print(f"⚠️ Telegram-Foto-Fehler: {e}")

# === Datensammlung von Reddit ===
all_tickers = []
ticker_sentiments = {}

for sub in SUBREDDITS:
    for post in reddit.subreddit(sub).hot(limit=LIMIT):
        tickers = extract_tickers(post.title + " " + post.selftext)
        score = sentiment_score(post.title + " " + post.selftext)
        for t in tickers:
            all_tickers.append(t)
            ticker_sentiments.setdefault(t, []).append(score)

# === Aggregation ===
counts = Counter(all_tickers)
top_tickers = counts.most_common(TOP_N)

# === Kursdaten abrufen ===
data = []
for ticker, mentions in top_tickers:
    avg_sent = round(sum(ticker_sentiments[ticker]) / len(ticker_sentiments[ticker]), 3)
    try:
        current_price = round(yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1], 2)
    except Exception:
        current_price = None

    data.append({
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Ticker": ticker,
        "Mentions": mentions,
        "Sentiment": avg_sent,
        "CurrentPrice": current_price
    })

df_new = pd.DataFrame(data)

# === Alte Daten laden und zusammenführen ===
if os.path.exists(CSV_FILE):
    df_old = pd.read_csv(CSV_FILE)
    df_all = pd.concat([df_old, df_new], ignore_index=True)
    df_all.drop_duplicates(subset=["Timestamp", "Ticker"], inplace=True)
else:
    df_all = df_new

df_all.to_csv(CSV_FILE, index=False)
print(f"✅ CSV aktualisiert ({len(df_new)} neue Einträge).")

# === Asymmetrisches Optionsfenster: Dynamik-Erkennung ===
df_all["Timestamp"] = pd.to_datetime(df_all["Timestamp"])
df_all = df_all.sort_values("Timestamp")

alerts = []
for t in df_new["Ticker"].tolist():
    sub = df_all[df_all["Ticker"] == t].tail(3)
    if len(sub) < 3:
        continue

    m1, m2, m3 = sub["Mentions"].tolist()
    s1, s2, s3 = sub["Sentiment"].tolist()
    p1, p2, p3 = sub["CurrentPrice"].tolist()

    d_mentions = m3 - m2
    d2_mentions = (m3 - m2) - (m2 - m1)
    d_sentiment = s3 - s2
    price_change = (p3 - p2) / p2 if p2 else 0

    if (
        d_mentions > 3 and
        d2_mentions > 0 and
        (d_sentiment > 0.2 or s3 > 0.35) and
        price_change < 0.003
    ):
        alerts.append(t)

if alerts:
    send_telegram_message(
        "ASYMMETRISCHES OPTIONSFENSTER: " + ", ".join(alerts)
    )


# === Telegram-Zusammenfassung ===
top_message = "<b>📊 Aktuelle Reddit-Trends</b>\n\n"
top_message_2 = "\n<b>Aktuell interessante Sentiments</b>:\n"
send_tm2 = False;
for _, row in df_new.head(TOP_N).iterrows():
    line = f"• <b>{row['Ticker']}</b>: {row['Mentions']} Erwähnungen, Sentiment {row['Sentiment']:+.2f}, Kurs {row['CurrentPrice']}\n"
    if(row['Sentiment'] > 0.2 or row['Sentiment'] < -0.2):
        line2 = f"• <b>{row['Ticker']}</b>: {row['Mentions']} Erwähnungen, Sentiment {row['Sentiment']:+.2f}, Kurs {row['CurrentPrice']}\n"
        top_message_2 += line2
        send_tm2 = True;
    top_message += line

top_message += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}"
#send_telegram_message(top_message)
if(send_tm2):
    send_telegram_message(top_message_2)

print("📨 Telegram-Update gesendet!")

try:
    df_all["Timestamp"] = pd.to_datetime(df_all["Timestamp"])
    df_all = df_all.sort_values("Timestamp")
    top_tickers = df_new["Ticker"].tolist()

    plt.figure(figsize=(12, 7))
    colors = ["#0072B2", "#E69F00", "#009E73"]

    for i, t in enumerate(top_tickers):
        subset = df_all[df_all["Ticker"] == t].tail(20)
        if subset.empty:
            continue

        # Kombinierte „Hype-Stärke“ (Erwähnungen * Sentiment)
        subset["HypeStrength"] = subset["Mentions"] * subset["Sentiment"]

        # Linke y-Achse: Kurs
        ax1 = plt.gca()
        ax1.plot(subset["Timestamp"], subset["CurrentPrice"], color=colors[i], label=f"{t} Kurs", linewidth=2)
        ax1.set_ylabel("💰 Kurs (USD)")
        ax1.tick_params(axis="y")

        # Rechte y-Achse: Hype-Stärke
        ax2 = ax1.twinx()
        ax2.plot(subset["Timestamp"], subset["HypeStrength"], color=colors[i], linestyle="--", alpha=0.6, label=f"{t} Hype")
        ax2.set_ylabel("🔥 Hype-Stärke (Mentions × Sentiment)")

    plt.title("📈 Reddit-Hype & Kursentwicklung der Top-Ticker")
    ax1.set_xlabel("Datum")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Gemeinsame Legende aus beiden Achsen
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    plt.tight_layout()

    # Bild an Telegram senden
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    send_telegram_photo(buf, caption="📊 Kurs & Reddit-Hype (Erwähnungen × Sentiment)")
    plt.close()
    print("✅ Kombiniertes Chart gesendet!")

except Exception as e:
    print(f"⚠️ Fehler bei kombiniertem Chart: {e}")
