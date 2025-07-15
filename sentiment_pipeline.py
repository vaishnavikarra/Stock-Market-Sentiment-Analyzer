import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
from datetime import datetime, timedelta
import praw
import feedparser
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.graph_objects as go
from bs4 import BeautifulSoup
from email.utils import parsedate_to_datetime


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
analyzer = SentimentIntensityAnalyzer()

reddit = praw.Reddit(
    client_id="0iWxFubAdoZ_MVt98OeE5w",
    client_secret="twUcZatP4QYppYiNh-o4mkp3gIHgTQ",
    user_agent="stock_sentiment_analyzer by u/Ordinary-Hunt-8330"
)



junk_words = {
    "com", "www", "http", "https", "font", "targetblank", "nbsp",
    "barchartcom", "marketscreener", "nasdaqfont", "color6f6f6f",
    "org", "net", "inc", "amp", "co", "said", "u", "rt", "som"
}

def clean_text(text):
    # Remove HTML tags
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove non-alphanumeric
    text = re.sub(r"[^A-Za-z0-9 ]+", " ", text)

    # Tokenize and filter
    tokens = text.lower().split()
    tokens = [
        word for word in tokens
        if word not in stop_words and word not in junk_words and len(word) > 2
    ]
    return " ".join(tokens)


def get_sentiment_label(text):
    score = analyzer.polarity_scores(text)
    if score["compound"] >= 0.05:
        return 'positive'
    elif score["compound"] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

def fetch_reddit_posts(company_name, subreddit_name="stocks"):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for submission in subreddit.new(limit=100):
        content = (submission.title + " " + submission.selftext).lower()
        if company_name.lower() in content:
            posts.append({
                "title": submission.title,
                "content": submission.title + " " + submission.selftext
            })
    return posts


def fetch_rss_news(rss_url):
    feed = feedparser.parse(rss_url)
    news_items = []
    for entry in feed.entries:
        news_items.append({
            "title": entry.title,
            "content": entry.title + " " + entry.summary
        })
    return news_items

def analyze_sentiment(text):
    return get_sentiment_label(text)

def process_sentiment(stock_ticker, stock_name_for_news, num_days=30):
    news_url = f'https://news.google.com/rss/search?q={stock_name_for_news}+stock'
    news_articles = fetch_rss_news(news_url)
    reddit_posts = fetch_reddit_posts(stock_name_for_news)


    df = pd.DataFrame(news_articles + reddit_posts)
    df['clean_text'] = df['content'].apply(clean_text)
    df['sentiment'] = df['clean_text'].apply(analyze_sentiment)

    today = datetime.today().date()
    dates = [today - timedelta(days=i) for i in range(num_days)]
    df['date'] = [dates[i % num_days] for i in range(len(df))]

    sentiment_counts = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0).reset_index()

    stock = yf.Ticker(stock_ticker)
    period_str = f"{num_days}d"
    stock_data = stock.history(period=period_str).reset_index()
    stock_data['Date'] = stock_data['Date'].dt.date

    merged = pd.merge(sentiment_counts, stock_data[['Date', 'Close']], left_on='date', right_on='Date', how='inner')

    return df,merged


def plot_stock_sentiment(merged, stock_ticker):
    merged = merged.sort_values('date')

    fig = go.Figure()

    # Close Price line
    fig.add_trace(go.Scatter(
        x=merged['date'],
        y=merged['Close'],
        mode='lines+markers',
        name=f'{stock_ticker} Close Price',
        marker=dict(color='yellow')
    ))

    # Positive Sentiment line
    if 'positive' in merged.columns:
        fig.add_trace(go.Scatter(
            x=merged['date'],
            y=merged['positive'],
            mode='lines+markers',
            name='Positive Sentiment',
            marker=dict(color='green'),
            yaxis='y2'
        ))

    # Negative Sentiment line
    if 'negative' in merged.columns:
        fig.add_trace(go.Scatter(
            x=merged['date'],
            y=merged['negative'],
            mode='lines+markers',
            name='Negative Sentiment',
            marker=dict(color='red'),
            yaxis='y2'
        ))

    # Neutral Sentiment line
    if 'neutral' in merged.columns:
        fig.add_trace(go.Scatter(
            x=merged['date'],
            y=merged['neutral'],
            mode='lines+markers',
            name='Neutral Sentiment',
            marker=dict(color='grey'),
            yaxis='y2'
        ))

    fig.update_layout(
        title=f'{stock_ticker} Close Price vs Sentiment Trends (Past 30 Days)',
        xaxis=dict(
            title='Date',
            tickangle=45,
            tickfont=dict(size=10),
            tickvals=merged['date'],
            tickformat="%d-%m",
        ),
        yaxis=dict(
            title=f'{stock_ticker} Close Price',
            color='black',
        ),
        yaxis2=dict(
            title='Sentiment Count',
            color='blue',
            overlaying='y',
            side='right',
        ),
        legend=dict(orientation='h', y=-0.2),
        margin=dict(l=40, r=40, t=60, b=80),
        template='plotly_white'

    )


    return fig

# def plot_stock_sentiment(merged, stock_ticker):
#     merged = merged.sort_values('date')
#     fig, ax1 = plt.subplots(figsize=(12, 6))
#     ax1.plot(merged['date'], merged['Close'], color='black', marker='o', label=f'{stock_ticker} Close Price')
#     ax1.set_xlabel('Date')
#     ax1.set_ylabel(f'{stock_ticker} Close Price', color='black')

#     ax2 = ax1.twinx()
#     ax2.plot(merged['date'], merged.get('positive', pd.Series(0, index=merged.index)), color='green', marker='o', label='Positive Sentiment')
#     ax2.plot(merged['date'], merged.get('negative', pd.Series(0, index=merged.index)), color='red', marker='o', label='Negative Sentiment')
#     ax2.plot(merged['date'], merged.get('neutral', pd.Series(0, index=merged.index)), color='grey', marker='o', label='Neutral Sentiment')
#     ax2.set_ylabel('Sentiment Count', color='blue')

#     lines_1, labels_1 = ax1.get_legend_handles_labels()
#     lines_2, labels_2 = ax2.get_legend_handles_labels()
#     ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

#     plt.title(f'{stock_ticker} Close Price vs Sentiment Trends (Past 30 Days)')
#     plt.tight_layout()

#     return fig


def generate_wordcloud(df, sentiment_type):
    text = " ".join(df[df["sentiment"] == sentiment_type]["clean_text"])
    if not text.strip():
        return None

    # Remove overlap with opposite sentiment
    sentiments = {"positive", "negative", "neutral"}
    other_sentiments = sentiments - {sentiment_type}
    other_text = " ".join(df[df["sentiment"].isin(other_sentiments)]["clean_text"])
    other_words = set(other_text.split())

    # Remove words appearing in other sentiments
    words = [word.lower() for word in text.split() if word.lower() not in other_words and len(word) > 2]

    filtered_text = " ".join(words)

    if not filtered_text.strip():
        return None

    colormap = "Greens" if sentiment_type == "positive" else "Reds" if sentiment_type == "negative" else "Blues"
    wc = WordCloud(width=800, height=400, background_color='black', colormap=colormap).generate(filtered_text)
    
    return wc

def plot_correlation_heatmap(merged, stock_ticker):
    # Prepare correlation dataframe
    corr_df = merged[['Close']].copy()
    for sentiment in ['positive', 'negative', 'neutral']:
        if sentiment in merged.columns:
            corr_df[sentiment.capitalize()] = merged[sentiment]

    corr = corr_df.corr()

    fig, ax = plt.subplots(figsize=(3,3))
    cax = ax.matshow(corr, cmap="coolwarm")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)
    fig.colorbar(cax)
    plt.title(f"{stock_ticker}: Sentiment vs Close Price Correlation Heatmap")
    plt.tight_layout()

    return fig
