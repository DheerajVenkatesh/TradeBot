import praw  # Reddit API
import yfinance as yf  # Stock data
import numpy as np
import pandas as pd
import pandas_ta as ta  # Technical indicators
import joblib  # Save and load models
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from sklearn.ensemble import RandomForestClassifier
from alpaca.trading.client import TradingClient
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from alpaca.data.requests import StockBarsRequest
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderClass, OrderSide
from alpaca.data.historical.stock import StockHistoricalDataClient

# API Credentials
api_key = "PK7SY9ATFI7M8OCEVGHK"
api_secret = "Dw74xnRlY3ImyHa3yvCTMO7ArNGxl7d5sacUizhq"

reddit = praw.Reddit(
    client_id='rLAtyw4_uLGI-Zpd1WdxwA',
    client_secret='tXotg2-KyG9YL4Q_iIA5c5ZRu68dxw',
    user_agent='SentiAnalysisBot/1.0 by InfiniteAd347')

trading_client = TradingClient(api_key, api_secret, paper=True)
stock_data_client = StockHistoricalDataClient(api_key, api_secret)

# Load NLTK Sentiment Analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Define stock list
stock_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

# Function to fetch fundamental data from Yahoo Finance
def get_fundamental_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "market_cap": info.get("marketCap", 0),
        "pe_ratio": info.get("trailingPE", 0),
        "roe": info.get("returnOnEquity", 0),
        "eps": info.get("trailingEps", 0),
    }

# Function to perform sentiment analysis on Reddit
def get_sentiment_score(ticker):
    subreddit = reddit.subreddit("stocks")
    mentions = subreddit.search(ticker, limit=50)
    sentiment_scores = [sia.polarity_scores(post.title)['compound'] for post in mentions]
    return np.mean(sentiment_scores) if sentiment_scores else 0

# Prepare dataset for ranking
stock_data = []
for ticker in stock_list:
    fundamentals = get_fundamental_data(ticker)
    sentiment = get_sentiment_score(ticker)
    fundamentals["sentiment"] = sentiment
    fundamentals["ticker"] = ticker
    stock_data.append(fundamentals)

df = pd.DataFrame(stock_data)

# Define ranking model (RandomForest)
X = df.drop(columns=["ticker"])
y = np.arange(len(X))  # Dummy ranking labels
model = RandomForestClassifier()
model.fit(X, y)

# Rank stocks
df["rank_score"] = model.predict_proba(X)[:, 1]
df = df.sort_values(by="rank_score", ascending=False)

# Select the best stock
top_stock = df.iloc[0]["ticker"]
print(f"Selected Stock for Trading: {top_stock}")

# Function to fetch stock data for MACD strategy
def get_stock_data(ticker):
    now = datetime.now(ZoneInfo("America/New_York"))
    req = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame(1, TimeFrameUnit.Day),
        start=now - timedelta(days=300),
        limit=300,
    )
    return stock_data_client.get_stock_bars(req).df

# Apply MACD strategy
stock_df = get_stock_data(top_stock)
stock_df["SMA200"] = ta.sma(stock_df['close'], length=200)
macd = stock_df.ta.macd(close='close', fast=12, slow=26, signal=9)
stock_df = stock_df.join(macd)

stock_df["Below_Zero"] = (stock_df["MACD_12_26_9"] < 0) & (stock_df["MACDs_12_26_9"] < 0)
stock_df["Crossover"] = (stock_df["MACD_12_26_9"] > stock_df["MACDs_12_26_9"]) & (stock_df["MACD_12_26_9"].shift(1) <= stock_df["MACDs_12_26_9"].shift(1))

# Check for a buy signal
latest_signal = stock_df.iloc[-1]
if latest_signal["close"] > latest_signal["SMA200"] and latest_signal["Crossover"] and latest_signal["Below_Zero"] :
    buy_price = latest_signal["close"]
    take_profit = round(buy_price * 1.1, 2)
    stop_loss = round(buy_price * 0.95, 2)
    
    market_order_data = MarketOrderRequest(
        symbol=top_stock,
        qty=1,
        side=OrderSide.BUY,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=take_profit),
        stop_loss=StopLossRequest(stop_price=stop_loss),
        time_in_force="day",
        extended_hours=True
    )
    
    trading_client.submit_order(order_data=market_order_data)
    print(f"✅ Trade executed: Bought {top_stock}")
else:
    print(f"❌ No buy signal for {top_stock}")
