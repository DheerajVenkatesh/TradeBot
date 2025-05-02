import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib
import praw
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from alpaca.trading.client import TradingClient
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.requests import StockBarsRequest
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderClass, OrderSide
from alpaca.data.historical.stock import StockHistoricalDataClient
import os
from dotenv import load_dotenv

load_dotenv()

reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
reddit_user_agent = os.getenv('REDDIT_USER_AGENT')

alpaca_key = os.getenv('ALPACA_KEY')
alpaca_secret = os.getenv('ALPACA_SECRET')

# API Credentials
api_key = alpaca_key
api_secret = alpaca_secret

reddit = praw.Reddit(
    client_id=reddit_client_id,
    client_secret=reddit_client_secret,
    user_agent=reddit_user_agent)

trading_client = TradingClient(api_key, api_secret, paper=True)
stock_data_client = StockHistoricalDataClient(api_key, api_secret)

st.markdown("<h1 style='text-align: center; font-size: 48px; color: #4CAF50;'>TradeBot</h1>", unsafe_allow_html=True)

initial_user_balance = 10000  # or whatever your starting money is
st.markdown(f"<h3 style='text-align: center; font-size: 24px;'>Initial User Balance: ${initial_user_balance:,.2f}</h3>", unsafe_allow_html=True)

# Load NLTK Sentiment Analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Function to perform sentiment analysis on Reddit
def get_sentiment_score(ticker):
    subreddit = reddit.subreddit("stocks")
    mentions = subreddit.search(ticker, limit=50)
    sentiment_scores = [sia.polarity_scores(post.title)['compound'] for post in mentions]
    return np.mean(sentiment_scores) if sentiment_scores else 0

# Function to fetch stock data for MACD strategy
def get_stock_data(ticker):
    now = datetime.now(ZoneInfo("America/New_York"))
    req = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame(1, TimeFrameUnit.Day),
        start=now - timedelta(days=300),
        limit=300,
    )
    df = stock_data_client.get_stock_bars(req).df
    
    # Reset index to avoid key errors
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()  # Reset index
    df = df[df['symbol'] == ticker]  # Keep only this ticker's data
    df = df.set_index('timestamp')   # Set timestamp as index
    return df

# Function to plot candlestick chart
def plot_candlestick_chart(stock_df):
    fig = go.Figure(data=[go.Candlestick(
        x=stock_df.index,
        open=stock_df['open'],
        high=stock_df['high'],
        low=stock_df['low'],
        close=stock_df['close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        showlegend=False
    )])

    fig.update_layout(
        title=f"Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,
        template="plotly_dark"
    )

    st.plotly_chart(fig)

model = joblib.load('logistic_model.pkl')

df = pd.read_csv('predicted_stocks.csv')

# Rank stocks using the trained model
good_stocks = df[df['predicted_good_stock'] == 1]

# Select the best stock
top_stock = good_stocks
st.write(f"### Selected Stock for Trading:")
st.dataframe(top_stock[['Symbol','Name']])

def prepare_stock_data(ticker):
    """
    Fetch stock data, calculate SMA200 and MACD indicators, and return the prepared DataFrame.
    """
    stock_df = get_stock_data(ticker)
    stock_df["SMA200"] = ta.sma(stock_df['close'], length=200)
    macd = stock_df.ta.macd(close='close', fast=12, slow=26, signal=9)
    stock_df = stock_df.join(macd)
    return stock_df

# Initialize buy_price to None
buy_price = None
user_money = 10000
# Check for a buy signal
for idx, row in top_stock.iterrows():
    ticker = row['Symbol']  # Assuming your column is named 'Symbol'

    st.write(f"#### Processing {ticker}")

    # Fetch and prepare stock data
    stock_df = prepare_stock_data(ticker)

    # Plot candlestick chart 
    plot_candlestick_chart(stock_df)

    # Get latest signal
    latest_signal = stock_df.iloc[-1]

    # Check MACD + SMA200 Buy condition
    if (
        latest_signal["close"] > latest_signal["SMA200"]
        and latest_signal["MACD_12_26_9"] > latest_signal["MACDs_12_26_9"]
    ):
        buy_price = latest_signal["close"]
        take_profit = round(buy_price * 1.1, 2)
        stop_loss = round(buy_price * 0.95, 2)

        # Place order
        market_order_data = MarketOrderRequest(
            symbol=ticker,
            qty=1,
            side=OrderSide.BUY,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=take_profit),
            stop_loss=StopLossRequest(stop_price=stop_loss),
            time_in_force="day",
            extended_hours=True
        )

        try:
            trading_client.submit_order(order_data=market_order_data)
            st.success(f"Trade executed: Bought {ticker} at ${buy_price}")
        except Exception as e:
            st.error(f"Failed to place order for {ticker}: {e}")
            continue  # move to the next stock
    else:
        st.info(f"No buy signal for {ticker}")

# Display placed orders

orders_placed = []
total_spent = 0

for index,row in top_stock.iterrows():
    stock_name = row['Name']
    stock_symbol = row['Symbol']
    stock_data = get_stock_data(stock_symbol)
    
    if stock_data is not None and not stock_data.empty:
        latest = stock_data.iloc[-1]    # Get the most recent row
        buy_price = latest['close']  

        orders_placed.append({
                "stock": stock_name,
                "price": buy_price,
                "quantity": 1
            })
        total_spent += buy_price

st.write("### Orders Placed:")
if orders_placed:
    orders_df = pd.DataFrame(orders_placed)
    st.dataframe(orders_df)
else:
    st.write('No Orders Placed')

if buy_price is not None:
    user_money -= total_spent
    st.write(f" ### User Balance: ${user_money:.2f}")
else:
    st.write(f" User Balance: ${user_money:.2f}")  # Display balance if no buy order executed

