# TradeBot

![Trading Robot](https://img.shields.io/badge/Trading-Automated-brightgreen)
![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Python](https://img.shields.io/badge/Python-3.7+-blue)

TradeBot is an automated stock trading application built with Streamlit that uses machine learning to identify promising stocks and execute trades based on technical indicators.

## Features

- **Machine Learning Stock Selection**: Uses logistic regression to identify promising stocks based on key financial metrics
- **Technical Analysis**: Implements MACD and SMA200 technical indicators for trade signal generation
- **Reddit Sentiment Analysis**: Incorporates social media sentiment from r/stocks subreddit
- **Live Trading**: Integrates with Alpaca Trading API for paper (simulated) trading
- **Interactive Dashboard**: Visualizes stock data with candlestick charts and trade information

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tradebot.git
cd tradebot
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=your_reddit_user_agent
ALPACA_KEY=your_alpaca_api_key
ALPACA_SECRET=your_alpaca_secret_key
```

## Usage

1. Run the model training script first:
```bash
python model_training.py
```

2. Launch the Streamlit application:
```bash
streamlit run app.py
```

3. Access the dashboard at http://localhost:8501

## How It Works

### Stock Selection Model

The application uses a logistic regression model trained on financial data to identify "good stocks" based on the following criteria:
- Price/Earnings ratio between 0 and 25
- Positive Earnings per Share
- Market Cap >= $2 billion
- Price/Book ratio between 0 and 3
- Dividend Yield between 2% and 6%

### Trading Strategy

TradeBot implements a technical analysis strategy combining:
1. **MACD (Moving Average Convergence Divergence)**: A trend-following momentum indicator
2. **SMA200 (200-day Simple Moving Average)**: A long-term trend indicator

A buy signal is generated when:
- The stock price is above its SMA200
- The MACD line crosses above the signal line

The strategy includes automatic take-profit (10% gain) and stop-loss (5% loss) orders.

### Sentiment Analysis

The application incorporates sentiment analysis from Reddit's r/stocks subreddit to gauge market sentiment for particular stocks.

## Files

- **app.py**: Main Streamlit application file
- **model_training.py**: Script for training the stock selection model
- **logistic_model.pkl**: Saved machine learning model
- **predicted_stocks.csv**: CSV file with stock predictions
- **constituents-financials.csv**: Financial data for training
- **.env_sample**: For all api keys and secrets

## Dependencies

- streamlit
- plotly
- pandas
- pandas_ta
- numpy
- joblib
- praw
- nltk
- alpaca-trade-api
- python-dotenv

## Disclaimer

This application is for educational purposes only. Trading stocks involves risk, and this tool should not be used as financial advice. Always do your own research before making investment decisions.
