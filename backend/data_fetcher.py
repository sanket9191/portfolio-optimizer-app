import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# Benchmark ticker mapping
BENCHMARK_MAP = {
    "NIFTY50": "^NSEI",        # NIFTY 50
    "NIFTYBANK": "^NSEBANK",   # NIFTY Bank
    "NIFTY500": "^CRSLDX",     # NIFTY 500 proxy
}

def fetch_stock_data(tickers, start_date, end_date):
    """
    Fetch historical stock data from Yahoo Finance
    
    Parameters:
    - tickers: List of stock tickers (e.g., ['RELIANCE.NS', 'TCS.NS'])
    - start_date: Start date (YYYY-MM-DD)
    - end_date: End date (YYYY-MM-DD)
    
    Returns:
    - DataFrame with multi-index (date, ticker) and OHLCV data
    """
    try:
        print(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}...")
        
        # Download data
        df = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False
        )
        
        if df.empty:
            raise ValueError("No data downloaded. Check tickers and dates.")
        
        # Stack to get multi-index format
        df = df.stack()
        df.index.names = ['date', 'ticker']
        df.columns = df.columns.str.lower()
        df.columns.name = None
        
        print(f"Successfully downloaded {len(df)} data points")
        
        return df
        
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        raise

def fetch_benchmark_data(benchmark_name, start_date, end_date):
    """
    Fetch benchmark index data between start_date and end_date.
    Returns a DataFrame with 'date' index and 'adj_close' column, or None if not available.
    """
    if benchmark_name not in BENCHMARK_MAP:
        return None

    ticker = BENCHMARK_MAP[benchmark_name]
    print(f"Fetching benchmark {benchmark_name} ({ticker}) from {start_date} to {end_date}...")

    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if data.empty:
            print(f"No data for benchmark {benchmark_name}")
            return None

        # Handle both single ticker (Series) and multi-ticker (DataFrame) cases
        if 'Adj Close' in data.columns:
            data = data[['Adj Close']].copy()
            data.columns = ['adj_close']
        else:
            print(f"Available columns: {data.columns.tolist()}")
            return None
            
        data.index.name = 'date'
        print(f"Successfully fetched {len(data)} data points for {benchmark_name}")
        return data
    except Exception as e:
        print(f"Error fetching benchmark data: {str(e)}")
        return None