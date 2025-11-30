import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

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