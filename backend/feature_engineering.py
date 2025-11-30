import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import MACD

def calculate_features(df):
    """
    Calculate technical indicators and features for clustering
    
    Parameters:
    - df: DataFrame with stock data (multi-index: date, ticker)
    
    Returns:
    - DataFrame with calculated features
    """
    print("Calculating Garman-Klass Volatility...")
    df['garman_klass_vol'] = (
        (np.log(df['high']) - np.log(df['low']))**2 / 2 - 
        (2*np.log(2) - 1) * (np.log(df['adj close']) - np.log(df['open']))**2
    )
    
    print("Calculating RSI...")
    def calc_rsi(x):
        try:
            rsi = RSIIndicator(close=x, window=20)
            return rsi.rsi()
        except:
            return pd.Series(index=x.index, dtype=float)
    
    df['rsi'] = df.groupby(level=1)['adj close'].transform(calc_rsi)
    
    print("Calculating Bollinger Bands...")
    def calc_bollinger(x):
        try:
            bb = BollingerBands(close=np.log1p(x), window=20, window_dev=2)
            return pd.DataFrame({
                'bb_low': bb.bollinger_lband(),
                'bb_mid': bb.bollinger_mavg(),
                'bb_high': bb.bollinger_hband()
            })
        except:
            return pd.DataFrame({
                'bb_low': pd.Series(index=x.index, dtype=float),
                'bb_mid': pd.Series(index=x.index, dtype=float),
                'bb_high': pd.Series(index=x.index, dtype=float)
            })
    
    bb_data = df.groupby(level=1)['adj close'].apply(calc_bollinger)
    df['bb_low'] = bb_data['bb_low'].values
    df['bb_mid'] = bb_data['bb_mid'].values
    df['bb_high'] = bb_data['bb_high'].values
    
    print("Calculating ATR...")
    def compute_atr(stock_data):
        try:
            atr_indicator = AverageTrueRange(
                high=stock_data['high'],
                low=stock_data['low'],
                close=stock_data['close'],
                window=14
            )
            atr = atr_indicator.average_true_range()
            return atr.sub(atr.mean()).div(atr.std())
        except:
            return pd.Series(index=stock_data.index, dtype=float)
    
    df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr)
    
    print("Calculating MACD...")
    def compute_macd(close):
        try:
            macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
            macd_line = macd.macd()
            return macd_line.sub(macd_line.mean()).div(macd_line.std())
        except:
            return pd.Series(index=close.index, dtype=float)
    
    df['macd'] = df.groupby(level=1, group_keys=False)['adj close'].apply(compute_macd)
    
    print("Calculating Dollar Volume...")
    df['dollar_volume'] = (df['adj close'] * df['volume']) / 1e6
    
    print("Calculating returns for multiple horizons...")
    df = df.groupby(level=1, group_keys=False).apply(calculate_returns)
    
    # Aggregate to monthly
    print("Aggregating to monthly frequency...")
    last_cols = [c for c in df.columns if c not in ['dollar_volume', 'volume', 'open', 'high', 'low', 'close']]
    
    data = pd.concat([
        df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume'),
        df.unstack()[last_cols].resample('M').last().stack('ticker')
    ], axis=1).dropna()
    
    # Filter top liquid stocks
    print("Filtering top 100 liquid stocks...")
    data['dollar_volume'] = data.groupby('ticker')['dollar_volume'].transform(
        lambda x: x.rolling(24, min_periods=12).mean()
    )
    data['dollar_vol_rank'] = data.groupby('date')['dollar_volume'].rank(ascending=False)
    data = data[data['dollar_vol_rank'] < 100].drop(['dollar_volume', 'dollar_vol_rank'], axis=1)
    
    return data.dropna()

def calculate_returns(df):
    """Calculate returns for multiple time horizons"""
    outlier_cutoff = 0.005
    lags = [1, 2, 3, 6, 9, 12]
    
    for lag in lags:
        df[f'return_{lag}m'] = (
            df['adj close']
            .pct_change(lag)
            .pipe(lambda x: x.clip(
                lower=x.quantile(outlier_cutoff),
                upper=x.quantile(1-outlier_cutoff)
            ))
            .add(1)
            .pow(1/lag)
            .sub(1)
        )
    
    return df
