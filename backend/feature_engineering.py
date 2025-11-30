import pandas as pd
import numpy as np
import pandas_ta

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
    df['rsi'] = df.groupby(level=1)['adj close'].transform(
        lambda x: pandas_ta.rsi(close=x, length=20)
    )
    
    print("Calculating Bollinger Bands...")
    df['bb_low'] = df.groupby(level=1)['adj close'].transform(
        lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,0]
    )
    df['bb_mid'] = df.groupby(level=1)['adj close'].transform(
        lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,1]
    )
    df['bb_high'] = df.groupby(level=1)['adj close'].transform(
        lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,2]
    )
    
    print("Calculating ATR...")
    def compute_atr(stock_data):
        atr = pandas_ta.atr(
            high=stock_data['high'],
            low=stock_data['low'],
            close=stock_data['close'],
            length=14
        )
        return atr.sub(atr.mean()).div(atr.std())
    
    df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr)
    
    print("Calculating MACD...")
    def compute_macd(close):
        macd = pandas_ta.macd(close=close, length=20).iloc[:,0]
        return macd.sub(macd.mean()).div(macd.std())
    
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