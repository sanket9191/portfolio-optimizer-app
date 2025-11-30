import numpy as np
import pandas as pd


def compute_equity_curve_from_prices(prices: pd.Series, initial_capital: float):
    """
    Given a price series, compute an equity curve assuming buy-and-hold
    with 100% allocation to this asset.
    
    Parameters:
    -----------
    prices : pd.Series
        Price series indexed by date
    initial_capital : float
        Starting capital
    
    Returns:
    --------
    pd.Series : Equity curve indexed by date
    """
    if prices is None or prices.empty:
        return None
    
    returns = prices.pct_change().fillna(0.0)
    equity = (1 + returns).cumprod() * initial_capital
    return equity


def compute_equal_weight_benchmark(stock_prices_wide: pd.DataFrame, initial_capital: float):
    """
    Given a wide price DataFrame (index=date, columns=tickers),
    simulate a buy-and-hold equal-weight portfolio.
    
    Parameters:
    -----------
    stock_prices_wide : pd.DataFrame
        Wide-format price data (dates × tickers)
    initial_capital : float
        Starting capital
    
    Returns:
    --------
    pd.Series : Equity curve indexed by date
    """
    if stock_prices_wide is None or stock_prices_wide.empty:
        return None
    
    returns = stock_prices_wide.pct_change().dropna()
    if returns.empty:
        return None

    n_assets = returns.shape[1]
    if n_assets == 0:
        return None

    equal_weights = np.full(n_assets, 1.0 / n_assets)
    portfolio_returns = (returns * equal_weights).sum(axis=1)
    equity = (1 + portfolio_returns).cumprod() * initial_capital
    return equity


def normalize_to_common_dates(strategy_equity, index_equity, ew_equity):
    """
    Normalize all equity curves to common dates and same starting point.
    
    This ensures fair comparison in charts:
    - All curves start at same date
    - All curves start at same value (initial capital)
    - No forward-looking bias
    
    Parameters:
    -----------
    strategy_equity : pd.Series
        Strategy equity curve
    index_equity : pd.Series or None
        Index benchmark equity curve
    ew_equity : pd.Series or None
        Equal-weight benchmark equity curve
    
    Returns:
    --------
    dict : {
        'strategy': pd.Series,
        'index': pd.Series or None,
        'equal_weight': pd.Series or None,
        'common_dates': pd.DatetimeIndex
    }
    """
    if strategy_equity is None or strategy_equity.empty:
        return {
            'strategy': pd.Series(),
            'index': None,
            'equal_weight': None,
            'common_dates': pd.DatetimeIndex([])
        }
    
    # Get common date range (intersection of all available curves)
    common_dates = strategy_equity.index
    
    if index_equity is not None and not index_equity.empty:
        common_dates = common_dates.intersection(index_equity.index)
    
    if ew_equity is not None and not ew_equity.empty:
        common_dates = common_dates.intersection(ew_equity.index)
    
    if len(common_dates) == 0:
        return {
            'strategy': strategy_equity,
            'index': index_equity,
            'equal_weight': ew_equity,
            'common_dates': strategy_equity.index
        }
    
    # Normalize to common dates
    strategy_norm = strategy_equity.loc[common_dates]
    initial_value = strategy_norm.iloc[0]
    
    # Normalize benchmarks to same starting point
    index_norm = None
    if index_equity is not None and not index_equity.empty:
        index_aligned = index_equity.loc[common_dates]
        # Rebase to same initial value
        index_norm = (index_aligned / index_aligned.iloc[0]) * initial_value
    
    ew_norm = None
    if ew_equity is not None and not ew_equity.empty:
        ew_aligned = ew_equity.loc[common_dates]
        # Rebase to same initial value
        ew_norm = (ew_aligned / ew_aligned.iloc[0]) * initial_value
    
    return {
        'strategy': strategy_norm,
        'index': index_norm,
        'equal_weight': ew_norm,
        'common_dates': common_dates
    }


def compute_performance_metrics(equity_curve: pd.Series, risk_free_rate_annual: float):
    """
    Given an equity curve, compute comprehensive performance metrics.
    
    Parameters:
    -----------
    equity_curve : pd.Series
        Equity curve indexed by date
    risk_free_rate_annual : float
        Annual risk-free rate (e.g., 0.05 for 5%)
    
    Returns:
    --------
    dict : Performance metrics including returns, Sharpe, drawdown, etc.
    """
    if equity_curve is None or equity_curve.empty:
        return None

    initial_capital = float(equity_curve.iloc[0])
    final_value = float(equity_curve.iloc[-1])
    total_return = (final_value / initial_capital) - 1.0

    daily_returns = equity_curve.pct_change().dropna()
    if daily_returns.empty:
        return None

    mean_daily = daily_returns.mean()
    std_daily = daily_returns.std()

    # Assume 252 trading days/year
    n_days = len(daily_returns)
    n_years = n_days / 252.0
    
    # Annualized return using compound formula
    if n_years > 0:
        annualized_return = (1 + total_return) ** (1 / n_years) - 1
    else:
        annualized_return = 0.0
    
    annualized_vol = std_daily * np.sqrt(252)

    # Sharpe ratio
    rf_daily = (1 + risk_free_rate_annual) ** (1 / 252) - 1
    excess_daily = mean_daily - rf_daily
    sharpe_daily = excess_daily / std_daily if std_daily > 0 else 0.0
    sharpe_annual = sharpe_daily * np.sqrt(252)

    # Drawdown analysis
    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max) - 1
    max_drawdown = drawdown.min()
    
    # Drawdown duration
    is_drawdown = drawdown < 0
    drawdown_periods = is_drawdown.astype(int).groupby((is_drawdown != is_drawdown.shift()).cumsum()).sum()
    max_drawdown_duration = drawdown_periods.max() if not drawdown_periods.empty else 0

    return {
        "initial_capital": initial_capital,
        "final_value": final_value,
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "volatility": float(annualized_vol),
        "sharpe_ratio": float(sharpe_annual),
        "max_drawdown": float(max_drawdown),
        "max_drawdown_duration_days": int(max_drawdown_duration),
        "n_periods": int(n_days),
        "n_years": float(n_years),
        "time_series": {
            "dates": [d.strftime("%Y-%m-%d") for d in equity_curve.index],
            "values": [float(v) for v in equity_curve.values],
        },
    }


def compute_relative_performance(strategy_metrics, benchmark_metrics):
    """
    Compute relative performance metrics vs benchmark.
    
    Parameters:
    -----------
    strategy_metrics : dict
        Strategy performance metrics
    benchmark_metrics : dict
        Benchmark performance metrics
    
    Returns:
    --------
    dict : Relative metrics (alpha, beta, information ratio, tracking error)
    """
    if strategy_metrics is None or benchmark_metrics is None:
        return None
    
    # Alpha (excess return)
    alpha = strategy_metrics['annualized_return'] - benchmark_metrics['annualized_return']
    
    # Information ratio (alpha / tracking error)
    # Simplified: excess return / excess volatility
    tracking_error = abs(strategy_metrics['volatility'] - benchmark_metrics['volatility'])
    information_ratio = alpha / tracking_error if tracking_error > 0 else 0.0
    
    return {
        'alpha': float(alpha),
        'tracking_error': float(tracking_error),
        'information_ratio': float(information_ratio),
        'sharpe_diff': float(strategy_metrics['sharpe_ratio'] - benchmark_metrics['sharpe_ratio'])
    }
