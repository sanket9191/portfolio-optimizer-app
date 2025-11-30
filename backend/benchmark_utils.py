import numpy as np
import pandas as pd


def compute_equity_curve_from_prices(prices: pd.Series, initial_capital: float):
    """
    Given a price series, compute an equity curve assuming buy-and-hold
    with 100% allocation to this asset.
    """
    returns = prices.pct_change().fillna(0.0)
    equity = (1 + returns).cumprod() * initial_capital
    return equity


def compute_equal_weight_benchmark(stock_prices_wide: pd.DataFrame, initial_capital: float):
    """
    Given a wide price DataFrame (index=date, columns=tickers),
    simulate a buy-and-hold equal-weight portfolio.
    """
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


def compute_performance_metrics(equity_curve: pd.Series, risk_free_rate_annual: float):
    """
    Given an equity curve, compute total/annualized returns, volatility, Sharpe, max drawdown.
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
    annualized_return = (1 + mean_daily) ** 252 - 1
    annualized_vol = std_daily * np.sqrt(252)

    rf_daily = (1 + risk_free_rate_annual) ** (1 / 252) - 1
    excess_daily = mean_daily - rf_daily
    sharpe = excess_daily / std_daily if std_daily > 0 else 0.0

    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max) - 1
    max_drawdown = drawdown.min()

    return {
        "initial_capital": initial_capital,
        "final_value": final_value,
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "volatility": float(annualized_vol),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "time_series": {
            "dates": [d.strftime("%Y-%m-%d") for d in equity_curve.index],
            "values": [float(v) for v in equity_curve.values],
        },
    }