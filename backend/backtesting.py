import numpy as np
import pandas as pd

def backtest_strategy(stock_data, portfolio_weights, initial_capital=100000):
    """
    Backtest the portfolio strategy
    
    Parameters:
    - stock_data: DataFrame with stock prices
    - portfolio_weights: Dictionary of ticker: weight
    - initial_capital: Starting capital
    
    Returns:
    - Dictionary with backtest results and metrics
    """
    try:
        # Get price data
        price_data = stock_data['adj close'].unstack('ticker')
        
        # Filter to portfolio stocks
        portfolio_tickers = list(portfolio_weights.keys())
        prices = price_data[portfolio_tickers].dropna()
        
        # Calculate portfolio value
        weights = np.array([portfolio_weights[t] for t in portfolio_tickers])
        returns = prices.pct_change().fillna(0)
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        portfolio_value = initial_capital * cumulative_returns
        
        # Calculate metrics
        total_return = (portfolio_value.iloc[-1] / initial_capital) - 1
        annualized_return = ((1 + total_return) ** (252 / len(portfolio_returns))) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.05) / volatility if volatility > 0 else 0
        
        # Max drawdown
        cumulative = portfolio_value
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Prepare time series data for frontend
        dates = portfolio_value.index.strftime('%Y-%m-%d').tolist()
        values = portfolio_value.tolist()
        
        results = {
            'initial_capital': initial_capital,
            'final_value': float(portfolio_value.iloc[-1]),
            'total_return': float(total_return),
            'annualized_return': float(annualized_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'time_series': {
                'dates': dates,
                'portfolio_values': values
            }
        }
        
        print(f"Backtest complete. Total Return: {total_return:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.3f}, Max Drawdown: {max_drawdown:.2%}")
        
        return results
        
    except Exception as e:
        print(f"Error in backtesting: {str(e)}")
        raise