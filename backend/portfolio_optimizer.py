import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation

def optimize_portfolio(stock_data, cluster_labels, risk_free_rate=0.05, min_weight=0.0, max_weight=0.25):
    """
    Optimize portfolio using Efficient Frontier (Max Sharpe Ratio)
    
    Parameters:
    - stock_data: DataFrame with stock prices
    - cluster_labels: Cluster assignments for each stock
    - risk_free_rate: Risk-free rate for Sharpe calculation
    - min_weight: Minimum weight per asset (default: 0.0)
    - max_weight: Maximum weight per asset (default: 0.25 = 25%)
    
    Returns:
    - Dictionary with optimized weights, metrics, and allocation
    """
    try:
        # Get the latest data for each ticker
        latest_date = stock_data.index.get_level_values('date').max()
        price_data = stock_data['adj close'].unstack('ticker')
        
        # Select only stocks with valid cluster assignments
        valid_tickers = price_data.columns[:len(cluster_labels)]
        price_data = price_data[valid_tickers]
        
        print(f"Optimizing portfolio with {len(valid_tickers)} stocks")
        print(f"Weight constraints: min={min_weight:.1%}, max={max_weight:.1%}")
        
        # Calculate expected returns and covariance
        mu = expected_returns.mean_historical_return(price_data)
        S = risk_models.sample_cov(price_data)
        
        # Optimize for max Sharpe ratio with weight bounds
        ef = EfficientFrontier(mu, S, weight_bounds=(min_weight, max_weight))
        weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
        cleaned_weights = ef.clean_weights()
        
        # Get performance metrics
        performance = ef.portfolio_performance(risk_free_rate=risk_free_rate, verbose=False)
        
        # Filter out zero weights
        non_zero_weights = {k: v for k, v in cleaned_weights.items() if v > 0.0001}
        
        # Sort by weight
        sorted_weights = dict(sorted(non_zero_weights.items(), key=lambda x: x[1], reverse=True))
        
        results = {
            'weights': sorted_weights,
            'expected_return': float(performance[0]),
            'volatility': float(performance[1]),
            'sharpe_ratio': float(performance[2]),
            'n_selected_stocks': len(sorted_weights),
            'max_weight': max_weight,
            'objective': 'ex_post_max_sharpe_single_period'
        }
        
        print(f"Optimization complete. Sharpe Ratio: {performance[2]:.3f}")
        print(f"Expected Return: {performance[0]:.2%}, Volatility: {performance[1]:.2%}")
        print(f"Max single stock weight: {max(sorted_weights.values()):.2%}")
        
        return results
        
    except Exception as e:
        print(f"Error in portfolio optimization: {str(e)}")
        raise