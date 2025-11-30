import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import calculate_features
from clustering import perform_clustering
from portfolio_optimizer import optimize_portfolio


class WalkForwardEngine:
    """
    Institutional-grade walk-forward backtesting engine.
    
    Implements realistic out-of-sample testing with:
    - Rolling optimization windows
    - Periodic rebalancing
    - Transaction cost modeling
    - Proper date alignment
    - No look-ahead bias
    """
    
    def __init__(self, stock_data, config):
        """
        Initialize walk-forward engine.
        
        Parameters:
        -----------
        stock_data : pd.DataFrame
            Multi-index DataFrame (date, ticker) with OHLCV data
        config : dict
            Configuration with keys:
            - lookback_months: int (e.g., 24)
            - rebalance_freq: str ('M'=monthly, 'Q'=quarterly, 'W'=weekly)
            - n_clusters: int
            - risk_free_rate: float
            - max_weight: float
            - transaction_cost_bps: float (basis points per trade)
            - initial_capital: float
        """
        self.stock_data = stock_data
        self.config = config
        
        # Extract date range
        self.dates = stock_data.index.get_level_values('date').unique().sort_values()
        self.start_date = self.dates.min()
        self.end_date = self.dates.max()
        
        # Results storage
        self.rebalance_history = []
        self.portfolio_values = []
        self.weights_history = []
        self.transaction_costs = []
        
        print(f"\n{'='*80}")
        print("WALK-FORWARD BACKTESTING ENGINE INITIALIZED")
        print(f"{'='*80}")
        print(f"Data range: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Lookback period: {config['lookback_months']} months")
        print(f"Rebalancing frequency: {config['rebalance_freq']}")
        print(f"Transaction cost: {config['transaction_cost_bps']} bps per trade")
        print(f"Initial capital: ₹{config['initial_capital']:,.0f}")
        print(f"{'='*80}\n")
    
    def generate_rebalance_dates(self):
        """
        Generate rebalancing dates based on frequency.
        Ensures sufficient lookback data exists.
        """
        lookback_months = self.config['lookback_months']
        freq = self.config['rebalance_freq']
        
        # First rebalance date must have enough lookback
        first_rebalance = self.start_date + relativedelta(months=lookback_months)
        
        # Generate dates based on frequency
        rebalance_dates = pd.date_range(
            start=first_rebalance,
            end=self.end_date,
            freq=freq
        )
        
        # Align to actual trading dates in data
        aligned_dates = []
        for target_date in rebalance_dates:
            # Find closest actual date in data
            available_dates = self.dates[self.dates >= target_date]
            if len(available_dates) > 0:
                aligned_dates.append(available_dates[0])
        
        print(f"Generated {len(aligned_dates)} rebalancing dates")
        return pd.DatetimeIndex(aligned_dates)
    
    def get_lookback_data(self, rebalance_date):
        """
        Extract lookback window data for optimization.
        
        Returns data from [rebalance_date - lookback] to [rebalance_date - 1 day]
        """
        lookback_months = self.config['lookback_months']
        lookback_start = rebalance_date - relativedelta(months=lookback_months)
        
        # Get data up to (but not including) rebalance date
        mask = (
            (self.stock_data.index.get_level_values('date') >= lookback_start) &
            (self.stock_data.index.get_level_values('date') < rebalance_date)
        )
        
        lookback_data = self.stock_data.loc[mask].copy()
        
        return lookback_data, lookback_start
    
    def optimize_at_date(self, rebalance_date):
        """
        Run full optimization pipeline at a specific rebalance date.
        Uses only data available up to that date (no look-ahead).
        """
        print(f"\n[{rebalance_date.date()}] Optimizing portfolio...")
        
        # Get lookback data
        lookback_data, lookback_start = self.get_lookback_data(rebalance_date)
        
        if lookback_data.empty:
            print(f"  ⚠️  No data available for lookback window")
            return None
        
        print(f"  Lookback window: {lookback_start.date()} to {rebalance_date.date()}")
        print(f"  Data points: {len(lookback_data)}")
        
        try:
            # Calculate features on lookback data only
            features_df = calculate_features(lookback_data)
            
            if features_df.empty:
                print(f"  ⚠️  Feature calculation failed")
                return None
            
            # Perform clustering
            cluster_results = perform_clustering(
                features_df, 
                self.config['n_clusters']
            )
            
            # Optimize portfolio
            portfolio_results = optimize_portfolio(
                lookback_data,
                cluster_results['labels'],
                risk_free_rate=self.config['risk_free_rate'],
                min_weight=0.0,
                max_weight=self.config['max_weight']
            )
            
            weights = portfolio_results['weights']
            print(f"  ✓ Optimized: {len(weights)} stocks selected")
            print(f"  ✓ Sharpe ratio: {portfolio_results['sharpe_ratio']:.3f}")
            
            return {
                'date': rebalance_date,
                'weights': weights,
                'expected_return': portfolio_results['expected_return'],
                'volatility': portfolio_results['volatility'],
                'sharpe_ratio': portfolio_results['sharpe_ratio'],
                'clusters': cluster_results
            }
            
        except Exception as e:
            print(f"  ❌ Optimization failed: {str(e)}")
            return None
    
    def calculate_transaction_costs(self, old_weights, new_weights, portfolio_value):
        """
        Calculate transaction costs based on turnover.
        
        Cost = (sum of |weight_change|) * portfolio_value * (cost_bps / 10000)
        """
        cost_bps = self.config['transaction_cost_bps']
        
        # Get all tickers
        all_tickers = set(list(old_weights.keys()) + list(new_weights.keys()))
        
        # Calculate turnover
        turnover = 0.0
        for ticker in all_tickers:
            old_w = old_weights.get(ticker, 0.0)
            new_w = new_weights.get(ticker, 0.0)
            turnover += abs(new_w - old_w)
        
        # Cost in currency
        cost = (turnover * portfolio_value * cost_bps) / 10000
        
        return cost, turnover
    
    def calculate_period_returns(self, weights, start_date, end_date):
        """
        Calculate portfolio returns for a holding period.
        
        Returns:
        --------
        pd.Series : Daily portfolio values indexed by date
        """
        # Get price data for the period
        mask = (
            (self.stock_data.index.get_level_values('date') >= start_date) &
            (self.stock_data.index.get_level_values('date') <= end_date)
        )
        period_data = self.stock_data.loc[mask]
        
        if period_data.empty:
            return pd.Series(dtype=float)
        
        # Build price matrix
        price_df = period_data['adj close'].unstack('ticker')
        
        # Filter to only stocks in portfolio
        tickers_in_portfolio = list(weights.keys())
        available_tickers = [t for t in tickers_in_portfolio if t in price_df.columns]
        
        if not available_tickers:
            return pd.Series(dtype=float)
        
        price_df = price_df[available_tickers].fillna(method='ffill')
        
        # Normalize weights for available tickers
        available_weights = {t: weights[t] for t in available_tickers}
        total_weight = sum(available_weights.values())
        
        if total_weight == 0:
            return pd.Series(dtype=float)
        
        normalized_weights = {t: w/total_weight for t, w in available_weights.items()}
        
        # Calculate returns
        returns = price_df.pct_change().fillna(0.0)
        
        # Portfolio returns = weighted sum
        weight_series = pd.Series(normalized_weights)
        portfolio_returns = (returns * weight_series).sum(axis=1)
        
        return portfolio_returns
    
    def run(self):
        """
        Execute full walk-forward backtest.
        
        Returns:
        --------
        dict : Comprehensive backtest results
        """
        print("\n🚀 Starting walk-forward backtest...\n")
        
        # Generate rebalancing schedule
        rebalance_dates = self.generate_rebalance_dates()
        
        if len(rebalance_dates) == 0:
            raise ValueError("No valid rebalancing dates generated. Check data range and lookback period.")
        
        # Initialize
        current_capital = self.config['initial_capital']
        current_weights = {}
        total_transaction_costs = 0.0
        
        # Storage
        equity_curve = []
        rebalance_records = []
        
        # Process each rebalance date
        for i, rebal_date in enumerate(rebalance_dates):
            # Optimize at this date
            optimization_result = self.optimize_at_date(rebal_date)
            
            if optimization_result is None:
                print(f"  ⚠️  Skipping rebalance due to optimization failure")
                continue
            
            new_weights = optimization_result['weights']
            
            # Calculate transaction costs
            txn_cost, turnover = self.calculate_transaction_costs(
                current_weights, 
                new_weights, 
                current_capital
            )
            
            # Deduct transaction costs
            current_capital -= txn_cost
            total_transaction_costs += txn_cost
            
            print(f"  💰 Transaction cost: ₹{txn_cost:,.0f} (Turnover: {turnover:.1%})")
            
            # Determine next rebalance date (or end date)
            if i < len(rebalance_dates) - 1:
                next_rebal_date = rebalance_dates[i + 1]
            else:
                next_rebal_date = self.end_date
            
            # Calculate returns for holding period
            period_returns = self.calculate_period_returns(
                new_weights,
                rebal_date,
                next_rebal_date
            )
            
            if not period_returns.empty:
                # Build equity curve for this period
                period_values = (1 + period_returns).cumprod() * current_capital
                
                # Update capital for next period
                current_capital = period_values.iloc[-1]
                
                # Store equity curve points
                for date, value in period_values.items():
                    equity_curve.append({'date': date, 'value': value})
            
            # Record rebalance
            rebalance_records.append({
                'date': rebal_date,
                'weights': new_weights.copy(),
                'n_stocks': len(new_weights),
                'turnover': turnover,
                'transaction_cost': txn_cost,
                'portfolio_value': current_capital,
                'expected_return': optimization_result['expected_return'],
                'volatility': optimization_result['volatility'],
                'sharpe_ratio': optimization_result['sharpe_ratio']
            })
            
            # Update current weights
            current_weights = new_weights
        
        # Build results
        equity_df = pd.DataFrame(equity_curve).set_index('date')['value']
        
        # Calculate performance metrics
        results = self._calculate_performance_metrics(
            equity_df,
            rebalance_records,
            total_transaction_costs
        )
        
        print(f"\n{'='*80}")
        print("WALK-FORWARD BACKTEST COMPLETE")
        print(f"{'='*80}")
        print(f"Total rebalances: {len(rebalance_records)}")
        print(f"Final value: ₹{current_capital:,.0f}")
        print(f"Total return: {results['total_return']:.2%}")
        print(f"Annualized return: {results['annualized_return']:.2%}")
        print(f"Volatility: {results['volatility']:.2%}")
        print(f"Sharpe ratio: {results['sharpe_ratio']:.3f}")
        print(f"Max drawdown: {results['max_drawdown']:.2%}")
        print(f"Total transaction costs: ₹{total_transaction_costs:,.0f} ({total_transaction_costs/self.config['initial_capital']:.2%})")
        print(f"Avg turnover per rebalance: {results['avg_turnover']:.1%}")
        print(f"{'='*80}\n")
        
        return results
    
    def _calculate_performance_metrics(self, equity_curve, rebalance_records, total_txn_costs):
        """
        Calculate comprehensive performance metrics.
        """
        initial_capital = self.config['initial_capital']
        final_value = equity_curve.iloc[-1]
        total_return = (final_value / initial_capital) - 1.0
        
        # Daily returns
        daily_returns = equity_curve.pct_change().dropna()
        
        if daily_returns.empty:
            return self._empty_metrics()
        
        # Annualized metrics (252 trading days)
        n_days = len(daily_returns)
        n_years = n_days / 252.0
        
        annualized_return = (1 + total_return) ** (1 / n_years) - 1
        annualized_vol = daily_returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        rf_daily = (1 + self.config['risk_free_rate']) ** (1/252) - 1
        excess_daily = daily_returns.mean() - rf_daily
        sharpe = excess_daily / daily_returns.std() if daily_returns.std() > 0 else 0.0
        sharpe_annualized = sharpe * np.sqrt(252)
        
        # Drawdown
        running_max = equity_curve.cummax()
        drawdown = (equity_curve / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Turnover statistics
        turnovers = [r['turnover'] for r in rebalance_records]
        avg_turnover = np.mean(turnovers) if turnovers else 0.0
        
        return {
            'initial_capital': initial_capital,
            'final_value': float(final_value),
            'total_return': float(total_return),
            'annualized_return': float(annualized_return),
            'volatility': float(annualized_vol),
            'sharpe_ratio': float(sharpe_annualized),
            'max_drawdown': float(max_drawdown),
            'n_rebalances': len(rebalance_records),
            'total_transaction_costs': float(total_txn_costs),
            'transaction_costs_pct': float(total_txn_costs / initial_capital),
            'avg_turnover': float(avg_turnover),
            'time_series': {
                'dates': [d.strftime('%Y-%m-%d') for d in equity_curve.index],
                'portfolio_values': [float(v) for v in equity_curve.values]
            },
            'rebalance_history': rebalance_records
        }
    
    def _empty_metrics(self):
        """Return empty metrics structure."""
        return {
            'initial_capital': self.config['initial_capital'],
            'final_value': self.config['initial_capital'],
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'n_rebalances': 0,
            'total_transaction_costs': 0.0,
            'transaction_costs_pct': 0.0,
            'avg_turnover': 0.0,
            'time_series': {'dates': [], 'portfolio_values': []},
            'rebalance_history': []
        }