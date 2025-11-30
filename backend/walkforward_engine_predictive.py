"""
Predictive Walk-Forward Portfolio Optimization Engine

Institutional-grade implementation with:
- Machine learning return forecasts (alpha)
- Robust risk estimation (covariance)
- Proper alpha-risk separation
- Time-series cross-validation
- IC tracking and diagnostics
- Transaction cost modeling
- Position and risk controls
"""

import numpy as np
import pandas as pd
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

from walkforward_engine import WalkForwardEngine
from predictive_models import PredictiveAlphaModel, EnsembleAlphaModel
from feature_engineering import calculate_features
from pypfopt import expected_returns, risk_models, EfficientFrontier
from pypfopt.efficient_frontier import EfficientFrontier


class PredictiveWalkForwardEngine(WalkForwardEngine):
    """
    Extended walk-forward engine with predictive alpha models.
    
    Key enhancements over Phase 2:
    1. ML-based return forecasts instead of historical means
    2. Separate alpha (short window) and risk (long window) estimation
    3. Information Coefficient tracking
    4. Model diagnostics and feature importance
    5. Ensemble modeling for robustness
    """
    
    def __init__(
        self,
        stock_data,
        config,
        use_predictive=True,
        model_type='ridge',
        forecast_horizon=3,
        use_ensemble=False,
        alpha_lookback_months=12,
        risk_lookback_months=36
    ):
        """
        Initialize predictive walk-forward engine.
        
        Parameters:
        -----------
        stock_data : pd.DataFrame
            Multi-index stock data (date, ticker)
        config : dict
            Configuration parameters
        use_predictive : bool
            Use ML forecasts (True) or historical means (False)
        model_type : str
            'ridge', 'lasso', 'elastic_net', 'random_forest', 'gradient_boosting'
        forecast_horizon : int
            Forecasting horizon in months
        use_ensemble : bool
            Use ensemble of models for robustness
        alpha_lookback_months : int
            Lookback window for alpha signals (features)
        risk_lookback_months : int
            Lookback window for risk estimation (covariance)
        """
        super().__init__(stock_data, config)
        
        self.use_predictive = use_predictive
        self.model_type = model_type
        self.forecast_horizon = forecast_horizon
        self.use_ensemble = use_ensemble
        self.alpha_lookback_months = alpha_lookback_months
        self.risk_lookback_months = risk_lookback_months
        
        # Model tracking
        self.alpha_model = None
        self.ic_history = []
        self.forecast_history = []
        self.feature_importance_history = []
        
        # Enhanced diagnostics
        self.realized_vs_forecast = []
        
        print(f"\n{'='*80}")
        print(f"PREDICTIVE WALK-FORWARD ENGINE")
        print(f"{'='*80}")
        print(f"Mode: {'Predictive ML' if use_predictive else 'Historical Means'}")
        if use_predictive:
            print(f"Model: {model_type.upper()}")
            print(f"Forecast Horizon: {forecast_horizon} months")
            print(f"Ensemble: {'Yes' if use_ensemble else 'No'}")
            print(f"Alpha Lookback: {alpha_lookback_months} months")
            print(f"Risk Lookback: {risk_lookback_months} months")
        print(f"{'='*80}\n")
    
    def run(self):
        """
        Execute predictive walk-forward backtest.
        
        At each rebalance date:
        1. Train predictive model on past data
        2. Generate forecasts for next period
        3. Estimate covariance from longer history
        4. Optimize portfolio with forecasts
        5. Track performance and diagnostics
        """
        results = super().run()
        
        # Add predictive-specific diagnostics
        if self.use_predictive and len(self.ic_history) > 0:
            results['predictive_diagnostics'] = {
                'ic_history': self.ic_history,
                'mean_ic': np.mean([x['ic'] for x in self.ic_history if x['ic'] is not None]),
                'ic_stability': np.std([x['ic'] for x in self.ic_history if x['ic'] is not None]),
                'forecast_quality': self._assess_forecast_quality(),
                'feature_importance': self.feature_importance_history[-1] if self.feature_importance_history else None,
                'model_type': self.model_type,
                'forecast_horizon': self.forecast_horizon
            }
        
        return results
    
    def optimize_at_date(self, rebalance_date):
        """
        Optimize portfolio at rebalance date with predictive forecasts.
        
        This is the core method that differs from Phase 2.
        
        Key steps:
        1. Get alpha lookback data for feature calculation
        2. Get risk lookback data for covariance estimation
        3. Train predictive model (if enabled)
        4. Generate forecasts
        5. Optimize with forecasts
        """
        print(f"\n{'─'*80}")
        print(f"Rebalancing at {rebalance_date.strftime('%Y-%m-%d')}")
        print(f"{'─'*80}")
        
        # Get data for alpha estimation (shorter window)
        alpha_data, alpha_start = self._get_lookback_data(
            rebalance_date,
            lookback_months=self.alpha_lookback_months
        )
        
        # Get data for risk estimation (longer window)
        risk_data, risk_start = self._get_lookback_data(
            rebalance_date,
            lookback_months=self.risk_lookback_months
        )
        
        if alpha_data.empty or risk_data.empty:
            print("  ⚠️  Insufficient data for optimization")
            return None
        
        print(f"  Alpha window: {alpha_start.strftime('%Y-%m-%d')} to {rebalance_date.strftime('%Y-%m-%d')}")
        print(f"  Risk window: {risk_start.strftime('%Y-%m-%d')} to {rebalance_date.strftime('%Y-%m-%d')}")
        
        # Calculate features from alpha window
        print("\n  📊 Calculating technical features...")
        features_df = calculate_features(alpha_data)
        
        if features_df.empty:
            print("  ⚠️  No features computed")
            return None
        
        # Get current universe of stocks
        current_tickers = alpha_data.index.get_level_values('ticker').unique().tolist()
        
        # Generate forecasts
        if self.use_predictive:
            forecast_dict = self._generate_ml_forecasts(
                features_df,
                alpha_data,
                rebalance_date,
                current_tickers
            )
        else:
            forecast_dict = None
        
        # Optimize portfolio with forecasts
        portfolio_results = self._optimize_with_forecasts(
            risk_data,
            forecast_dict,
            rebalance_date
        )
        
        if portfolio_results is None:
            return None
        
        # Track forecast quality (for next rebalance)
        if forecast_dict is not None:
            self._track_forecast_quality(forecast_dict, rebalance_date)
        
        return portfolio_results
    
    def _get_lookback_data(self, rebalance_date, lookback_months):
        """
        Get lookback data for specified window.
        
        Returns:
        --------
        data : pd.DataFrame
            Historical data
        start_date : pd.Timestamp
            Start of lookback period
        """
        # Calculate start date
        start_date = rebalance_date - pd.DateOffset(months=lookback_months)
        
        # Filter data
        mask = (
            (self.stock_data.index.get_level_values('date') < rebalance_date) &
            (self.stock_data.index.get_level_values('date') >= start_date)
        )
        
        lookback_data = self.stock_data[mask].copy()
        
        return lookback_data, start_date
    
    def _generate_ml_forecasts(self, features_df, price_data, rebalance_date, current_tickers):
        """
        Generate ML-based return forecasts.
        
        Returns:
        --------
        dict : {ticker: expected_return}
        """
        print("\n  🤖 Training predictive model...")
        
        # Initialize or reinitialize model
        if self.use_ensemble:
            # Ensemble for robustness
            model_configs = [
                {'type': 'ridge', 'alpha': 1.0},
                {'type': 'elastic_net', 'alpha': 1.0},
                {'type': 'random_forest'}
            ]
            self.alpha_model = EnsembleAlphaModel(model_configs)
        else:
            self.alpha_model = PredictiveAlphaModel(
                model_type=self.model_type,
                horizon_months=self.forecast_horizon,
                alpha=1.0
            )
        
        # Prepare training data
        # Use monthly rebalances for training
        price_df = price_data['adj close'].unstack('ticker')
        
        # Generate past rebalance dates (monthly)
        train_start = features_df.index.get_level_values('date').min()
        train_end = rebalance_date - pd.DateOffset(months=1)  # Exclude current month
        
        train_rebalance_dates = pd.date_range(
            start=train_start,
            end=train_end,
            freq='ME'  # Month-end
        )
        
        if len(train_rebalance_dates) < 6:
            print("  ⚠️  Insufficient training periods (<6 months)")
            return None
        
        X, y, dates, tickers = self.alpha_model.prepare_training_data(
            features_df,
            price_df,
            train_rebalance_dates
        )
        
        if len(X) < 30:  # Minimum 30 samples
            print(f"  ⚠️  Insufficient training samples ({len(X)})")
            return None
        
        # Train with cross-validation
        cv_results = self.alpha_model.train_with_cross_validation(X, y, dates, n_splits=3)
        
        # Store IC
        self.ic_history.append({
            'date': rebalance_date,
            'ic': cv_results['mean_ic'],
            'ic_std': cv_results['std_ic']
        })
        
        print(f"\n  ✅ Model trained. CV IC: {cv_results['mean_ic']:.4f}±{cv_results['std_ic']:.4f}")
        
        # Train on full data
        self.alpha_model.train(X, y)
        
        # Get feature importance
        if hasattr(self.alpha_model, 'get_feature_importance'):
            importance = self.alpha_model.get_feature_importance(
                feature_names=features_df.columns.tolist()
            )
            if importance is not None:
                self.feature_importance_history.append({
                    'date': rebalance_date,
                    'importance': importance.to_dict('records')
                })
        
        # Generate forecasts for current universe
        try:
            latest_date = features_df.index.get_level_values('date').max()
            features_now = features_df.loc[latest_date]
            
            # Filter to current tickers
            features_now = features_now.loc[features_now.index.isin(current_tickers)]
            
            if features_now.empty:
                print("  ⚠️  No features for current tickers")
                return None
            
            # Predict
            predictions = self.alpha_model.predict(features_now.values)
            
            # Create forecast dictionary
            forecast_dict = dict(zip(features_now.index, predictions))
            
            # Store forecasts
            self.forecast_history.append({
                'date': rebalance_date,
                'forecasts': forecast_dict.copy()
            })
            
            print(f"\n  📈 Forecasts generated for {len(forecast_dict)} stocks")
            print(f"     Mean: {np.mean(list(forecast_dict.values())):.2%}")
            print(f"     Std:  {np.std(list(forecast_dict.values())):.2%}")
            print(f"     Min:  {np.min(list(forecast_dict.values())):.2%}")
            print(f"     Max:  {np.max(list(forecast_dict.values())):.2%}")
            
            return forecast_dict
        
        except Exception as e:
            print(f"  ❌ Forecast generation failed: {str(e)}")
            return None
    
    def _optimize_with_forecasts(self, risk_data, forecast_dict, rebalance_date):
        """
        Optimize portfolio using ML forecasts and robust risk estimates.
        
        Key principle: Alpha-Risk Separation
        - Alpha (expected returns): From ML forecasts (short window)
        - Risk (covariance): From historical data (long window)
        """
        print("\n  ⚙️  Optimizing portfolio...")
        
        # Get price data for risk estimation
        price_df = risk_data['adj close'].unstack('ticker')
        
        # Remove stocks with insufficient data
        min_obs = int(0.7 * len(price_df))  # Require 70% data availability
        valid_tickers = price_df.columns[price_df.count() >= min_obs].tolist()
        
        if len(valid_tickers) < 5:
            print(f"  ⚠️  Insufficient stocks with complete data ({len(valid_tickers)})")
            return None
        
        price_df = price_df[valid_tickers]
        
        # Estimate covariance (RISK)
        try:
            # Use Ledoit-Wolf shrinkage for robustness
            S = risk_models.CovarianceShrinkage(price_df).ledoit_wolf()
        except Exception as e:
            print(f"  ⚠️  Covariance estimation failed: {str(e)}")
            # Fallback to sample covariance
            S = risk_models.sample_cov(price_df)
        
        # Estimate expected returns (ALPHA)
        if forecast_dict is not None:
            # Use ML forecasts
            mu = pd.Series({
                ticker: forecast_dict.get(ticker, 0.0)
                for ticker in valid_tickers
            })
            
            # Shift forecasts to ensure positive expected returns
            # (EfficientFrontier requires positive expected returns for max Sharpe)
            min_return = mu.min()
            if min_return < 0:
                mu = mu - min_return + 0.01  # Shift to slightly positive
            
            print(f"     Using ML forecasts (alpha)")
        else:
            # Fallback to historical means
            mu = expected_returns.mean_historical_return(price_df, frequency=252)
            print(f"     Using historical means (fallback)")
        
        # Portfolio constraints
        max_weight = self.config.get('max_weight', 0.15)  # Max 15% per stock
        min_weight = self.config.get('min_weight', 0.0)   # Long-only
        
        # Optimize
        try:
            ef = EfficientFrontier(
                mu, 
                S, 
                weight_bounds=(min_weight, max_weight)
            )
            
            # Max Sharpe ratio
            ef.max_sharpe(risk_free_rate=self.config.get('risk_free_rate', 0.05))
            
            # Get weights
            cleaned_weights = ef.clean_weights(cutoff=0.001)  # Remove tiny weights
            
            # Filter non-zero weights
            weights = {k: v for k, v in cleaned_weights.items() if v > 0.001}
            
            # Performance metrics
            perf = ef.portfolio_performance(
                risk_free_rate=self.config.get('risk_free_rate', 0.05)
            )
            
            print(f"\n  ✅ Optimization complete")
            print(f"     Allocated: {len(weights)} stocks")
            print(f"     Expected Return: {perf[0]:.2%}")
            print(f"     Volatility: {perf[1]:.2%}")
            print(f"     Sharpe Ratio: {perf[2]:.2f}")
            
            return {
                'date': rebalance_date,
                'weights': weights,
                'expected_return': float(perf[0]),
                'volatility': float(perf[1]),
                'sharpe_ratio': float(perf[2]),
                'forecast_based': forecast_dict is not None,
                'num_stocks': len(weights)
            }
        
        except Exception as e:
            print(f"  ❌ Optimization failed: {str(e)}")
            return None
    
    def _track_forecast_quality(self, forecast_dict, rebalance_date):
        """
        Track realized returns vs forecasts for IC calculation.
        
        This is checked at the NEXT rebalance to compute realized IC.
        """
        self.realized_vs_forecast.append({
            'date': rebalance_date,
            'forecasts': forecast_dict.copy()
        })
    
    def _assess_forecast_quality(self):
        """
        Assess overall forecast quality across all rebalances.
        
        Returns:
        --------
        dict : Quality metrics
        """
        if len(self.ic_history) == 0:
            return None
        
        ics = [x['ic'] for x in self.ic_history if x['ic'] is not None]
        
        if len(ics) == 0:
            return None
        
        return {
            'mean_ic': np.mean(ics),
            'median_ic': np.median(ics),
            'std_ic': np.std(ics),
            'min_ic': np.min(ics),
            'max_ic': np.max(ics),
            'positive_ic_ratio': np.mean(np.array(ics) > 0),
            'num_periods': len(ics)
        }


def optimize_portfolio_with_forecast(
    stock_data,
    forecast_dict=None,
    cluster_labels=None,
    risk_free_rate=0.05,
    min_weight=0.0,
    max_weight=0.15,
    use_shrinkage=True
):
    """
    Standalone portfolio optimization with ML forecasts.
    
    This is the core optimizer that accepts forecasts.
    
    Parameters:
    -----------
    stock_data : pd.DataFrame
        Historical stock data (multi-index: date, ticker)
    forecast_dict : dict or None
        {ticker: expected_return} from ML model
        If None, uses historical means
    cluster_labels : dict or None
        {ticker: cluster_id} for cluster-aware constraints
    risk_free_rate : float
        Risk-free rate for Sharpe calculation
    min_weight : float
        Minimum weight per stock (0 for long-only)
    max_weight : float
        Maximum weight per stock (e.g., 0.15 for 15% cap)
    use_shrinkage : bool
        Use Ledoit-Wolf shrinkage for covariance
    
    Returns:
    --------
    dict : Portfolio results
    """
    # Get price data
    price_df = stock_data['adj close'].unstack('ticker')
    
    # Risk estimation
    if use_shrinkage:
        try:
            S = risk_models.CovarianceShrinkage(price_df).ledoit_wolf()
        except:
            S = risk_models.sample_cov(price_df)
    else:
        S = risk_models.sample_cov(price_df)
    
    # Alpha estimation
    if forecast_dict is not None:
        # Use ML forecasts
        tickers = price_df.columns
        mu = pd.Series({t: forecast_dict.get(t, 0.0) for t in tickers})
        
        # Ensure positive returns
        if mu.min() < 0:
            mu = mu - mu.min() + 0.01
    else:
        # Historical means
        mu = expected_returns.mean_historical_return(price_df, frequency=252)
    
    # Optimize
    ef = EfficientFrontier(mu, S, weight_bounds=(min_weight, max_weight))
    ef.max_sharpe(risk_free_rate=risk_free_rate)
    
    cleaned_weights = ef.clean_weights(cutoff=0.001)
    performance = ef.portfolio_performance(risk_free_rate=risk_free_rate)
    
    return {
        'weights': {k: v for k, v in cleaned_weights.items() if v > 0.001},
        'expected_return': float(performance[0]),
        'volatility': float(performance[1]),
        'sharpe_ratio': float(performance[2]),
        'forecast_based': forecast_dict is not None
    }
