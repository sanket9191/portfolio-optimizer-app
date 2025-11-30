"""Predictive Walk-Forward Portfolio Optimization Engine

Institutional-grade implementation with:
- Machine learning return forecasts (alpha)
- Robust risk estimation (covariance)
- Proper alpha-risk separation
- Time-series cross-validation
- IC tracking and diagnostics
- Transaction cost modeling
- Position and risk controls
- Forward-looking portfolio recommendations
"""

import numpy as np
import pandas as pd
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

from walkforward_engine import WalkForwardEngine
from predictive_models import PredictiveAlphaModel, EnsembleAlphaModel
from feature_engineering import calculate_features
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier


class PredictiveWalkForwardEngine(WalkForwardEngine):
    """Extended walk-forward engine with predictive alpha models and forward recommendations."""
    
    def __init__(self, stock_data, config, use_predictive=True, model_type='ridge',
                 forecast_horizon=3, use_ensemble=False, alpha_lookback_months=12,
                 risk_lookback_months=36):
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
        """Execute predictive walk-forward backtest with forward portfolio generation."""
        # Call parent's run method
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
            results['ml_diagnostics'] = results['predictive_diagnostics']
        
        # 🆕 Generate forward-looking portfolio for next period
        if self.use_predictive:
            forward_portfolio = self._generate_forward_portfolio()
            if forward_portfolio:
                results['forward_portfolio'] = forward_portfolio
                print(f"\n{'='*80}")
                print("📈 FORWARD-LOOKING PORTFOLIO GENERATED")
                print(f"{'='*80}")
                print(f"Next Period Portfolio ({len(forward_portfolio['weights'])} stocks):")
                for ticker, weight in sorted(forward_portfolio['weights'].items(), key=lambda x: x[1], reverse=True):
                    print(f"  {ticker:15s} {weight:>7.2%}")
                print(f"\nExpected Return: {forward_portfolio['expected_return']:.2%}")
                print(f"Expected Volatility: {forward_portfolio['volatility']:.2%}")
                print(f"Expected Sharpe: {forward_portfolio['sharpe_ratio']:.2f}")
                print(f"Forecast Horizon: {self.forecast_horizon} months")
                print(f"{'='*80}\n")
        
        return results
    
    def _generate_forward_portfolio(self):
        """
        Generate forward-looking portfolio recommendation for the NEXT period.
        
        This is the key deliverable: "What should I invest in NOW?"
        
        Returns:
        --------
        dict : {
            'weights': {ticker: weight},
            'forecasts': {ticker: expected_return},
            'expected_return': float,
            'volatility': float,
            'sharpe_ratio': float,
            'recommendation_date': str,
            'valid_for_period': str (e.g., "2025-12 to 2026-02")
        }
        """
        try:
            print(f"\n{'─'*80}")
            print("🔮 GENERATING FORWARD PORTFOLIO RECOMMENDATION")
            print(f"{'─'*80}")
            
            # Use latest available date
            latest_date = self.dates.max()
            print(f"  As of: {latest_date.strftime('%Y-%m-%d')}")
            
            # Get data for alpha estimation
            alpha_data, alpha_start = self._get_lookback_data(
                latest_date,
                lookback_months=self.alpha_lookback_months
            )
            
            # Get data for risk estimation
            risk_data, risk_start = self._get_lookback_data(
                latest_date,
                lookback_months=self.risk_lookback_months
            )
            
            if alpha_data.empty or risk_data.empty:
                print("  ⚠️  Insufficient data for forward portfolio")
                return None
            
            # Calculate features
            print("\n  📊 Computing features on latest data...")
            features_df = calculate_features(alpha_data)
            
            if features_df.empty:
                print("  ⚠️  Feature calculation failed")
                return None
            
            # Get current universe
            current_tickers = alpha_data.index.get_level_values('ticker').unique().tolist()
            
            # Generate ML forecasts
            forecast_dict = self._generate_ml_forecasts(
                features_df,
                alpha_data,
                latest_date,
                current_tickers
            )
            
            if forecast_dict is None:
                print("  ⚠️  Forecast generation failed")
                return None
            
            # Optimize portfolio with forecasts
            print("\n  ⚙️  Optimizing forward portfolio...")
            
            price_df = risk_data['adj close'].unstack('ticker')
            min_obs = int(0.7 * len(price_df))
            valid_tickers = price_df.columns[price_df.count() >= min_obs].tolist()
            
            if len(valid_tickers) < 5:
                print(f"  ⚠️  Insufficient stocks ({len(valid_tickers)})")
                return None
            
            price_df = price_df[valid_tickers]
            
            # Estimate covariance
            try:
                S = risk_models.CovarianceShrinkage(price_df).ledoit_wolf()
            except:
                S = risk_models.sample_cov(price_df)
            
            # Use ML forecasts
            mu = pd.Series({ticker: forecast_dict.get(ticker, 0.0) for ticker in valid_tickers})
            
            # Shift to positive
            min_return = mu.min()
            if min_return < 0:
                mu = mu - min_return + 0.01
            
            # Optimize
            max_weight = self.config.get('max_weight', 0.17)
            min_weight = self.config.get('min_weight', 0.0)
            
            ef = EfficientFrontier(mu, S, weight_bounds=(min_weight, max_weight))
            ef.max_sharpe(risk_free_rate=self.config.get('risk_free_rate', 0.05))
            
            cleaned_weights = ef.clean_weights(cutoff=0.001)
            weights = {k: v for k, v in cleaned_weights.items() if v > 0.001}
            
            perf = ef.portfolio_performance(risk_free_rate=self.config.get('risk_free_rate', 0.05))
            
            # Calculate valid period
            next_month_start = latest_date + pd.DateOffset(months=1)
            forecast_end = next_month_start + pd.DateOffset(months=self.forecast_horizon)
            
            return {
                'weights': weights,
                'forecasts': {k: forecast_dict.get(k, 0.0) for k in weights.keys()},
                'expected_return': float(perf[0]),
                'volatility': float(perf[1]),
                'sharpe_ratio': float(perf[2]),
                'recommendation_date': latest_date.strftime('%Y-%m-%d'),
                'valid_for_period': f"{next_month_start.strftime('%Y-%m')} to {forecast_end.strftime('%Y-%m')}",
                'n_stocks': len(weights),
                'forecast_horizon_months': self.forecast_horizon
            }
            
        except Exception as e:
            print(f"  ❌ Forward portfolio generation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def optimize_at_date(self, rebalance_date):
        """Override parent method to use predictive forecasts."""
        if not self.use_predictive:
            return super().optimize_at_date(rebalance_date)
        
        print(f"\n{'─'*80}")
        print(f"Rebalancing at {rebalance_date.strftime('%Y-%m-%d')} (PREDICTIVE MODE)")
        print(f"{'─'*80}")
        
        alpha_data, alpha_start = self._get_lookback_data(rebalance_date, lookback_months=self.alpha_lookback_months)
        risk_data, risk_start = self._get_lookback_data(rebalance_date, lookback_months=self.risk_lookback_months)
        
        if alpha_data.empty or risk_data.empty:
            print("  ⚠️  Insufficient data for optimization")
            return None
        
        print(f"  Alpha window: {alpha_start.strftime('%Y-%m-%d')} to {rebalance_date.strftime('%Y-%m-%d')}")
        print(f"  Risk window: {risk_start.strftime('%Y-%m-%d')} to {rebalance_date.strftime('%Y-%m-%d')}")
        
        print("\n  📊 Calculating technical features...")
        features_df = calculate_features(alpha_data)
        
        if features_df.empty:
            print("  ⚠️  No features computed")
            return None
        
        current_tickers = alpha_data.index.get_level_values('ticker').unique().tolist()
        forecast_dict = self._generate_ml_forecasts(features_df, alpha_data, rebalance_date, current_tickers)
        
        if forecast_dict is None:
            print("  ⚠️  Falling back to historical means")
            return super().optimize_at_date(rebalance_date)
        
        portfolio_results = self._optimize_with_forecasts(risk_data, forecast_dict, rebalance_date)
        return portfolio_results
    
    def _get_lookback_data(self, rebalance_date, lookback_months):
        """Get lookback data for specified window."""
        start_date = rebalance_date - pd.DateOffset(months=lookback_months)
        mask = (
            (self.stock_data.index.get_level_values('date') < rebalance_date) &
            (self.stock_data.index.get_level_values('date') >= start_date)
        )
        lookback_data = self.stock_data[mask].copy()
        return lookback_data, start_date
    
    def _generate_ml_forecasts(self, features_df, price_data, rebalance_date, current_tickers):
        """Generate ML-based return forecasts."""
        print("\n  🤖 Training predictive model...")
        
        if self.use_ensemble:
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
        
        price_df = price_data['adj close'].unstack('ticker')
        train_start = features_df.index.get_level_values('date').min()
        train_end = rebalance_date - pd.DateOffset(months=1)
        
        train_rebalance_dates = pd.date_range(start=train_start, end=train_end, freq='ME')
        
        if len(train_rebalance_dates) < 6:
            print("  ⚠️  Insufficient training periods (<6 months)")
            return None
        
        try:
            X, y, dates, tickers = self.alpha_model.prepare_training_data(features_df, price_df, train_rebalance_dates)
            
            if len(X) < 30:
                print(f"  ⚠️  Insufficient training samples ({len(X)})")
                return None
            
            cv_results = self.alpha_model.train_with_cross_validation(X, y, dates, n_splits=3)
            
            self.ic_history.append({
                'date': rebalance_date,
                'ic': cv_results['mean_ic'],
                'ic_std': cv_results['std_ic']
            })
            
            print(f"\n  ✅ Model trained. CV IC: {cv_results['mean_ic']:.4f}±{cv_results['std_ic']:.4f}")
            
            self.alpha_model.train(X, y)
            
            if hasattr(self.alpha_model, 'get_feature_importance'):
                importance = self.alpha_model.get_feature_importance(feature_names=features_df.columns.tolist())
                if importance is not None:
                    self.feature_importance_history.append({
                        'date': rebalance_date,
                        'importance': importance.to_dict('records')
                    })
            
            latest_date = features_df.index.get_level_values('date').max()
            features_now = features_df.loc[latest_date]
            features_now = features_now.loc[features_now.index.isin(current_tickers)]
            
            if features_now.empty:
                print("  ⚠️  No features for current tickers")
                return None
            
            predictions = self.alpha_model.predict(features_now.values)
            forecast_dict = dict(zip(features_now.index, predictions))
            
            self.forecast_history.append({'date': rebalance_date, 'forecasts': forecast_dict.copy()})
            
            print(f"\n  📈 Forecasts generated for {len(forecast_dict)} stocks")
            print(f"     Mean: {np.mean(list(forecast_dict.values())):.2%}")
            print(f"     Std:  {np.std(list(forecast_dict.values())):.2%}")
            print(f"     Min:  {np.min(list(forecast_dict.values())):.2%}")
            print(f"     Max:  {np.max(list(forecast_dict.values())):.2%}")
            
            return forecast_dict
        
        except Exception as e:
            print(f"  ❌ Forecast generation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _optimize_with_forecasts(self, risk_data, forecast_dict, rebalance_date):
        """Optimize portfolio using ML forecasts and robust risk estimates."""
        print("\n  ⚙️  Optimizing portfolio...")
        
        price_df = risk_data['adj close'].unstack('ticker')
        min_obs = int(0.7 * len(price_df))
        valid_tickers = price_df.columns[price_df.count() >= min_obs].tolist()
        
        if len(valid_tickers) < 5:
            print(f"  ⚠️  Insufficient stocks with complete data ({len(valid_tickers)})")
            return None
        
        price_df = price_df[valid_tickers]
        
        try:
            S = risk_models.CovarianceShrinkage(price_df).ledoit_wolf()
        except Exception as e:
            print(f"  ⚠️  Shrinkage failed, using sample covariance: {str(e)}")
            S = risk_models.sample_cov(price_df)
        
        mu = pd.Series({ticker: forecast_dict.get(ticker, 0.0) for ticker in valid_tickers})
        
        min_return = mu.min()
        if min_return < 0:
            mu = mu - min_return + 0.01
        
        print(f"     Using ML forecasts (alpha)")
        
        max_weight = self.config.get('max_weight', 0.17)
        min_weight = self.config.get('min_weight', 0.0)
        
        try:
            ef = EfficientFrontier(mu, S, weight_bounds=(min_weight, max_weight))
            ef.max_sharpe(risk_free_rate=self.config.get('risk_free_rate', 0.05))
            
            cleaned_weights = ef.clean_weights(cutoff=0.001)
            weights = {k: v for k, v in cleaned_weights.items() if v > 0.001}
            
            perf = ef.portfolio_performance(risk_free_rate=self.config.get('risk_free_rate', 0.05))
            
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
                'n_stocks': len(weights),
                'clusters': {}
            }
        
        except Exception as e:
            print(f"  ❌ Optimization failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _assess_forecast_quality(self):
        """Assess overall forecast quality across all rebalances."""
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
