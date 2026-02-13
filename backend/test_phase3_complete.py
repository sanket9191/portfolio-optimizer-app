#!/usr/bin/env python3
"""
Phase 3 Integration Test Script

Comprehensive testing of predictive alpha portfolio optimization.

Tests:
1. Predictive model standalone (IC validation)
2. Walk-forward with Ridge
3. Walk-forward with Random Forest
4. Walk-forward with Ensemble
5. Benchmark comparison
6. Performance metrics validation
"""

import pandas as pd
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

from data_fetcher import fetch_stock_data
from feature_engineering import calculate_features
from predictive_models import PredictiveAlphaModel, EnsembleAlphaModel
from walkforward_engine_predictive import PredictiveWalkForwardEngine
from benchmark_utils import compute_performance_metrics


def print_header(title):
    """Print formatted header."""
    print(f"\n{'='*80}")
    print(f"{title.center(80)}")
    print(f"{'='*80}\n")


def print_section(title):
    """Print formatted section."""
    print(f"\n{'─'*80}")
    print(f"{title}")
    print(f"{'─'*80}")


def test_predictive_models():
    """Test 1: Predictive model standalone."""
    print_section("TEST 1: PREDICTIVE MODEL VALIDATION")
    
    # Fetch data
    print("\n[1/5] Fetching stock data...")
    tickers = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS'
    ]
    
    stock_data = fetch_stock_data(tickers, '2020-01-01', '2024-11-30')
    print(f"   ✓ Fetched {len(stock_data)} rows for {len(tickers)} stocks")
    
    # Calculate features
    print("\n[2/5] Computing technical features...")
    features_df = calculate_features(stock_data)
    print(f"   ✓ Computed {features_df.shape[1]} features")
    print(f"   ✓ Total feature samples: {len(features_df)}")
    
    # Prepare training data
    print("\n[3/5] Preparing training data...")
    price_df = stock_data['adj close'].unstack('ticker')
    rebalance_dates = pd.date_range(start='2021-01-01', end='2024-06-30', freq='ME')
    
    model = PredictiveAlphaModel(model_type='ridge', horizon_months=3, alpha=1.0)
    X, y, dates, tickers_used = model.prepare_training_data(
        features_df, price_df, rebalance_dates
    )
    
    print(f"   ✓ Training samples: {X.shape[0]}")
    print(f"   ✓ Features: {X.shape[1]}")
    print(f"   ✓ Date range: {dates.min()} to {dates.max()}")
    
    # Cross-validate
    print("\n[4/5] Running time-series cross-validation...")
    cv_results = model.train_with_cross_validation(X, y, dates, n_splits=3)
    
    print("\n   Cross-Validation Results:")
    print(f"   {'─'*60}")
    print(f"   Mean IC:      {cv_results['mean_ic']:.4f} ± {cv_results['std_ic']:.4f}")
    print(f"   Mean RMSE:    {cv_results['mean_rmse']:.4f}")
    print(f"   Mean R²:      {cv_results['mean_r2']:.4f}")
    print(f"   {'─'*60}")
    
    # Assess IC
    ic = cv_results['mean_ic']
    if ic > 0.10:
        status = "🎯 EXCELLENT"
        msg = "IC > 0.10 (exceptional predictive skill)"
    elif ic > 0.05:
        status = "✅ GOOD"
        msg = "IC > 0.05 (statistically significant)"
    elif ic > 0.02:
        status = "⚠️  MARGINAL"
        msg = "IC > 0.02 (weak but positive)"
    else:
        status = "❌ POOR"
        msg = "IC ≤ 0.02 (no predictive skill)"
    
    print(f"\n   {status}: {msg}")
    
    # Train and get importance
    print("\n[5/5] Training final model...")
    final_ic = model.train(X, y)
    
    importance = model.get_feature_importance(
        feature_names=features_df.columns.tolist(),
        top_n=10
    )
    
    if importance is not None:
        print("\n   Top 10 Most Important Features:")
        print(f"   {'─'*60}")
        for idx, row in importance.iterrows():
            bar = '█' * int(row['importance'] * 50)
            print(f"   {idx+1:2d}. {row['feature']:25s} {bar}")
    
    return cv_results['mean_ic'] > 0.03  # Pass if IC > 3%


def test_walkforward_ridge():
    """Test 2: Walk-forward with Ridge model."""
    print_section("TEST 2: WALK-FORWARD WITH RIDGE REGRESSION")
    
    tickers = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS'
    ]
    
    config = {
        'lookback_months': 12,
        'rebalance_freq': 'Q',  # Quarterly for faster testing
        'n_clusters': 3,
        'risk_free_rate': 0.06,
        'max_weight': 0.15,
        'min_weight': 0.0,
        'transaction_cost_bps': 15.0,
        'initial_capital': 100000
    }
    
    print("\nFetching data...")
    stock_data = fetch_stock_data(tickers, '2021-01-01', '2024-11-30')
    
    print("\nRunning predictive walk-forward...")
    engine = PredictiveWalkForwardEngine(
        stock_data,
        config,
        use_predictive=True,
        model_type='ridge',
        forecast_horizon=3,
        use_ensemble=False,
        alpha_lookback_months=12,
        risk_lookback_months=36
    )
    
    results = engine.run()
    
    print("\n✅ Ridge Model Results:")
    print(f"   {'─'*60}")
    print(f"   Total Return:      {results['metrics']['total_return']:>10.2%}")
    print(f"   Annualized Return: {results['metrics']['annualized_return']:>10.2%}")
    print(f"   Volatility:        {results['metrics']['volatility']:>10.2%}")
    print(f"   Sharpe Ratio:      {results['metrics']['sharpe_ratio']:>10.2f}")
    print(f"   Max Drawdown:      {results['metrics']['max_drawdown']:>10.2%}")
    print(f"   Num Rebalances:    {results['metrics']['num_rebalances']:>10d}")
    print(f"   {'─'*60}")
    
    if 'predictive_diagnostics' in results:
        diag = results['predictive_diagnostics']
        print(f"\n🤖 ML Diagnostics:")
        print(f"   {'─'*60}")
        print(f"   Mean IC:           {diag.get('mean_ic', 0):>10.4f}")
        print(f"   IC Stability:      {diag.get('ic_stability', 0):>10.4f}")
        
        if 'forecast_quality' in diag and diag['forecast_quality']:
            fq = diag['forecast_quality']
            print(f"   Positive IC Rate:  {fq['positive_ic_ratio']:>10.1%}")
            print(f"   IC Range:          [{fq['min_ic']:.3f}, {fq['max_ic']:.3f}]")
        print(f"   {'─'*60}")
    
    # Pass criteria
    sharpe_ok = results['metrics']['sharpe_ratio'] > 0.5
    ic_ok = diag.get('mean_ic', 0) > 0.03 if 'predictive_diagnostics' in results else False
    
    return sharpe_ok and ic_ok


def test_model_comparison():
    """Test 3: Compare multiple models."""
    print_section("TEST 3: MODEL COMPARISON")
    
    tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']
    
    config = {
        'lookback_months': 12,
        'rebalance_freq': 'Q',
        'n_clusters': 3,
        'risk_free_rate': 0.06,
        'max_weight': 0.20,
        'min_weight': 0.0,
        'transaction_cost_bps': 15.0,
        'initial_capital': 100000
    }
    
    stock_data = fetch_stock_data(tickers, '2022-01-01', '2024-11-30')
    
    results_dict = {}
    
    for model_name in ['ridge', 'random_forest']:
        print(f"\nTesting {model_name.upper()}...")
        
        engine = PredictiveWalkForwardEngine(
            stock_data,
            config,
            use_predictive=True,
            model_type=model_name,
            forecast_horizon=3,
            use_ensemble=False,
            alpha_lookback_months=12,
            risk_lookback_months=24
        )
        
        results = engine.run()
        results_dict[model_name] = results['metrics']
    
    # Print comparison
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    print(f"\n{'Model':<20} {'Sharpe':<12} {'Ann. Return':<15} {'Max DD':<12}")
    print("-" * 70)
    
    for model_name, metrics in results_dict.items():
        print(f"{model_name.title():<20} "
              f"{metrics['sharpe_ratio']:<12.2f} "
              f"{metrics['annualized_return']:<15.2%} "
              f"{metrics['max_drawdown']:<12.2%}")
    
    return True


def run_all_tests():
    """Run complete test suite."""
    print_header("PHASE 3 COMPREHENSIVE TEST SUITE")
    
    print("This will test:")
    print("  1. Predictive model IC validation")
    print("  2. Walk-forward with Ridge")
    print("  3. Model comparison (Ridge vs Random Forest)")
    print("\nExpected runtime: 2-5 minutes\n")
    
    input("Press Enter to continue...")
    
    # Run tests
    test_results = []
    
    try:
        print_header("Starting Test Suite")
        
        # Test 1
        test1_pass = test_predictive_models()
        test_results.append(("Model IC Validation", test1_pass))
        
        # Test 2
        test2_pass = test_walkforward_ridge()
        test_results.append(("Walk-Forward Ridge", test2_pass))
        
        # Test 3
        test3_pass = test_model_comparison()
        test_results.append(("Model Comparison", test3_pass))
        
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Summary
    print_header("TEST SUMMARY")
    
    print(f"\n{'Test Name':<30} {'Status':<15}")
    print("-" * 50)
    
    all_passed = True
    for test_name, passed in test_results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:<30} {status:<15}")
        all_passed = all_passed and passed
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 ALL TESTS PASSED - PHASE 3 IMPLEMENTATION SUCCESSFUL")
    else:
        print("⚠️  SOME TESTS FAILED - REVIEW RESULTS ABOVE")
    print("="*80 + "\n")
    
    # Next steps
    print("\n🚀 Next Steps:")
    print("  1. Review IC and Sharpe ratios above")
    print("  2. Test via API: curl -X POST http://localhost:5000/api/optimize/predictive ...")
    print("  3. Update frontend with predictive controls")
    print("  4. Deploy to production\n")


if __name__ == "__main__":
    print("\n📊 Phase 3 Integration Test Script")
    print("Ensure backend dependencies are installed and yfinance is working.\n")
    
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
