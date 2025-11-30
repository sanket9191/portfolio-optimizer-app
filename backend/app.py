from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import traceback
from data_fetcher import fetch_stock_data, fetch_benchmark_data
from feature_engineering import calculate_features
from clustering import perform_clustering
from portfolio_optimizer import optimize_portfolio
from backtesting import backtest_strategy
from benchmark_utils import (
    compute_equity_curve_from_prices,
    compute_performance_metrics,
    compute_equal_weight_benchmark,
)
from walkforward_engine import WalkForwardEngine
from walkforward_engine_predictive import PredictiveWalkForwardEngine

app = Flask(__name__)
CORS(app)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Portfolio Optimizer API is running"})

@app.route('/api/optimize', methods=['POST'])
def optimize():
    """
    Phase 1: Single-period ex-post optimization (legacy endpoint)
    """
    try:
        data = request.json
        
        # Extract parameters
        tickers = data.get('tickers', [])
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        n_clusters = data.get('n_clusters', 4)
        risk_free_rate = data.get('risk_free_rate', 0.05)
        initial_capital = data.get('initial_capital', 100000)
        benchmark_name = data.get('benchmark', 'NIFTY50')
        max_weight = float(data.get('max_weight', 0.25))
        
        # Check if walk-forward mode is requested
        use_walkforward = data.get('use_walkforward', False)
        
        if use_walkforward:
            # Redirect to walk-forward engine
            return optimize_walkforward()
        
        if not tickers or not start_date or not end_date:
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Step 1: Fetch data
        print(f"Fetching data for {len(tickers)} stocks...")
        stock_data = fetch_stock_data(tickers, start_date, end_date)
        
        if stock_data.empty:
            return jsonify({"error": "No data fetched. Check tickers and date range."}), 400
        
        # Step 2: Calculate features
        print("Calculating technical indicators...")
        features_df = calculate_features(stock_data)
        
        # Step 3: Perform clustering
        print(f"Performing K-means clustering with {n_clusters} clusters...")
        cluster_results = perform_clustering(features_df, n_clusters)
        
        # Step 4: Build price DataFrame for optimization
        price_df = stock_data['adj close'].unstack('ticker').dropna()
        
        # Step 5: Optimize portfolio with weight constraints
        print("Optimizing portfolio...")
        portfolio_results = optimize_portfolio(
            stock_data, 
            cluster_results['labels'], 
            risk_free_rate=risk_free_rate,
            min_weight=0.0,
            max_weight=max_weight
        )
        
        # Step 6: Backtest
        print("Backtesting strategy...")
        backtest_results = backtest_strategy(
            stock_data, 
            portfolio_results['weights'],
            initial_capital=initial_capital
        )
        
        # Step 7: Compute index benchmark
        benchmark_metrics = None
        if benchmark_name in ["NIFTY50", "NIFTYBANK", "NIFTY500"]:
            print(f"Fetching {benchmark_name} benchmark...")
            benchmark_data = fetch_benchmark_data(benchmark_name, start_date, end_date)
            if benchmark_data is not None:
                benchmark_equity = compute_equity_curve_from_prices(
                    benchmark_data['adj_close'],
                    initial_capital
                )
                benchmark_metrics = compute_performance_metrics(
                    benchmark_equity,
                    risk_free_rate_annual=risk_free_rate
                )
        
        # Step 8: Compute equal-weight benchmark
        print("Computing equal-weight benchmark...")
        equal_weight_metrics = None
        equal_weight_equity = compute_equal_weight_benchmark(price_df, initial_capital)
        if equal_weight_equity is not None:
            equal_weight_metrics = compute_performance_metrics(
                equal_weight_equity,
                risk_free_rate_annual=risk_free_rate
            )
        
        response = {
            "success": True,
            "mode": "single_period",
            "clusters": cluster_results,
            "portfolio": portfolio_results,
            "backtest": backtest_results,
            "benchmark": {
                "selected": benchmark_name,
                "index": benchmark_metrics,
                "equal_weight": equal_weight_metrics
            },
            "message": "Portfolio optimization completed successfully"
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in optimize endpoint: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/optimize/walkforward', methods=['POST'])
def optimize_walkforward():
    """
    Phase 2: Walk-forward backtesting with periodic rebalancing
    """
    try:
        data = request.json
        
        # Extract parameters
        tickers = data.get('tickers', [])
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        n_clusters = data.get('n_clusters', 4)
        risk_free_rate = data.get('risk_free_rate', 0.05)
        initial_capital = data.get('initial_capital', 100000)
        benchmark_name = data.get('benchmark', 'NIFTY50')
        max_weight = float(data.get('max_weight', 0.25))
        
        # Walk-forward specific parameters
        lookback_months = int(data.get('lookback_months', 24))
        rebalance_freq = data.get('rebalance_freq', 'M')  # M=monthly, Q=quarterly
        transaction_cost_bps = float(data.get('transaction_cost_bps', 15.0))
        
        if not tickers or not start_date or not end_date:
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Validate parameters
        if lookback_months < 12:
            return jsonify({"error": "Lookback period must be at least 12 months"}), 400
        
        if rebalance_freq not in ['M', 'Q', 'W']:
            return jsonify({"error": "Rebalance frequency must be 'M' (monthly), 'Q' (quarterly), or 'W' (weekly)"}), 400
        
        # Step 1: Fetch data
        print(f"\n{'='*80}")
        print("WALK-FORWARD OPTIMIZATION REQUESTED")
        print(f"{'='*80}")
        print(f"Fetching data for {len(tickers)} stocks...")
        stock_data = fetch_stock_data(tickers, start_date, end_date)
        
        if stock_data.empty:
            return jsonify({"error": "No data fetched. Check tickers and date range."}), 400
        
        # Step 2: Configure and run walk-forward engine
        config = {
            'lookback_months': lookback_months,
            'rebalance_freq': rebalance_freq,
            'n_clusters': n_clusters,
            'risk_free_rate': risk_free_rate,
            'max_weight': max_weight,
            'transaction_cost_bps': transaction_cost_bps,
            'initial_capital': initial_capital
        }
        
        engine = WalkForwardEngine(stock_data, config)
        walkforward_results = engine.run()
        
        # Step 3: Compute benchmarks over same period
        price_df = stock_data['adj close'].unstack('ticker').dropna()
        
        # Index benchmark
        benchmark_metrics = None
        if benchmark_name in ["NIFTY50", "NIFTYBANK", "NIFTY500"]:
            print(f"\nFetching {benchmark_name} benchmark...")
            benchmark_data = fetch_benchmark_data(benchmark_name, start_date, end_date)
            if benchmark_data is not None:
                benchmark_equity = compute_equity_curve_from_prices(
                    benchmark_data['adj_close'],
                    initial_capital
                )
                benchmark_metrics = compute_performance_metrics(
                    benchmark_equity,
                    risk_free_rate_annual=risk_free_rate
                )
        
        # Equal-weight benchmark
        print("Computing equal-weight benchmark...")
        equal_weight_equity = compute_equal_weight_benchmark(price_df, initial_capital)
        equal_weight_metrics = None
        if equal_weight_equity is not None:
            equal_weight_metrics = compute_performance_metrics(
                equal_weight_equity,
                risk_free_rate_annual=risk_free_rate
            )
        
        # Build response
        response = {
            "success": True,
            "mode": "walkforward",
            "strategy": walkforward_results,
            "benchmark": {
                "selected": benchmark_name,
                "index": benchmark_metrics,
                "equal_weight": equal_weight_metrics
            },
            "config": {
                "lookback_months": lookback_months,
                "rebalance_freq": rebalance_freq,
                "transaction_cost_bps": transaction_cost_bps,
                "max_weight": max_weight
            },
            "message": "Walk-forward backtest completed successfully"
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"\nError in walk-forward optimization: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/optimize/predictive', methods=['POST'])
def optimize_predictive():
    """
    Phase 3: Predictive walk-forward with ML-based alpha forecasts
    
    This is the institutional-grade endpoint that uses machine learning
    to forecast returns and separates alpha from risk estimation.
    """
    try:
        data = request.json
        
        # Standard parameters
        tickers = data.get('tickers', [])
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        n_clusters = data.get('n_clusters', 4)
        risk_free_rate = data.get('risk_free_rate', 0.05)
        initial_capital = data.get('initial_capital', 100000)
        benchmark_name = data.get('benchmark', 'NIFTY50')
        max_weight = float(data.get('max_weight', 0.15))  # More conservative for ML
        
        # Walk-forward parameters
        rebalance_freq = data.get('rebalance_freq', 'M')
        transaction_cost_bps = float(data.get('transaction_cost_bps', 15.0))
        
        # Predictive-specific parameters
        model_type = data.get('model_type', 'ridge')  # ridge, lasso, elastic_net, random_forest, gradient_boosting
        forecast_horizon = int(data.get('forecast_horizon', 3))  # months
        use_ensemble = data.get('use_ensemble', False)
        alpha_lookback = int(data.get('alpha_lookback_months', 12))  # Short window for alpha
        risk_lookback = int(data.get('risk_lookback_months', 36))  # Long window for risk
        
        # Validation
        if not tickers or not start_date or not end_date:
            return jsonify({"error": "Missing required parameters"}), 400
        
        if model_type not in ['ridge', 'lasso', 'elastic_net', 'random_forest', 'gradient_boosting']:
            return jsonify({"error": f"Invalid model type: {model_type}"}), 400
        
        if forecast_horizon < 1 or forecast_horizon > 6:
            return jsonify({"error": "Forecast horizon must be between 1 and 6 months"}), 400
        
        if alpha_lookback < 6:
            return jsonify({"error": "Alpha lookback must be at least 6 months"}), 400
        
        if risk_lookback < 24:
            return jsonify({"error": "Risk lookback must be at least 24 months"}), 400
        
        # Step 1: Fetch data
        print(f"\n{'='*80}")
        print("PREDICTIVE ML OPTIMIZATION REQUESTED")
        print(f"{'='*80}")
        print(f"Model: {model_type.upper()}")
        print(f"Forecast Horizon: {forecast_horizon} months")
        print(f"Ensemble: {'Yes' if use_ensemble else 'No'}")
        print(f"Alpha Lookback: {alpha_lookback} months")
        print(f"Risk Lookback: {risk_lookback} months")
        print(f"{'='*80}")
        print(f"\nFetching data for {len(tickers)} stocks...")
        
        stock_data = fetch_stock_data(tickers, start_date, end_date)
        
        if stock_data.empty:
            return jsonify({"error": "No data fetched. Check tickers and date range."}), 400
        
        # Step 2: Configure predictive engine
        config = {
            'lookback_months': alpha_lookback,  # Used for feature calculation
            'rebalance_freq': rebalance_freq,
            'n_clusters': n_clusters,
            'risk_free_rate': risk_free_rate,
            'max_weight': max_weight,
            'min_weight': 0.0,  # Long-only
            'transaction_cost_bps': transaction_cost_bps,
            'initial_capital': initial_capital
        }
        
        # Step 3: Run predictive walk-forward
        engine = PredictiveWalkForwardEngine(
            stock_data,
            config,
            use_predictive=True,
            model_type=model_type,
            forecast_horizon=forecast_horizon,
            use_ensemble=use_ensemble,
            alpha_lookback_months=alpha_lookback,
            risk_lookback_months=risk_lookback
        )
        
        predictive_results = engine.run()
        
        # Step 4: Compute benchmarks
        price_df = stock_data['adj close'].unstack('ticker').dropna()
        
        # Index benchmark
        benchmark_metrics = None
        if benchmark_name in ["NIFTY50", "NIFTYBANK", "NIFTY500"]:
            print(f"\nFetching {benchmark_name} benchmark...")
            benchmark_data = fetch_benchmark_data(benchmark_name, start_date, end_date)
            if benchmark_data is not None:
                benchmark_equity = compute_equity_curve_from_prices(
                    benchmark_data['adj_close'],
                    initial_capital
                )
                benchmark_metrics = compute_performance_metrics(
                    benchmark_equity,
                    risk_free_rate_annual=risk_free_rate
                )
        
        # Equal-weight benchmark
        print("Computing equal-weight benchmark...")
        equal_weight_equity = compute_equal_weight_benchmark(price_df, initial_capital)
        equal_weight_metrics = None
        if equal_weight_equity is not None:
            equal_weight_metrics = compute_performance_metrics(
                equal_weight_equity,
                risk_free_rate_annual=risk_free_rate
            )
        
        # Step 5: Build response with ML diagnostics
        response = {
            "success": True,
            "mode": "predictive_ml",
            "strategy": predictive_results,
            "benchmark": {
                "selected": benchmark_name,
                "index": benchmark_metrics,
                "equal_weight": equal_weight_metrics
            },
            "config": {
                "model_type": model_type,
                "forecast_horizon": forecast_horizon,
                "use_ensemble": use_ensemble,
                "alpha_lookback_months": alpha_lookback,
                "risk_lookback_months": risk_lookback,
                "rebalance_freq": rebalance_freq,
                "transaction_cost_bps": transaction_cost_bps,
                "max_weight": max_weight
            },
            "message": "Predictive ML optimization completed successfully"
        }
        
        # Add ML-specific diagnostics if available
        if 'predictive_diagnostics' in predictive_results:
            response['ml_diagnostics'] = predictive_results['predictive_diagnostics']
        
        return jsonify(response)
        
    except Exception as e:
        print(f"\nError in predictive optimization: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/tickers/<index>', methods=['GET'])
def get_index_tickers(index):
    """Get list of tickers for a given index"""
    try:
        if index.upper() == 'NIFTY50':
            tickers = [
                'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
                'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
                'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'SUNPHARMA.NS',
                'TITAN.NS', 'ULTRACEMCO.NS', 'BAJFINANCE.NS', 'NESTLEIND.NS', 'WIPRO.NS'
            ]
        elif index.upper() == 'NIFTYBANK':
            tickers = [
                'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'AXISBANK.NS',
                'INDUSINDBK.NS', 'BANDHANBNK.NS', 'FEDERALBNK.NS', 'IDFCFIRSTB.NS', 'PNB.NS'
            ]
        else:
            return jsonify({"error": "Index not supported"}), 400
        
        return jsonify({"tickers": tickers})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
