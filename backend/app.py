from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import traceback
from data_fetcher import fetch_stock_data
from feature_engineering import calculate_features
from clustering import perform_clustering
from portfolio_optimizer import optimize_portfolio
from backtesting import backtest_strategy

app = Flask(__name__)
CORS(app)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Portfolio Optimizer API is running"})

@app.route('/api/optimize', methods=['POST'])
def optimize():
    try:
        data = request.json
        
        # Extract parameters
        tickers = data.get('tickers', [])
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        n_clusters = data.get('n_clusters', 4)
        risk_free_rate = data.get('risk_free_rate', 0.05)
        
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
        
        # Step 4: Optimize portfolio
        print("Optimizing portfolio...")
        portfolio_results = optimize_portfolio(
            stock_data, 
            cluster_results['labels'], 
            risk_free_rate
        )
        
        # Step 5: Backtest
        print("Backtesting strategy...")
        backtest_results = backtest_strategy(
            stock_data, 
            portfolio_results['weights']
        )
        
        response = {
            "success": True,
            "clusters": cluster_results,
            "portfolio": portfolio_results,
            "backtest": backtest_results,
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