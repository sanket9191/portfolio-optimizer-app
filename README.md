# 🎯 Portfolio Optimizer - ML-Based Portfolio Optimization

A full-stack web application that uses **Unsupervised Machine Learning** (K-means clustering) and **Modern Portfolio Theory** to optimize stock portfolios with Maximum Sharpe Ratio.

## ✨ Features

- **Automated Stock Selection**: Load NIFTY 50, NIFTY Bank, or add custom tickers
- **Technical Indicators**: RSI, Bollinger Bands, ATR, MACD, Garman-Klass Volatility
- **K-Means Clustering**: Groups similar stocks based on features
- **Portfolio Optimization**: Efficient Frontier with Max Sharpe Ratio
- **Backtesting**: Historical performance analysis
- **Interactive Visualizations**: Charts for portfolio performance and allocation

## 💻 Tech Stack

### Backend
- **Flask**: RESTful API
- **pandas & numpy**: Data processing
- **yfinance**: Stock data fetching
- **scikit-learn**: K-means clustering
- **PyPortfolioOpt**: Portfolio optimization
- **pandas-ta**: Technical indicators

### Frontend
- **React**: UI framework
- **Recharts**: Data visualization
- **Axios**: API communication

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 14+
- pip & npm

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Flask server
python app.py
```

Backend will run on `http://localhost:5000`

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start React development server
npm start
```

Frontend will run on `http://localhost:3000`

## 📚 How It Works

### 1. Data Collection
- Fetches historical stock data from Yahoo Finance
- Downloads OHLCV data for selected tickers

### 2. Feature Engineering
Calculates technical indicators:
- **Garman-Klass Volatility**: Range-based volatility measure
- **RSI (Relative Strength Index)**: Momentum indicator
- **Bollinger Bands**: Volatility bands
- **ATR (Average True Range)**: Volatility measure
- **MACD**: Trend-following momentum indicator
- **Returns**: 1m, 2m, 3m, 6m, 9m, 12m horizons

### 3. K-Means Clustering
- Groups stocks with similar characteristics
- Identifies patterns in technical indicators
- Helps diversification across different stock behaviors

### 4. Portfolio Optimization
- Uses **Efficient Frontier** theory
- Maximizes **Sharpe Ratio** (risk-adjusted returns)
- Calculates optimal weights for each stock

### 5. Backtesting
- Tests portfolio performance on historical data
- Calculates:
  - Total Return
  - Annualized Return
  - Volatility
  - Sharpe Ratio
  - Maximum Drawdown

## 📈 API Endpoints

### `POST /api/optimize`
Optimize portfolio based on parameters

**Request Body:**
```json
{
  "tickers": ["RELIANCE.NS", "TCS.NS", ...],
  "start_date": "2020-01-01",
  "end_date": "2024-12-31",
  "n_clusters": 4,
  "risk_free_rate": 0.05
}
```

### `GET /api/tickers/{index}`
Get list of tickers for an index (NIFTY50, NIFTYBANK)

## 📋 Usage Example

1. **Select Stocks**
   - Click "Load NIFTY 50" or add custom tickers
   - Minimum 5 stocks required

2. **Set Parameters**
   - Choose date range (e.g., last 3 years)
   - Set number of clusters (2-10)
   - Set risk-free rate (default 5%)

3. **Optimize**
   - Click "🚀 Optimize Portfolio"
   - Wait for processing (15-30 seconds)

4. **View Results**
   - Portfolio allocation (weights)
   - Expected return & volatility
   - Sharpe ratio
   - Backtesting performance
   - Cluster analysis

## 🛠️ Configuration

### Environment Variables

Create `.env` file in frontend directory:
```
REACT_APP_API_URL=http://localhost:5000
```

## 💡 Key Concepts

### Sharpe Ratio
```
Sharpe = (Portfolio Return - Risk-Free Rate) / Portfolio Volatility
```
Higher Sharpe = Better risk-adjusted returns

### Efficient Frontier
Set of optimal portfolios offering:
- Highest expected return for a given risk level
- Lowest risk for a given return level

### K-Means Clustering
Groups stocks into clusters based on:
- Technical indicators
- Volatility patterns
- Momentum characteristics

## 📝 Project Structure

```
portfolio-optimizer-app/
├── backend/
│   ├── app.py                 # Flask API
│   ├── data_fetcher.py        # Stock data fetching
│   ├── feature_engineering.py # Technical indicators
│   ├── clustering.py          # K-means implementation
│   ├── portfolio_optimizer.py # Efficient Frontier
│   ├── backtesting.py         # Performance analysis
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── StockSelector.js
│   │   │   ├── ParameterInputs.js
│   │   │   └── Results.js
│   │   ├── App.js
│   │   └── api.js
│   └── package.json
│
└── README.md
```

## ⚠️ Limitations

- Requires stable internet for stock data fetching
- Historical data may have gaps
- Past performance doesn't guarantee future results
- Assumes no transaction costs

## 🔮 Future Enhancements

- [ ] Real-time portfolio tracking
- [ ] Multiple optimization objectives (min volatility, max return)
- [ ] Sector constraints
- [ ] Transaction cost modeling
- [ ] Advanced clustering algorithms (DBSCAN, Hierarchical)
- [ ] Monte Carlo simulation
- [ ] Risk parity allocation
- [ ] Integration with broker APIs

## 💬 Support

For issues or questions:
- Open an issue on GitHub
- Email: your-email@example.com

## 📜 License

MIT License - feel free to use for personal or commercial projects

## 🚀 Deployment

### Backend (Heroku/Render)
```bash
gunicorn app:app
```

### Frontend (Vercel/Netlify)
```bash
npm run build
```

---

**Built with ❤️ using React, Flask, and Machine Learning**

**Disclaimer**: This is an educational project. Always consult with a financial advisor before making investment decisions.