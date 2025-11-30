import React, { useState } from 'react';
import './App.css';
import StockSelector from './components/StockSelector';
import ParameterInputs from './components/ParameterInputs';
import Results from './components/Results';
import { optimizePortfolio } from './api';

function App() {
  const [tickers, setTickers] = useState([]);
  const [parameters, setParameters] = useState({
    startDate: '2020-01-01',
    endDate: new Date().toISOString().split('T')[0],
    nClusters: 4,
    riskFreeRate: 0.05
  });
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleOptimize = async () => {
    if (tickers.length < 5) {
      setError('Please select at least 5 stocks');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const data = await optimizePortfolio({
        tickers,
        start_date: parameters.startDate,
        end_date: parameters.endDate,
        n_clusters: parameters.nClusters,
        risk_free_rate: parameters.riskFreeRate
      });

      setResults(data);
    } catch (err) {
      setError(err.message || 'An error occurred during optimization');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🎯 ML Portfolio Optimizer</h1>
        <p>Adaptive Portfolio Optimization via Unsupervised Learning</p>
      </header>

      <main className="App-main">
        <div className="input-section">
          <StockSelector 
            selectedTickers={tickers}
            onTickersChange={setTickers}
          />

          <ParameterInputs
            parameters={parameters}
            onParametersChange={setParameters}
          />

          <button 
            className="optimize-button"
            onClick={handleOptimize}
            disabled={loading || tickers.length < 5}
          >
            {loading ? '⏳ Optimizing...' : '🚀 Optimize Portfolio'}
          </button>

          {error && (
            <div className="error-message">
              <strong>Error:</strong> {error}
            </div>
          )}
        </div>

        {results && <Results data={results} />}
      </main>

      <footer className="App-footer">
        <p>Built with React & Flask | ML-Powered Portfolio Optimization</p>
      </footer>
    </div>
  );
}

export default App;