import React, { useState } from 'react';
import './App.css';
import StockSelector from './components/StockSelector';
import ParameterInputs from './components/ParameterInputs';
import Results from './components/Results';
import WalkForwardResults from './components/WalkForwardResults';
import { optimizePortfolio, optimizeWalkForward } from './api';

function App() {
  const [tickers, setTickers] = useState([]);
  const [parameters, setParameters] = useState({
    startDate: '2020-01-01',
    endDate: new Date().toISOString().split('T')[0],
    nClusters: 4,
    riskFreeRate: 0.05,
    initialCapital: 100000
  });
  const [benchmark, setBenchmark] = useState('NIFTY50');
  const [useWalkForward, setUseWalkForward] = useState(false);
  const [walkForwardParams, setWalkForwardParams] = useState({
    lookbackMonths: 24,
    rebalanceFreq: 'M',
    transactionCostBps: 15.0
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
      let data;
      
      if (useWalkForward) {
        // Call walk-forward endpoint
        data = await optimizeWalkForward({
          tickers,
          start_date: parameters.startDate,
          end_date: parameters.endDate,
          n_clusters: parameters.nClusters,
          risk_free_rate: parameters.riskFreeRate,
          initial_capital: parameters.initialCapital,
          benchmark: benchmark,
          max_weight: 0.25,
          lookback_months: walkForwardParams.lookbackMonths,
          rebalance_freq: walkForwardParams.rebalanceFreq,
          transaction_cost_bps: walkForwardParams.transactionCostBps
        });
      } else {
        // Call single-period endpoint
        data = await optimizePortfolio({
          tickers,
          start_date: parameters.startDate,
          end_date: parameters.endDate,
          n_clusters: parameters.nClusters,
          risk_free_rate: parameters.riskFreeRate,
          initial_capital: parameters.initialCapital,
          benchmark: benchmark,
          max_weight: 0.25
        });
      }

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

          <div className="form-group">
            <label className="form-label">
              📊 Benchmark
              <select
                value={benchmark}
                onChange={e => setBenchmark(e.target.value)}
                className="form-select"
              >
                <option value="NIFTY50">NIFTY 50</option>
                <option value="NIFTYBANK">NIFTY Bank</option>
                <option value="NIFTY500">NIFTY 500</option>
              </select>
            </label>
          </div>

          {/* Walk-Forward Mode Toggle */}
          <div className="form-group walkforward-toggle">
            <label className="form-label">
              <input
                type="checkbox"
                checked={useWalkForward}
                onChange={e => setUseWalkForward(e.target.checked)}
              />
              <span style={{marginLeft: '8px'}}>
                🔄 Use Walk-Forward Backtesting (Realistic Out-of-Sample)
              </span>
            </label>
          </div>

          {/* Walk-Forward Parameters */}
          {useWalkForward && (
            <div className="walkforward-params">
              <h3 style={{fontSize: '1.1em', marginTop: '16px', marginBottom: '12px'}}>
                ⚙️ Walk-Forward Configuration
              </h3>
              
              <div className="form-group">
                <label className="form-label">
                  Lookback Period (months)
                  <input
                    type="number"
                    value={walkForwardParams.lookbackMonths}
                    onChange={e => setWalkForwardParams({
                      ...walkForwardParams,
                      lookbackMonths: parseInt(e.target.value)
                    })}
                    min="12"
                    max="60"
                    className="form-input"
                  />
                  <small style={{display: 'block', color: '#666', marginTop: '4px'}}>
                    History used for each optimization (12-60 months)
                  </small>
                </label>
              </div>

              <div className="form-group">
                <label className="form-label">
                  Rebalancing Frequency
                  <select
                    value={walkForwardParams.rebalanceFreq}
                    onChange={e => setWalkForwardParams({
                      ...walkForwardParams,
                      rebalanceFreq: e.target.value
                    })}
                    className="form-select"
                  >
                    <option value="M">Monthly</option>
                    <option value="Q">Quarterly</option>
                    <option value="W">Weekly</option>
                  </select>
                </label>
              </div>

              <div className="form-group">
                <label className="form-label">
                  Transaction Cost (basis points)
                  <input
                    type="number"
                    value={walkForwardParams.transactionCostBps}
                    onChange={e => setWalkForwardParams({
                      ...walkForwardParams,
                      transactionCostBps: parseFloat(e.target.value)
                    })}
                    min="0"
                    max="100"
                    step="5"
                    className="form-input"
                  />
                  <small style={{display: 'block', color: '#666', marginTop: '4px'}}>
                    Cost per trade (15 bps = 0.15%)
                  </small>
                </label>
              </div>
            </div>
          )}

          <button 
            className="optimize-button"
            onClick={handleOptimize}
            disabled={loading || tickers.length < 5}
          >
            {loading ? '⏳ Optimizing...' : (
              useWalkForward ? '🚀 Run Walk-Forward Backtest' : '🚀 Optimize Portfolio'
            )}
          </button>

          {error && (
            <div className="error-message">
              <strong>Error:</strong> {error}
            </div>
          )}
        </div>

        {results && (
          results.mode === 'walkforward' ? 
            <WalkForwardResults data={results} /> :
            <Results data={results} />
        )}
      </main>

      <footer className="App-footer">
        <p>Built with React & Flask | ML-Powered Portfolio Optimization</p>
      </footer>
    </div>
  );
}

export default App;