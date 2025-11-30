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
  
  // Phase 3: Predictive ML parameters
  const [usePredictive, setUsePredictive] = useState(false);
  const [predictiveParams, setPredictiveParams] = useState({
    modelType: 'ridge',
    forecastHorizon: 3,
    useEnsemble: false,
    alphaLookback: 12,
    riskLookback: 36
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
        // Call walk-forward endpoint with optional predictive params
        const payload = {
          tickers,
          start_date: parameters.startDate,
          end_date: parameters.endDate,
          n_clusters: parameters.nClusters,
          risk_free_rate: parameters.riskFreeRate,
          initial_capital: parameters.initialCapital,
          benchmark: benchmark,
          max_weight: 0.17,  // VP-approved conservative limit
          lookback_months: walkForwardParams.lookbackMonths,
          rebalance_freq: walkForwardParams.rebalanceFreq,
          transaction_cost_bps: walkForwardParams.transactionCostBps,
          // Phase 3 parameters
          use_predictive: usePredictive,
          ...(usePredictive && {
            model_type: predictiveParams.modelType,
            forecast_horizon: predictiveParams.forecastHorizon,
            use_ensemble: predictiveParams.useEnsemble,
            alpha_lookback_months: predictiveParams.alphaLookback,
            risk_lookback_months: predictiveParams.riskLookback
          })
        };
        
        data = await optimizeWalkForward(payload);
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
          max_weight: 0.17
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
        <p>Institutional-Grade Portfolio Optimization with Machine Learning</p>
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
          <div className="mode-toggle-container">
            <div className="toggle-card walkforward-toggle">
              <label className="toggle-label">
                <input
                  type="checkbox"
                  checked={useWalkForward}
                  onChange={e => setUseWalkForward(e.target.checked)}
                  className="toggle-checkbox"
                />
                <span className="toggle-switch"></span>
                <span className="toggle-text">
                  <strong>🔄 Walk-Forward Backtesting</strong>
                  <small>Realistic out-of-sample testing with periodic rebalancing</small>
                </span>
              </label>
            </div>
          </div>

          {/* Walk-Forward Parameters */}
          {useWalkForward && (
            <div className="config-panel walkforward-panel">
              <h3 className="panel-title">⚙️ Walk-Forward Configuration</h3>
              
              <div className="form-grid">
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
                    <small className="form-hint">
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
                    Transaction Cost (bps)
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
                    <small className="form-hint">
                      Per-trade cost (15 bps = 0.15%)
                    </small>
                  </label>
                </div>
              </div>

              {/* Phase 3: Predictive ML Toggle */}
              <div className="predictive-toggle-container">
                <div className="toggle-card predictive-toggle">
                  <label className="toggle-label">
                    <input
                      type="checkbox"
                      checked={usePredictive}
                      onChange={e => setUsePredictive(e.target.checked)}
                      className="toggle-checkbox"
                    />
                    <span className="toggle-switch predictive-switch"></span>
                    <span className="toggle-text">
                      <strong>🔮 Predictive ML Alpha</strong>
                      <small>Use machine learning to forecast returns (Phase 3)</small>
                    </span>
                  </label>
                </div>
              </div>

              {/* Phase 3: Predictive Parameters */}
              {usePredictive && (
                <div className="predictive-panel">
                  <h3 className="panel-title">🤖 Machine Learning Configuration</h3>
                  
                  <div className="form-grid">
                    <div className="form-group">
                      <label className="form-label">
                        ML Model Type
                        <select
                          value={predictiveParams.modelType}
                          onChange={e => setPredictiveParams({
                            ...predictiveParams,
                            modelType: e.target.value
                          })}
                          className="form-select"
                        >
                          <option value="ridge">Ridge Regression (Stable)</option>
                          <option value="lasso">LASSO (Feature Selection)</option>
                          <option value="elastic_net">Elastic Net (Balanced)</option>
                          <option value="random_forest">Random Forest (Non-Linear)</option>
                          <option value="gradient_boosting">Gradient Boosting (Advanced)</option>
                        </select>
                        <small className="form-hint">
                          Ridge recommended for stability
                        </small>
                      </label>
                    </div>

                    <div className="form-group">
                      <label className="form-label">
                        Forecast Horizon (months)
                        <input
                          type="number"
                          value={predictiveParams.forecastHorizon}
                          onChange={e => setPredictiveParams({
                            ...predictiveParams,
                            forecastHorizon: parseInt(e.target.value)
                          })}
                          min="1"
                          max="6"
                          className="form-input"
                        />
                        <small className="form-hint">
                          Prediction period (1-6 months)
                        </small>
                      </label>
                    </div>

                    <div className="form-group">
                      <label className="form-label">
                        Alpha Lookback (months)
                        <input
                          type="number"
                          value={predictiveParams.alphaLookback}
                          onChange={e => setPredictiveParams({
                            ...predictiveParams,
                            alphaLookback: parseInt(e.target.value)
                          })}
                          min="6"
                          max="24"
                          className="form-input"
                        />
                        <small className="form-hint">
                          Short window for signals
                        </small>
                      </label>
                    </div>

                    <div className="form-group">
                      <label className="form-label">
                        Risk Lookback (months)
                        <input
                          type="number"
                          value={predictiveParams.riskLookback}
                          onChange={e => setPredictiveParams({
                            ...predictiveParams,
                            riskLookback: parseInt(e.target.value)
                          })}
                          min="24"
                          max="60"
                          className="form-input"
                        />
                        <small className="form-hint">
                          Long window for covariance
                        </small>
                      </label>
                    </div>
                  </div>

                  <div className="form-group ensemble-toggle">
                    <label className="checkbox-label">
                      <input
                        type="checkbox"
                        checked={predictiveParams.useEnsemble}
                        onChange={e => setPredictiveParams({
                          ...predictiveParams,
                          useEnsemble: e.target.checked
                        })}
                        className="checkbox-input"
                      />
                      <span className="checkbox-text">
                        🎭 Use Ensemble Model (combines multiple models for robustness)
                      </span>
                    </label>
                  </div>
                </div>
              )}
            </div>
          )}

          <button 
            className={`optimize-button ${usePredictive ? 'predictive-mode' : ''}`}
            onClick={handleOptimize}
            disabled={loading || tickers.length < 5}
          >
            {loading ? '⏳ Optimizing...' : (
              usePredictive ? '🤖 Run ML-Powered Optimization' :
              useWalkForward ? '🚀 Run Walk-Forward Backtest' : 
              '🚀 Optimize Portfolio'
            )}
          </button>

          {error && (
            <div className="error-message">
              <strong>⚠️ Error:</strong> {error}
            </div>
          )}
        </div>

        {results && (
          results.mode === 'walkforward' || results.mode === 'predictive_ml' ? 
            <WalkForwardResults data={results} /> :
            <Results data={results} />
        )}
      </main>

      <footer className="App-footer">
        <p>Built with React & Flask | Institutional-Grade Portfolio Optimization</p>
        <p style={{fontSize: '0.85em', opacity: 0.8, marginTop: '4px'}}>
          Max Position Size: 17% | Transaction Costs: Included | Benchmarked vs NIFTY Indices
        </p>
      </footer>
    </div>
  );
}

export default App;
