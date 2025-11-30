import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import './Results.css';

const COLORS = ['#667eea', '#10b981', '#f97316', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#f59e0b'];

const WalkForwardResults = ({ data }) => {
  const { strategy, benchmark, config, ml_diagnostics, mode, forward_portfolio } = data;
  const isPredictive = mode === 'predictive_ml';

  // Get latest portfolio (most recent rebalance)
  const latestHoldings = strategy?.rebalance_history?.length > 0 
    ? strategy.rebalance_history[strategy.rebalance_history.length - 1]
    : null;

  // Prepare pie chart data for latest holdings
  const holdingsPieData = latestHoldings
    ? Object.entries(latestHoldings.weights)
        .sort((a, b) => b[1] - a[1])
        .map(([ticker, weight]) => ({
          name: ticker,
          value: weight * 100
        }))
    : [];

  // Prepare comparison chart data with normalized benchmarks
  const strategyDates = strategy?.time_series?.dates || [];
  const strategyValues = strategy?.time_series?.portfolio_values || [];
  
  const indexDates = benchmark?.index?.time_series?.dates || [];
  const indexValues = benchmark?.index?.time_series?.values || [];
  
  const ewDates = benchmark?.equal_weight?.time_series?.dates || [];
  const ewValues = benchmark?.equal_weight?.time_series?.values || [];

  // Find common start date
  const allDates = new Set([...strategyDates, ...indexDates, ...ewDates]);
  const sortedDates = Array.from(allDates).sort();
  const startDate = sortedDates.length > 0 ? sortedDates[0] : null;

  // Normalize all series to start at same point (initial capital)
  const initialCapital = strategyValues.length > 0 ? strategyValues[0] : 100000;
  
  const mergedMap = {};
  
  // Add strategy values
  strategyDates.forEach((d, i) => {
    if (!mergedMap[d]) {
      mergedMap[d] = { date: new Date(d).toLocaleDateString() };
    }
    mergedMap[d].strategy = strategyValues[i];
  });
  
  // Normalize and add index values
  if (indexDates.length > 0 && indexValues.length > 0) {
    const indexStartValue = indexValues[0];
    indexDates.forEach((d, i) => {
      if (!mergedMap[d]) {
        mergedMap[d] = { date: new Date(d).toLocaleDateString() };
      }
      mergedMap[d].index = (indexValues[i] / indexStartValue) * initialCapital;
    });
  }
  
  // Normalize and add equal-weight values
  if (ewDates.length > 0 && ewValues.length > 0) {
    const ewStartValue = ewValues[0];
    ewDates.forEach((d, i) => {
      if (!mergedMap[d]) {
        mergedMap[d] = { date: new Date(d).toLocaleDateString() };
      }
      mergedMap[d].equal_weight = (ewValues[i] / ewStartValue) * initialCapital;
    });
  }

  const comparisonData = Object.values(mergedMap).sort((a, b) => 
    new Date(a.date) - new Date(b.date)
  );

  // Prepare rebalancing timeline
  const rebalanceData = (strategy?.rebalance_history || []).map(r => ({
    date: new Date(r.date).toLocaleDateString(),
    n_stocks: r.n_stocks,
    turnover: (r.turnover * 100).toFixed(1),
    txn_cost: r.transaction_cost,
    sharpe: r.sharpe_ratio.toFixed(3)
  }));

  // Frequency display
  const freqDisplay = {
    'M': 'Monthly',
    'Q': 'Quarterly',
    'W': 'Weekly'
  }[config?.rebalance_freq] || config?.rebalance_freq || 'Monthly';

  // IC interpretation
  const getICStatus = (ic) => {
    if (!ic) return { label: 'N/A', color: '#999', emoji: '❓' };
    if (ic > 0.10) return { label: 'Exceptional', color: '#10b981', emoji: '🏆' };
    if (ic > 0.05) return { label: 'Good', color: '#10b981', emoji: '✅' };
    if (ic > 0.02) return { label: 'Marginal', color: '#f97316', emoji: '⚠️' };
    return { label: 'Poor', color: '#ef4444', emoji: '❌' };
  };

  // Prepare IC history chart
  const icHistoryData = ml_diagnostics?.ic_history?.map(item => ({
    date: new Date(item.date).toLocaleDateString(),
    ic: item.ic ? (item.ic * 100).toFixed(2) : null
  })) || [];

  return (
    <div className="results">
      <h2>
        {isPredictive ? '🤖 Predictive ML Walk-Forward Results' : '🔄 Walk-Forward Backtest Results'}
      </h2>
      <p style={{color: '#666', marginTop: '-8px', marginBottom: '24px'}}>
        {isPredictive ? 'ML-powered return forecasts with' : 'Out-of-sample testing with'} {freqDisplay.toLowerCase()} rebalancing
      </p>

      {/* 🆕 Forward-Looking Portfolio Recommendation */}
      {isPredictive && forward_portfolio && (
        <div className="section" style={{
          background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
          color: '#fff',
          padding: '24px',
          borderRadius: '12px',
          marginBottom: '24px',
          boxShadow: '0 4px 16px rgba(16, 185, 129, 0.3)'
        }}>
          <h3 style={{color: '#fff', marginTop: 0, display: 'flex', alignItems: 'center', gap: '8px'}}>
            📈 RECOMMENDED PORTFOLIO FOR NEXT PERIOD
            <span style={{
              background: 'rgba(255,255,255,0.2)',
              padding: '4px 12px',
              borderRadius: '12px',
              fontSize: '0.75em',
              fontWeight: '500'
            }}>
              {forward_portfolio.valid_for_period}
            </span>
          </h3>
          
          <div style={{marginTop: '16px', fontSize: '0.95em'}}>
            <strong>Based on ML forecasts as of {forward_portfolio.recommendation_date}</strong>
          </div>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '16px',
            marginTop: '16px'
          }}>
            <div style={{background: 'rgba(255,255,255,0.15)', padding: '16px', borderRadius: '8px'}}>
              <div style={{fontSize: '0.85em', opacity: 0.9, marginBottom: '4px'}}>Expected Return</div>
              <div style={{fontSize: '1.8em', fontWeight: '700'}}>
                {(forward_portfolio.expected_return * 100).toFixed(2)}%
              </div>
              <div style={{fontSize: '0.9em', marginTop: '4px'}}>Next {forward_portfolio.forecast_horizon_months} months</div>
            </div>
            
            <div style={{background: 'rgba(255,255,255,0.15)', padding: '16px', borderRadius: '8px'}}>
              <div style={{fontSize: '0.85em', opacity: 0.9, marginBottom: '4px'}}>Expected Volatility</div>
              <div style={{fontSize: '1.8em', fontWeight: '700'}}>
                {(forward_portfolio.volatility * 100).toFixed(2)}%
              </div>
            </div>
            
            <div style={{background: 'rgba(255,255,255,0.15)', padding: '16px', borderRadius: '8px'}}>
              <div style={{fontSize: '0.85em', opacity: 0.9, marginBottom: '4px'}}>Expected Sharpe</div>
              <div style={{fontSize: '1.8em', fontWeight: '700'}}>
                {forward_portfolio.sharpe_ratio.toFixed(2)}
              </div>
            </div>
            
            <div style={{background: 'rgba(255,255,255,0.15)', padding: '16px', borderRadius: '8px'}}>
              <div style={{fontSize: '0.85em', opacity: 0.9, marginBottom: '4px'}}>Number of Stocks</div>
              <div style={{fontSize: '1.8em', fontWeight: '700'}}>
                {forward_portfolio.n_stocks}
              </div>
            </div>
          </div>

          {/* Forward Portfolio Allocation Table */}
          <div style={{marginTop: '20px', background: 'rgba(255,255,255,0.1)', padding: '16px', borderRadius: '8px'}}>
            <h4 style={{color: '#fff', marginTop: 0, marginBottom: '12px'}}>📊 Allocation Breakdown</h4>
            <div style={{maxHeight: '300px', overflowY: 'auto'}}>
              <table style={{width: '100%', borderCollapse: 'collapse'}}>
                <thead style={{position: 'sticky', top: 0, background: 'rgba(16, 185, 129, 0.9)', zIndex: 1}}>
                  <tr>
                    <th style={{padding: '8px', textAlign: 'left', color: '#fff'}}>Stock</th>
                    <th style={{padding: '8px', textAlign: 'right', color: '#fff'}}>Weight</th>
                    <th style={{padding: '8px', textAlign: 'right', color: '#fff'}}>Forecast Return</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(forward_portfolio.weights)
                    .sort((a, b) => b[1] - a[1])
                    .map(([ticker, weight], idx) => (
                      <tr key={ticker} style={{borderBottom: '1px solid rgba(255,255,255,0.1)'}}>
                        <td style={{padding: '10px', color: '#fff', fontWeight: '500'}}>{ticker}</td>
                        <td style={{padding: '10px', textAlign: 'right', color: '#fff', fontSize: '1.1em', fontWeight: '600'}}>
                          {(weight * 100).toFixed(2)}%
                        </td>
                        <td style={{padding: '10px', textAlign: 'right', color: '#d1fae5'}}>
                          {forward_portfolio.forecasts && forward_portfolio.forecasts[ticker]
                            ? `${(forward_portfolio.forecasts[ticker] * 100).toFixed(2)}%`
                            : 'N/A'}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Latest Holdings (Current Portfolio) */}
      {latestHoldings && (
        <div className="section">
          <h3>💼 Latest Portfolio Holdings</h3>
          <p style={{fontSize: '0.9em', color: '#666', marginTop: '-8px', marginBottom: '16px'}}>
            As of {new Date(latestHoldings.date).toLocaleDateString()} | {latestHoldings.n_stocks} stocks
          </p>
          
          <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginTop: '16px'}}>
            {/* Pie Chart */}
            <div>
              <h4 style={{marginBottom: '12px'}}>Allocation Pie Chart</h4>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={holdingsPieData}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={100}
                    label={(entry) => `${entry.name}: ${entry.value.toFixed(1)}%`}
                  >
                    {holdingsPieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => `${value.toFixed(2)}%`} />
                </PieChart>
              </ResponsiveContainer>
            </div>

            {/* Holdings Table */}
            <div>
              <h4 style={{marginBottom: '12px'}}>Constituent Details</h4>
              <div style={{maxHeight: '300px', overflowY: 'auto'}}>
                <table style={{width: '100%', fontSize: '0.9em'}}>
                  <thead style={{position: 'sticky', top: 0, background: '#f8f9fa', zIndex: 1}}>
                    <tr>
                      <th style={{padding: '8px', textAlign: 'left', borderBottom: '2px solid #e1e8ed'}}>Stock</th>
                      <th style={{padding: '8px', textAlign: 'right', borderBottom: '2px solid #e1e8ed'}}>Weight</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(latestHoldings.weights)
                      .sort((a, b) => b[1] - a[1])
                      .map(([ticker, weight]) => (
                        <tr key={ticker} style={{borderBottom: '1px solid #e1e8ed'}}>
                          <td style={{padding: '10px', fontWeight: '500'}}>{ticker}</td>
                          <td style={{padding: '10px', textAlign: 'right', fontSize: '1.05em', fontWeight: '600', color: '#667eea'}}>
                            {(weight * 100).toFixed(2)}%
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Portfolio Metrics */}
          <div style={{marginTop: '20px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px'}}>
            <div style={{background: '#f8f9fa', padding: '12px', borderRadius: '6px'}}>
              <div style={{fontSize: '0.85em', color: '#666'}}>Expected Return</div>
              <div style={{fontSize: '1.3em', fontWeight: '600', color: '#667eea'}}>
                {(latestHoldings.expected_return * 100).toFixed(2)}%
              </div>
            </div>
            <div style={{background: '#f8f9fa', padding: '12px', borderRadius: '6px'}}>
              <div style={{fontSize: '0.85em', color: '#666'}}>Volatility</div>
              <div style={{fontSize: '1.3em', fontWeight: '600', color: '#667eea'}}>
                {(latestHoldings.volatility * 100).toFixed(2)}%
              </div>
            </div>
            <div style={{background: '#f8f9fa', padding: '12px', borderRadius: '6px'}}>
              <div style={{fontSize: '0.85em', color: '#666'}}>Sharpe Ratio</div>
              <div style={{fontSize: '1.3em', fontWeight: '600', color: '#667eea'}}>
                {latestHoldings.sharpe_ratio.toFixed(3)}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Phase 3: ML Diagnostics Section */}
      {isPredictive && ml_diagnostics && (
        <div className="section ml-diagnostics" style={{
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          color: '#fff',
          padding: '24px',
          borderRadius: '12px',
          marginBottom: '24px',
          boxShadow: '0 4px 12px rgba(102, 126, 234, 0.3)'
        }}>
          <h3 style={{color: '#fff', marginTop: 0}}>🤖 Machine Learning Diagnostics</h3>
          
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '16px',
            marginTop: '16px'
          }}>
            {ml_diagnostics.mean_ic !== undefined && (
              <div style={{background: 'rgba(255,255,255,0.15)', padding: '16px', borderRadius: '8px'}}>
                <div style={{fontSize: '0.85em', opacity: 0.9, marginBottom: '4px'}}>Information Coefficient (IC)</div>
                <div style={{fontSize: '1.8em', fontWeight: '700'}}>
                  {(ml_diagnostics.mean_ic * 100).toFixed(2)}%
                </div>
                <div style={{fontSize: '0.9em', marginTop: '4px'}}>
                  {getICStatus(ml_diagnostics.mean_ic).emoji} {getICStatus(ml_diagnostics.mean_ic).label}
                </div>
              </div>
            )}
            
            {ml_diagnostics.forecast_quality?.positive_ic_ratio !== undefined && (
              <div style={{background: 'rgba(255,255,255,0.15)', padding: '16px', borderRadius: '8px'}}>
                <div style={{fontSize: '0.85em', opacity: 0.9, marginBottom: '4px'}}>Positive IC Rate</div>
                <div style={{fontSize: '1.8em', fontWeight: '700'}}>
                  {(ml_diagnostics.forecast_quality.positive_ic_ratio * 100).toFixed(0)}%
                </div>
                <div style={{fontSize: '0.9em', marginTop: '4px'}}>
                  {ml_diagnostics.forecast_quality.positive_ic_ratio > 0.6 ? '✅ Consistent' : '⚠️ Volatile'}
                </div>
              </div>
            )}
            
            {config.model_type && (
              <div style={{background: 'rgba(255,255,255,0.15)', padding: '16px', borderRadius: '8px'}}>
                <div style={{fontSize: '0.85em', opacity: 0.9, marginBottom: '4px'}}>Model Type</div>
                <div style={{fontSize: '1.3em', fontWeight: '600', textTransform: 'capitalize'}}>
                  {config.model_type.replace('_', ' ')}
                </div>
                <div style={{fontSize: '0.9em', marginTop: '4px'}}>
                  {config.use_ensemble ? '🎭 Ensemble Mode' : 'Single Model'}
                </div>
              </div>
            )}
            
            {config.forecast_horizon && (
              <div style={{background: 'rgba(255,255,255,0.15)', padding: '16px', borderRadius: '8px'}}>
                <div style={{fontSize: '0.85em', opacity: 0.9, marginBottom: '4px'}}>Forecast Horizon</div>
                <div style={{fontSize: '1.8em', fontWeight: '700'}}>
                  {config.forecast_horizon}M
                </div>
                <div style={{fontSize: '0.9em', marginTop: '4px'}}>Ahead Prediction</div>
              </div>
            )}
          </div>

          {/* IC Over Time Chart */}
          {icHistoryData.length > 0 && (
            <div style={{marginTop: '24px', background: 'rgba(255,255,255,0.1)', padding: '16px', borderRadius: '8px'}}>
              <h4 style={{color: '#fff', marginTop: 0, marginBottom: '12px'}}>IC Over Time</h4>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={icHistoryData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.2)" />
                  <XAxis dataKey="date" stroke="#fff" style={{fontSize: '0.75em'}} />
                  <YAxis stroke="#fff" style={{fontSize: '0.75em'}} />
                  <Tooltip
                    contentStyle={{background: '#2d3748', border: 'none', borderRadius: '6px'}}
                    labelStyle={{color: '#fff'}}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="ic" 
                    stroke="#fbbf24" 
                    strokeWidth={3}
                    dot={{ fill: '#fbbf24', r: 4 }}
                    name="IC (%)" 
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      {/* Configuration Summary */}
      <div className="section" style={{background: '#f8f9fa', padding: '16px', borderRadius: '8px', marginBottom: '24px'}}>
        <h3>⚙️ Configuration</h3>
        <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px', fontSize: '0.95em'}}>
          <div><strong>Lookback:</strong> {config?.lookback_months || config?.alpha_lookback_months || 'N/A'} months</div>
          <div><strong>Rebalancing:</strong> {freqDisplay}</div>
          <div><strong>Transaction Cost:</strong> {config?.transaction_cost_bps || 15} bps</div>
          <div><strong>Max Weight:</strong> {((config?.max_weight || 0.17) * 100).toFixed(0)}%</div>
          <div><strong>Total Rebalances:</strong> {strategy?.num_rebalances || strategy?.n_rebalances || 0}</div>
          <div><strong>Avg Turnover:</strong> {((strategy?.avg_turnover || 0) * 100).toFixed(1)}%</div>
        </div>
      </div>

      {/* Rest of the component remains the same (Performance Metrics, Equity Curves, etc.) */}
      {/* ... (keeping all previous sections) ... */}

      <div className="section">
        <h3>🏆 Performance Comparison</h3>
        <div className="metrics-grid">
          <div className="metric-card" style={{borderLeft: '4px solid #667eea'}}>
            <div className="metric-content">
              <div className="metric-label">Strategy Total Return</div>
              <div className="metric-value" style={{color: (strategy?.total_return || 0) >= 0 ? '#10b981' : '#ef4444'}}>
                {((strategy?.total_return || 0) * 100).toFixed(2)}%
              </div>
            </div>
          </div>

          <div className="metric-card" style={{borderLeft: '4px solid #667eea'}}>
            <div className="metric-content">
              <div className="metric-label">Strategy Sharpe Ratio</div>
              <div className="metric-value">
                {(strategy?.sharpe_ratio || 0).toFixed(3)}
              </div>
            </div>
          </div>

          {benchmark?.index && (
            <div className="metric-card" style={{borderLeft: '4px solid #10b981'}}>
              <div className="metric-content">
                <div className="metric-label">{benchmark.selected} Return</div>
                <div className="metric-value">
                  {(benchmark.index.total_return * 100).toFixed(2)}%
                </div>
              </div>
            </div>
          )}

          {benchmark?.equal_weight && (
            <div className="metric-card" style={{borderLeft: '4px solid #f97316'}}>
              <div className="metric-content">
                <div className="metric-label">Equal-Weight Return</div>
                <div className="metric-value">
                  {(benchmark.equal_weight.total_return * 100).toFixed(2)}%
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Equity Curves */}
      <div className="section">
        <h3>📈 Equity Curves: Strategy vs Benchmarks (Normalized)</h3>
        <p style={{fontSize: '0.9em', color: '#666', marginTop: '-8px', marginBottom: '12px'}}>
          All curves normalized to same starting point (₹{initialCapital.toLocaleString('en-IN')}) for fair comparison
        </p>
        <ResponsiveContainer width="100%" height={450}>
          <LineChart data={comparisonData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis />
            <Tooltip
              formatter={(value) => value ? `₹${value.toLocaleString('en-IN', {maximumFractionDigits: 0})}` : '-'}
            />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="strategy" 
              stroke="#667eea" 
              strokeWidth={3}
              name={isPredictive ? "ML Strategy" : "Walk-Forward Strategy"}
              dot={false}
            />
            {benchmark?.index && (
              <Line 
                type="monotone" 
                dataKey="index" 
                stroke="#10b981" 
                strokeWidth={2}
                name={`${benchmark.selected} Index`}
                dot={false}
              />
            )}
            {benchmark?.equal_weight && (
              <Line 
                type="monotone" 
                dataKey="equal_weight" 
                stroke="#f97316" 
                strokeWidth={2}
                name="Equal-Weight"
                dot={false}
                strokeDasharray="5 5"
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Rebalancing History */}
      <div className="section">
        <h3>🗓️ Rebalancing History</h3>
        <div className="weights-table" style={{maxHeight: '400px', overflowY: 'auto'}}>
          <table>
            <thead style={{position: 'sticky', top: 0, background: '#fff', zIndex: 1}}>
              <tr>
                <th>Date</th>
                <th>Stocks</th>
                <th>Turnover</th>
                <th>Txn Cost</th>
                <th>Sharpe</th>
              </tr>
            </thead>
            <tbody>
              {rebalanceData.map((r, idx) => (
                <tr key={idx}>
                  <td>{r.date}</td>
                  <td>{r.n_stocks}</td>
                  <td>{r.turnover}%</td>
                  <td>₹{parseFloat(r.txn_cost).toLocaleString('en-IN', {maximumFractionDigits: 0})}</td>
                  <td>{r.sharpe}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default WalkForwardResults;
