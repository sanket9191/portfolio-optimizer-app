import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import './Results.css';

const COLORS = ['#667eea', '#10b981', '#f97316', '#ef4444'];

const WalkForwardResults = ({ data }) => {
  const { strategy, benchmark, config, ml_diagnostics, mode } = data;
  const isPredictive = mode === 'predictive_ml';

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
      // Normalize to same starting point
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
      // Normalize to same starting point
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

      {/* Performance Metrics Comparison */}
      <div className="section">
        <h3>🏆 Performance Comparison</h3>
        <div className="metrics-grid">
          {/* Strategy */}
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
              <div className="metric-label">Strategy Annualized Return</div>
              <div className="metric-value">
                {((strategy?.annualized_return || 0) * 100).toFixed(2)}%
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

          <div className="metric-card" style={{borderLeft: '4px solid #667eea'}}>
            <div className="metric-content">
              <div className="metric-label">Strategy Max Drawdown</div>
              <div className="metric-value" style={{color: '#ef4444'}}>
                {((strategy?.max_drawdown || 0) * 100).toFixed(2)}%
              </div>
            </div>
          </div>

          {/* Benchmark */}
          {benchmark?.index && (
            <>
              <div className="metric-card" style={{borderLeft: '4px solid #10b981'}}>
                <div className="metric-content">
                  <div className="metric-label">{benchmark.selected} Return</div>
                  <div className="metric-value">
                    {(benchmark.index.total_return * 100).toFixed(2)}%
                  </div>
                </div>
              </div>

              <div className="metric-card" style={{borderLeft: '4px solid #10b981'}}>
                <div className="metric-content">
                  <div className="metric-label">{benchmark.selected} Sharpe</div>
                  <div className="metric-value">
                    {benchmark.index.sharpe_ratio.toFixed(3)}
                  </div>
                </div>
              </div>
            </>
          )}

          {/* Equal Weight */}
          {benchmark?.equal_weight && (
            <>
              <div className="metric-card" style={{borderLeft: '4px solid #f97316'}}>
                <div className="metric-content">
                  <div className="metric-label">Equal-Weight Return</div>
                  <div className="metric-value">
                    {(benchmark.equal_weight.total_return * 100).toFixed(2)}%
                  </div>
                </div>
              </div>

              <div className="metric-card" style={{borderLeft: '4px solid #f97316'}}>
                <div className="metric-content">
                  <div className="metric-label">Equal-Weight Sharpe</div>
                  <div className="metric-value">
                    {benchmark.equal_weight.sharpe_ratio.toFixed(3)}
                  </div>
                </div>
              </div>
            </>
          )}

          {/* Transaction Costs */}
          <div className="metric-card">
            <div className="metric-content">
              <div className="metric-label">Total Transaction Costs</div>
              <div className="metric-value">
                ₹{(strategy?.total_transaction_costs || 0).toLocaleString('en-IN', {maximumFractionDigits: 0})}
                <span style={{fontSize: '0.8em', color: '#666', marginLeft: '8px'}}>
                  ({((strategy?.transaction_costs_pct || 0) * 100).toFixed(2)}%)
                </span>
              </div>
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-content">
              <div className="metric-label">Final Portfolio Value</div>
              <div className="metric-value">
                ₹{(strategy?.final_value || 0).toLocaleString('en-IN', {maximumFractionDigits: 0})}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Equity Curves Comparison */}
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

      {/* Key Insights */}
      <div className="section" style={{
        background: isPredictive ? 'linear-gradient(135deg, #f0f9ff 0%, #e0e7ff 100%)' : '#f0f9ff',
        padding: '20px',
        borderRadius: '8px',
        borderLeft: `4px solid ${isPredictive ? '#667eea' : '#3b82f6'}`
      }}>
        <h3>💡 Key Insights</h3>
        <ul style={{lineHeight: '1.8', fontSize: '0.95em'}}>
          {isPredictive && ml_diagnostics && (
            <li style={{fontWeight: '600', color: getICStatus(ml_diagnostics.mean_ic).color}}>
              {getICStatus(ml_diagnostics.mean_ic).emoji} <strong>ML Forecast Quality:</strong> Mean IC of {(ml_diagnostics.mean_ic * 100).toFixed(2)}% 
              ({getICStatus(ml_diagnostics.mean_ic).label}) - {ml_diagnostics.mean_ic > 0.05 ? 'Strong predictive skill' : ml_diagnostics.mean_ic > 0.02 ? 'Marginal but positive signal' : 'Weak signal'}
            </li>
          )}
          <li>
            <strong>Out-of-sample performance:</strong> This backtest uses only past data at each rebalance point, 
            providing realistic estimates of strategy performance.
          </li>
          <li>
            <strong>Transaction costs included:</strong> All rebalancing costs ({config?.transaction_cost_bps || 15} bps per trade) 
            are deducted from returns, totaling ₹{(strategy?.total_transaction_costs || 0).toLocaleString('en-IN', {maximumFractionDigits: 0})} 
            ({((strategy?.transaction_costs_pct || 0) * 100).toFixed(2)}% of initial capital).
          </li>
          <li>
            <strong>Average turnover:</strong> {((strategy?.avg_turnover || 0) * 100).toFixed(1)}% per rebalance suggests 
            {(strategy?.avg_turnover || 0) < 0.5 ? ' relatively stable allocations' : ' significant portfolio changes'}.
          </li>
          {benchmark?.index && (strategy?.total_return || 0) > benchmark.index.total_return && (
            <li style={{color: '#10b981', fontWeight: '600'}}>
              ✅ Strategy outperformed {benchmark.selected} by {(((strategy?.total_return || 0) - benchmark.index.total_return) * 100).toFixed(2)}% 
              (annualized: {(((strategy?.annualized_return || 0) - benchmark.index.annualized_return) * 100).toFixed(2)}%)
            </li>
          )}
          {isPredictive && (
            <li style={{color: '#667eea', fontWeight: '600'}}>
              🤖 <strong>Powered by {config.model_type?.toUpperCase()}</strong> machine learning with {config.forecast_horizon}-month forecasts. 
              Alpha-risk separation applied: {config.alpha_lookback_months}M for signals, {config.risk_lookback_months}M for risk.
            </li>
          )}
        </ul>
      </div>
    </div>
  );
};

export default WalkForwardResults;
