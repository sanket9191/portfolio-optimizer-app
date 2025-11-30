import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './Results.css';

const COLORS = ['#667eea', '#10b981', '#f97316', '#ef4444'];

const WalkForwardResults = ({ data }) => {
  const { strategy, benchmark, config } = data;

  // Prepare comparison chart data
  const strategyDates = strategy.time_series.dates;
  const strategyValues = strategy.time_series.portfolio_values;
  
  const indexDates = benchmark.index ? benchmark.index.time_series.dates : [];
  const indexValues = benchmark.index ? benchmark.index.time_series.values : [];
  
  const ewDates = benchmark.equal_weight ? benchmark.equal_weight.time_series.dates : [];
  const ewValues = benchmark.equal_weight ? benchmark.equal_weight.time_series.values : [];

  // Merge all data
  const mergedMap = {};
  
  strategyDates.forEach((d, i) => {
    mergedMap[d] = { date: new Date(d).toLocaleDateString(), strategy: strategyValues[i] };
  });
  
  if (benchmark.index) {
    indexDates.forEach((d, i) => {
      if (mergedMap[d]) mergedMap[d].index = indexValues[i];
    });
  }
  
  if (benchmark.equal_weight) {
    ewDates.forEach((d, i) => {
      if (mergedMap[d]) mergedMap[d].equal_weight = ewValues[i];
    });
  }

  const comparisonData = Object.values(mergedMap).sort((a, b) => 
    new Date(a.date) - new Date(b.date)
  );

  // Prepare rebalancing timeline
  const rebalanceData = strategy.rebalance_history.map(r => ({
    date: new Date(r.date).toLocaleDateString(),
    n_stocks: r.n_stocks,
    turnover: r.turnover * 100,
    txn_cost: r.transaction_cost,
    sharpe: r.sharpe_ratio
  }));

  // Frequency display
  const freqDisplay = {
    'M': 'Monthly',
    'Q': 'Quarterly',
    'W': 'Weekly'
  }[config.rebalance_freq] || config.rebalance_freq;

  return (
    <div className="results">
      <h2>🔄 Walk-Forward Backtest Results</h2>
      <p style={{color: '#666', marginTop: '-8px', marginBottom: '24px'}}>
        Out-of-sample testing with {freqDisplay.toLowerCase()} rebalancing
      </p>

      {/* Configuration Summary */}
      <div className="section" style={{background: '#f8f9fa', padding: '16px', borderRadius: '8px', marginBottom: '24px'}}>
        <h3>⚙️ Configuration</h3>
        <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px', fontSize: '0.95em'}}>
          <div><strong>Lookback:</strong> {config.lookback_months} months</div>
          <div><strong>Rebalancing:</strong> {freqDisplay}</div>
          <div><strong>Transaction Cost:</strong> {config.transaction_cost_bps} bps</div>
          <div><strong>Max Weight:</strong> {(config.max_weight * 100).toFixed(0)}%</div>
          <div><strong>Total Rebalances:</strong> {strategy.n_rebalances}</div>
          <div><strong>Avg Turnover:</strong> {(strategy.avg_turnover * 100).toFixed(1)}%</div>
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
              <div className="metric-value" style={{color: strategy.total_return >= 0 ? '#10b981' : '#ef4444'}}>
                {(strategy.total_return * 100).toFixed(2)}%
              </div>
            </div>
          </div>

          <div className="metric-card" style={{borderLeft: '4px solid #667eea'}}>
            <div className="metric-content">
              <div className="metric-label">Strategy Annualized Return</div>
              <div className="metric-value">
                {(strategy.annualized_return * 100).toFixed(2)}%
              </div>
            </div>
          </div>

          <div className="metric-card" style={{borderLeft: '4px solid #667eea'}}>
            <div className="metric-content">
              <div className="metric-label">Strategy Sharpe Ratio</div>
              <div className="metric-value">
                {strategy.sharpe_ratio.toFixed(3)}
              </div>
            </div>
          </div>

          <div className="metric-card" style={{borderLeft: '4px solid #667eea'}}>
            <div className="metric-content">
              <div className="metric-label">Strategy Max Drawdown</div>
              <div className="metric-value" style={{color: '#ef4444'}}>
                {(strategy.max_drawdown * 100).toFixed(2)}%
              </div>
            </div>
          </div>

          {/* Benchmark */}
          {benchmark.index && (
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
          {benchmark.equal_weight && (
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
                ₹{strategy.total_transaction_costs.toLocaleString('en-IN', {maximumFractionDigits: 0})}
                <span style={{fontSize: '0.8em', color: '#666', marginLeft: '8px'}}>
                  ({(strategy.transaction_costs_pct * 100).toFixed(2)}%)
                </span>
              </div>
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-content">
              <div className="metric-label">Final Portfolio Value</div>
              <div className="metric-value">
                ₹{strategy.final_value.toLocaleString('en-IN', {maximumFractionDigits: 0})}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Equity Curves Comparison */}
      <div className="section">
        <h3>📈 Equity Curves: Strategy vs Benchmarks</h3>
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
              name="Walk-Forward Strategy"
              dot={false}
            />
            {benchmark.index && (
              <Line 
                type="monotone" 
                dataKey="index" 
                stroke="#10b981" 
                strokeWidth={2}
                name={`${benchmark.selected} Index`}
                dot={false}
              />
            )}
            {benchmark.equal_weight && (
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
            <thead style={{position: 'sticky', top: 0, background: '#fff'}}>
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
                  <td>{r.turnover.toFixed(1)}%</td>
                  <td>₹{r.txn_cost.toLocaleString('en-IN', {maximumFractionDigits: 0})}</td>
                  <td>{r.sharpe.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Key Insights */}
      <div className="section" style={{background: '#f0f9ff', padding: '20px', borderRadius: '8px', borderLeft: '4px solid #667eea'}}>
        <h3>💡 Key Insights</h3>
        <ul style={{lineHeight: '1.8', fontSize: '0.95em'}}>
          <li>
            <strong>Out-of-sample performance:</strong> This backtest uses only past data at each rebalance point, 
            providing realistic estimates of strategy performance.
          </li>
          <li>
            <strong>Transaction costs included:</strong> All rebalancing costs ({config.transaction_cost_bps} bps per trade) 
            are deducted from returns, totaling ₹{strategy.total_transaction_costs.toLocaleString('en-IN', {maximumFractionDigits: 0})} 
            ({(strategy.transaction_costs_pct * 100).toFixed(2)}% of initial capital).
          </li>
          <li>
            <strong>Average turnover:</strong> {(strategy.avg_turnover * 100).toFixed(1)}% per rebalance suggests 
            {strategy.avg_turnover < 0.5 ? ' relatively stable allocations' : ' significant portfolio changes'}.
          </li>
          {benchmark.index && strategy.total_return > benchmark.index.total_return && (
            <li style={{color: '#10b981', fontWeight: '600'}}>
              ✅ Strategy outperformed {benchmark.selected} by {((strategy.total_return - benchmark.index.total_return) * 100).toFixed(2)}% 
              (annualized: {((strategy.annualized_return - benchmark.index.annualized_return) * 100).toFixed(2)}%)
            </li>
          )}
        </ul>
      </div>
    </div>
  );
};

export default WalkForwardResults;