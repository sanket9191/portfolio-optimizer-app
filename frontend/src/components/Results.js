import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import './Results.css';

const COLORS = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe', '#43e97b', '#fa709a', '#fee140'];

const Results = ({ data }) => {
  const { clusters, portfolio, backtest } = data;

  // Prepare portfolio allocation data for pie chart
  const allocationData = Object.entries(portfolio.weights).map(([ticker, weight]) => ({
    name: ticker,
    value: (weight * 100).toFixed(2),
    weight: weight
  }));

  // Prepare time series data
  const timeSeriesData = backtest.time_series.dates.map((date, index) => ({
    date: new Date(date).toLocaleDateString(),
    value: backtest.time_series.portfolio_values[index]
  }));

  return (
    <div className="results">
      <h2>🎯 Optimization Results</h2>

      {/* Portfolio Metrics */}
      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-icon">📈</div>
          <div className="metric-content">
            <div className="metric-label">Expected Return</div>
            <div className="metric-value">{(portfolio.expected_return * 100).toFixed(2)}%</div>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon">🎯</div>
          <div className="metric-content">
            <div className="metric-label">Sharpe Ratio</div>
            <div className="metric-value">{portfolio.sharpe_ratio.toFixed(3)}</div>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon">💠</div>
          <div className="metric-content">
            <div className="metric-label">Volatility</div>
            <div className="metric-value">{(portfolio.volatility * 100).toFixed(2)}%</div>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon">🏛️</div>
          <div className="metric-content">
            <div className="metric-label">Selected Stocks</div>
            <div className="metric-value">{portfolio.n_selected_stocks}</div>
          </div>
        </div>
      </div>

      {/* Backtest Metrics */}
      <div className="section">
        <h3>📋 Backtest Performance</h3>
        <div className="metrics-grid">
          <div className="metric-card">
            <div className="metric-content">
              <div className="metric-label">Total Return</div>
              <div className="metric-value" style={{color: backtest.total_return >= 0 ? '#10b981' : '#ef4444'}}>
                {(backtest.total_return * 100).toFixed(2)}%
              </div>
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-content">
              <div className="metric-label">Annualized Return</div>
              <div className="metric-value">{(backtest.annualized_return * 100).toFixed(2)}%</div>
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-content">
              <div className="metric-label">Max Drawdown</div>
              <div className="metric-value" style={{color: '#ef4444'}}>
                {(backtest.max_drawdown * 100).toFixed(2)}%
              </div>
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-content">
              <div className="metric-label">Final Value</div>
              <div className="metric-value">
                ₹{backtest.final_value.toLocaleString('en-IN', {maximumFractionDigits: 0})}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Portfolio Allocation Pie Chart */}
      <div className="section">
        <h3>🍰 Portfolio Allocation</h3>
        <ResponsiveContainer width="100%" height={400}>
          <PieChart>
            <Pie
              data={allocationData}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={({ name, value }) => `${name}: ${value}%`}
              outerRadius={120}
              fill="#8884d8"
              dataKey="value"
            >
              {allocationData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Portfolio Value Over Time */}
      <div className="section">
        <h3>📉 Portfolio Value Over Time</h3>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={timeSeriesData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis />
            <Tooltip 
              formatter={(value) => `₹${value.toLocaleString('en-IN', {maximumFractionDigits: 0})}`}
            />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="value" 
              stroke="#667eea" 
              strokeWidth={2}
              name="Portfolio Value"
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Cluster Information */}
      <div className="section">
        <h3>🧩 Cluster Analysis</h3>
        <div className="cluster-info">
          <p><strong>Silhouette Score:</strong> {clusters.silhouette_score.toFixed(3)}</p>
          <div className="clusters-grid">
            {clusters.cluster_stats.map((cluster, index) => (
              <div key={index} className="cluster-card" style={{borderColor: COLORS[index % COLORS.length]}}>
                <h4>Cluster {cluster.cluster_id + 1}</h4>
                <p><strong>Stocks:</strong> {cluster.n_stocks}</p>
                <p><strong>Avg RSI:</strong> {cluster.avg_rsi.toFixed(2)}</p>
                <p><strong>Avg Volatility:</strong> {(cluster.avg_volatility * 100).toFixed(4)}%</p>
                <details>
                  <summary>View Stocks</summary>
                  <div className="stock-list">
                    {cluster.stocks.map(stock => (
                      <span key={stock} className="stock-tag">{stock}</span>
                    ))}
                  </div>
                </details>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Portfolio Weights Table */}
      <div className="section">
        <h3>📊 Portfolio Weights</h3>
        <div className="weights-table">
          <table>
            <thead>
              <tr>
                <th>Ticker</th>
                <th>Weight (%)</th>
                <th>Allocation (₹)</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(portfolio.weights).map(([ticker, weight]) => (
                <tr key={ticker}>
                  <td>{ticker}</td>
                  <td>{(weight * 100).toFixed(2)}%</td>
                  <td>₹{(weight * backtest.initial_capital).toLocaleString('en-IN', {maximumFractionDigits: 0})}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default Results;