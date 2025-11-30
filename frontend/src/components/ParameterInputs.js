import React from 'react';
import './ParameterInputs.css';

const ParameterInputs = ({ parameters, onParametersChange }) => {
  const handleChange = (field, value) => {
    onParametersChange({
      ...parameters,
      [field]: value
    });
  };

  return (
    <div className="parameter-inputs">
      <h2>⚙️ Optimization Parameters</h2>
      
      <div className="input-grid">
        <div className="input-group">
          <label>Start Date</label>
          <input
            type="date"
            value={parameters.startDate}
            onChange={(e) => handleChange('startDate', e.target.value)}
          />
        </div>

        <div className="input-group">
          <label>End Date</label>
          <input
            type="date"
            value={parameters.endDate}
            onChange={(e) => handleChange('endDate', e.target.value)}
          />
        </div>

        <div className="input-group">
          <label>Number of Clusters</label>
          <input
            type="number"
            min="2"
            max="10"
            value={parameters.nClusters}
            onChange={(e) => handleChange('nClusters', parseInt(e.target.value))}
          />
        </div>

        <div className="input-group">
          <label>Risk-Free Rate (%)</label>
          <input
            type="number"
            min="0"
            max="20"
            step="0.1"
            value={parameters.riskFreeRate * 100}
            onChange={(e) => handleChange('riskFreeRate', parseFloat(e.target.value) / 100)}
          />
        </div>
      </div>
    </div>
  );
};

export default ParameterInputs;