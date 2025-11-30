import React, { useState } from 'react';
import { getIndexTickers } from '../api';
import './StockSelector.css';

const StockSelector = ({ selectedTickers, onTickersChange }) => {
  const [customTicker, setCustomTicker] = useState('');
  const [selectedIndex, setSelectedIndex] = useState('');

  const handleLoadIndex = async (index) => {
    try {
      const data = await getIndexTickers(index);
      onTickersChange(data.tickers);
      setSelectedIndex(index);
    } catch (error) {
      console.error('Error loading index:', error);
    }
  };

  const handleAddCustomTicker = () => {
    if (customTicker.trim()) {
      const ticker = customTicker.trim().toUpperCase();
      if (!ticker.endsWith('.NS')) {
        onTickersChange([...selectedTickers, `${ticker}.NS`]);
      } else {
        onTickersChange([...selectedTickers, ticker]);
      }
      setCustomTicker('');
    }
  };

  const handleRemoveTicker = (ticker) => {
    onTickersChange(selectedTickers.filter(t => t !== ticker));
  };

  return (
    <div className="stock-selector">
      <h2>📊 Select Stocks</h2>
      
      <div className="index-buttons">
        <button 
          className={selectedIndex === 'NIFTY50' ? 'active' : ''}
          onClick={() => handleLoadIndex('NIFTY50')}
        >
          Load NIFTY 50
        </button>
        <button 
          className={selectedIndex === 'NIFTYBANK' ? 'active' : ''}
          onClick={() => handleLoadIndex('NIFTYBANK')}
        >
          Load NIFTY Bank
        </button>
      </div>

      <div className="custom-ticker-input">
        <input
          type="text"
          placeholder="Add custom ticker (e.g., RELIANCE)"
          value={customTicker}
          onChange={(e) => setCustomTicker(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleAddCustomTicker()}
        />
        <button onClick={handleAddCustomTicker}>Add</button>
      </div>

      <div className="selected-tickers">
        <h3>Selected Tickers ({selectedTickers.length})</h3>
        <div className="ticker-chips">
          {selectedTickers.map(ticker => (
            <div key={ticker} className="ticker-chip">
              <span>{ticker}</span>
              <button onClick={() => handleRemoveTicker(ticker)}>×</button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default StockSelector;