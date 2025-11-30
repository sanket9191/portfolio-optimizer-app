import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const optimizePortfolio = async (params) => {
  try {
    const response = await api.post('/api/optimize', params);
    return response.data;
  } catch (error) {
    console.error('API Error:', error.response?.data || error.message);
    throw new Error(error.response?.data?.error || 'Failed to optimize portfolio');
  }
};

export const getIndexTickers = async (index) => {
  try {
    const response = await api.get(`/api/tickers/${index}`);
    return response.data;
  } catch (error) {
    console.error('API Error:', error.response?.data || error.message);
    throw new Error(error.response?.data?.error || 'Failed to fetch index tickers');
  }
};

export default api;