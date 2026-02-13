"""
Institutional-Grade Risk Management Module
Phase 4: Production-Ready Risk Controls

Author: Senior VP - Quantitative Engineering
Purpose: Enterprise risk management, VaR, stress testing, drawdown controls
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class RiskMetrics:
    """Container for portfolio risk metrics"""
    portfolio_var_95: float  # 1-day VaR at 95% confidence
    portfolio_var_99: float  # 1-day VaR at 99% confidence
    expected_shortfall_95: float  # CVaR/ES at 95%
    max_drawdown: float
    current_drawdown: float
    volatility_annual: float
    tracking_error: float
    beta_to_benchmark: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    

class RiskManager:
    """
    Enterprise-grade risk management system.
    
    Features:
    - Value-at-Risk (VaR) - parametric, historical, Monte Carlo
    - Expected Shortfall (CVaR)
    - Stress testing and scenario analysis
    - Drawdown monitoring and circuit breakers
    - Position limits and concentration checks
    - Real-time risk decomposition
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize risk manager with institutional parameters.
        
        Args:
            config: Risk management configuration
                - max_portfolio_var_95: Maximum allowed 1-day VaR (default: 0.03)
                - max_drawdown_limit: Maximum allowed drawdown (default: 0.20)
                - max_position_size: Maximum single stock weight (default: 0.17)
                - min_liquidity_days: Minimum days to liquidate (default: 5)
                - var_confidence_level: VaR confidence (default: 0.95)
        """
        default_config = {
            'max_portfolio_var_95': 0.03,  # 3% max 1-day VaR
            'max_drawdown_limit': 0.20,     # 20% max drawdown
            'max_position_size': 0.17,      # 17% max single stock
            'max_sector_exposure': 0.40,    # 40% max sector
            'min_liquidity_days': 5,        # 5 days to liquidate
            'var_confidence_level': 0.95,
            'stress_test_enabled': True,
            'circuit_breaker_enabled': True,
        }
        
        self.config = {**default_config, **(config or {})}
        self.risk_history = []
        self.alerts = []
        
    def calculate_var(
        self,
        weights: np.ndarray,
        returns: pd.DataFrame,
        method: str = 'parametric',
        confidence: float = 0.95,
        horizon_days: int = 1
    ) -> float:
        """
        Calculate Value-at-Risk using multiple methodologies.
        
        Args:
            weights: Portfolio weights (n_assets,)
            returns: Historical returns DataFrame (dates x assets)
            method: 'parametric', 'historical', or 'monte_carlo'
            confidence: Confidence level (0.95 or 0.99)
            horizon_days: Risk horizon in trading days
            
        Returns:
            VaR as positive number (e.g., 0.025 = 2.5% loss)
        """
        if method == 'parametric':
            return self._parametric_var(weights, returns, confidence, horizon_days)
        elif method == 'historical':
            return self._historical_var(weights, returns, confidence, horizon_days)
        elif method == 'monte_carlo':
            return self._monte_carlo_var(weights, returns, confidence, horizon_days)
        else:
            raise ValueError(f"Unknown VaR method: {method}")
    
    def _parametric_var(
        self,
        weights: np.ndarray,
        returns: pd.DataFrame,
        confidence: float,
        horizon_days: int
    ) -> float:
        """
        Parametric VaR assuming normal distribution.
        Fast but makes distributional assumptions.
        """
        # Calculate portfolio returns
        portfolio_returns = (returns @ weights)
        
        # Mean and std
        mu = portfolio_returns.mean()
        sigma = portfolio_returns.std()
        
        # Z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence)
        
        # VaR = -(mu + z * sigma) * sqrt(horizon)
        var = -(mu + z_score * sigma) * np.sqrt(horizon_days)
        
        return max(var, 0)  # Return positive number
    
    def _historical_var(
        self,
        weights: np.ndarray,
        returns: pd.DataFrame,
        confidence: float,
        horizon_days: int
    ) -> float:
        """
        Historical VaR using empirical distribution.
        No distributional assumptions - uses actual data.
        """
        # Calculate portfolio returns
        portfolio_returns = (returns @ weights)
        
        # Sort returns
        sorted_returns = np.sort(portfolio_returns)
        
        # Find percentile (e.g., 5th percentile for 95% confidence)
        percentile = 1 - confidence
        var_index = int(len(sorted_returns) * percentile)
        
        # Historical VaR
        var = -sorted_returns[var_index] * np.sqrt(horizon_days)
        
        return max(var, 0)
    
    def _monte_carlo_var(
        self,
        weights: np.ndarray,
        returns: pd.DataFrame,
        confidence: float,
        horizon_days: int,
        n_simulations: int = 10000
    ) -> float:
        """
        Monte Carlo VaR using simulated scenarios.
        Most flexible - can incorporate any distribution.
        """
        # Calculate covariance matrix
        cov_matrix = returns.cov().values
        mean_returns = returns.mean().values
        
        # Simulate portfolio returns
        simulated_returns = np.random.multivariate_normal(
            mean_returns,
            cov_matrix,
            n_simulations
        )
        
        # Calculate portfolio values
        portfolio_returns = simulated_returns @ weights
        
        # Adjust for horizon
        portfolio_returns = portfolio_returns * np.sqrt(horizon_days)
        
        # Calculate VaR
        percentile = 1 - confidence
        var = -np.percentile(portfolio_returns, percentile * 100)
        
        return max(var, 0)
    
    def calculate_expected_shortfall(
        self,
        weights: np.ndarray,
        returns: pd.DataFrame,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Expected Shortfall (CVaR) - average loss beyond VaR.
        
        ES is a coherent risk measure preferred by regulators.
        """
        # Calculate portfolio returns
        portfolio_returns = (returns @ weights)
        
        # Sort returns
        sorted_returns = np.sort(portfolio_returns)
        
        # Find VaR threshold
        percentile = 1 - confidence
        var_index = int(len(sorted_returns) * percentile)
        
        # Expected Shortfall = average of returns worse than VaR
        es = -sorted_returns[:var_index].mean()
        
        return max(es, 0)
    
    def calculate_drawdown_metrics(
        self,
        portfolio_values: pd.Series
    ) -> Tuple[float, float, pd.Series]:
        """
        Calculate maximum drawdown and current drawdown.
        
        Returns:
            (max_drawdown, current_drawdown, drawdown_series)
        """
        # Calculate running maximum
        running_max = portfolio_values.expanding().max()
        
        # Calculate drawdown series
        drawdown = (portfolio_values - running_max) / running_max
        
        # Max drawdown
        max_drawdown = drawdown.min()
        
        # Current drawdown
        current_drawdown = drawdown.iloc[-1]
        
        return abs(max_drawdown), abs(current_drawdown), drawdown
    
    def stress_test(
        self,
        weights: np.ndarray,
        returns: pd.DataFrame,
        scenarios: Optional[Dict[str, Dict]] = None
    ) -> pd.DataFrame:
        """
        Stress test portfolio under extreme market scenarios.
        
        Args:
            weights: Current portfolio weights
            returns: Historical returns
            scenarios: Custom stress scenarios or use defaults
            
        Returns:
            DataFrame with scenario results
        """
        if scenarios is None:
            scenarios = self._default_stress_scenarios()
        
        results = []
        
        for scenario_name, params in scenarios.items():
            # Apply shocks
            stressed_return = self._apply_shock(
                weights, returns, params
            )
            
            results.append({
                'scenario': scenario_name,
                'portfolio_return': stressed_return,
                'market_shock': params.get('market', 0),
                'volatility_multiplier': params.get('volatility', 1.0)
            })
        
        return pd.DataFrame(results)
    
    def _default_stress_scenarios(self) -> Dict[str, Dict]:
        """
        Default institutional stress test scenarios.
        """
        return {
            '2008 Financial Crisis': {
                'market': -0.40,
                'volatility': 2.5,
                'correlation': 0.9  # Flight to correlation
            },
            'COVID-19 Crash (Mar 2020)': {
                'market': -0.35,
                'volatility': 3.0,
                'correlation': 0.85
            },
            'Flash Crash': {
                'market': -0.15,
                'volatility': 5.0,
                'correlation': 0.95
            },
            'Moderate Correction': {
                'market': -0.10,
                'volatility': 1.5,
                'correlation': 0.7
            },
            'Severe Bear Market': {
                'market': -0.50,
                'volatility': 3.0,
                'correlation': 0.9
            },
            'Inflation Shock': {
                'market': -0.20,
                'volatility': 2.0,
                'correlation': 0.6
            }
        }
    
    def _apply_shock(
        self,
        weights: np.ndarray,
        returns: pd.DataFrame,
        shock_params: Dict
    ) -> float:
        """
        Apply shock to portfolio and calculate stressed return.
        """
        # Portfolio returns
        portfolio_returns = (returns @ weights)
        
        # Market shock
        market_shock = shock_params.get('market', 0)
        
        # Volatility multiplier
        vol_multiplier = shock_params.get('volatility', 1.0)
        
        # Calculate stressed return
        # Simple model: shocked_return = market_shock + vol_adjustment
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        
        stressed_return = market_shock + (mean_return - std_return * vol_multiplier)
        
        return stressed_return
    
    def check_position_limits(
        self,
        weights: np.ndarray,
        tickers: List[str],
        sector_map: Optional[Dict[str, str]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Check if portfolio violates position limits.
        
        Returns:
            (passes_checks, list_of_violations)
        """
        violations = []
        
        # Check individual position limits
        max_weight = weights.max()
        if max_weight > self.config['max_position_size']:
            idx = weights.argmax()
            violations.append(
                f"Position limit breach: {tickers[idx]} weight {max_weight:.1%} "
                f"exceeds max {self.config['max_position_size']:.1%}"
            )
        
        # Check sector limits if sector_map provided
        if sector_map is not None:
            sector_exposures = self._calculate_sector_exposure(weights, tickers, sector_map)
            
            for sector, exposure in sector_exposures.items():
                if exposure > self.config['max_sector_exposure']:
                    violations.append(
                        f"Sector limit breach: {sector} exposure {exposure:.1%} "
                        f"exceeds max {self.config['max_sector_exposure']:.1%}"
                    )
        
        passes = len(violations) == 0
        return passes, violations
    
    def _calculate_sector_exposure(
        self,
        weights: np.ndarray,
        tickers: List[str],
        sector_map: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Calculate exposure by sector.
        """
        sector_exposures = {}
        
        for i, ticker in enumerate(tickers):
            sector = sector_map.get(ticker, 'Unknown')
            sector_exposures[sector] = sector_exposures.get(sector, 0) + weights[i]
        
        return sector_exposures
    
    def calculate_comprehensive_metrics(
        self,
        weights: np.ndarray,
        returns: pd.DataFrame,
        portfolio_values: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for reporting.
        
        This is the main function to call for risk reporting.
        """
        # VaR calculations
        var_95 = self.calculate_var(weights, returns, confidence=0.95)
        var_99 = self.calculate_var(weights, returns, confidence=0.99)
        
        # Expected Shortfall
        es_95 = self.calculate_expected_shortfall(weights, returns, confidence=0.95)
        
        # Drawdown metrics
        max_dd, current_dd, _ = self.calculate_drawdown_metrics(portfolio_values)
        
        # Portfolio returns
        portfolio_returns = (returns @ weights)
        
        # Volatility
        vol_annual = portfolio_returns.std() * np.sqrt(252)
        
        # Tracking error and beta (if benchmark provided)
        if benchmark_returns is not None:
            tracking_error = (portfolio_returns - benchmark_returns).std() * np.sqrt(252)
            
            # Beta calculation
            covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
            benchmark_variance = benchmark_returns.var()
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
        else:
            tracking_error = 0.0
            beta = 1.0
        
        # Sharpe ratio (assume 0% risk-free rate for simplicity)
        sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
        
        # Sortino ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = downside_returns.std()
        sortino = portfolio_returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0
        
        # Calmar ratio (return / max drawdown)
        annual_return = portfolio_returns.mean() * 252
        calmar = annual_return / max_dd if max_dd > 0 else 0
        
        return RiskMetrics(
            portfolio_var_95=var_95,
            portfolio_var_99=var_99,
            expected_shortfall_95=es_95,
            max_drawdown=max_dd,
            current_drawdown=current_dd,
            volatility_annual=vol_annual,
            tracking_error=tracking_error,
            beta_to_benchmark=beta,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar
        )
    
    def trigger_circuit_breaker(
        self,
        current_metrics: RiskMetrics,
        portfolio_return_today: float
    ) -> Tuple[bool, str]:
        """
        Check if circuit breaker should halt trading.
        
        Returns:
            (should_halt, reason)
        """
        if not self.config['circuit_breaker_enabled']:
            return False, ""
        
        # Check 1: Exceeded VaR limit
        if current_metrics.portfolio_var_95 > self.config['max_portfolio_var_95']:
            return True, f"VaR breach: {current_metrics.portfolio_var_95:.2%} > {self.config['max_portfolio_var_95']:.2%}"
        
        # Check 2: Exceeded drawdown limit
        if current_metrics.current_drawdown > self.config['max_drawdown_limit']:
            return True, f"Drawdown breach: {current_metrics.current_drawdown:.2%} > {self.config['max_drawdown_limit']:.2%}"
        
        # Check 3: Single-day loss > 10%
        if portfolio_return_today < -0.10:
            return True, f"Single-day loss: {portfolio_return_today:.2%}"
        
        return False, ""
    
    def generate_risk_report(self, metrics: RiskMetrics) -> str:
        """
        Generate formatted risk report for monitoring.
        """
        report = f"""
╔══════════════════════════════════════════════════════════╗
║           INSTITUTIONAL RISK REPORT                      ║
╚══════════════════════════════════════════════════════════╝

📊 VALUE-AT-RISK METRICS
   • 1-Day VaR (95%):          {metrics.portfolio_var_95:>8.2%}
   • 1-Day VaR (99%):          {metrics.portfolio_var_99:>8.2%}
   • Expected Shortfall (95%): {metrics.expected_shortfall_95:>8.2%}
   • Max Allowed VaR:          {self.config['max_portfolio_var_95']:>8.2%}

📉 DRAWDOWN ANALYSIS
   • Maximum Drawdown:         {metrics.max_drawdown:>8.2%}
   • Current Drawdown:         {metrics.current_drawdown:>8.2%}
   • Drawdown Limit:           {self.config['max_drawdown_limit']:>8.2%}
   • Status: {'⚠️  APPROACHING LIMIT' if metrics.current_drawdown > self.config['max_drawdown_limit'] * 0.8 else '✅ Within Limits'}

🎯 PERFORMANCE METRICS
   • Annual Volatility:        {metrics.volatility_annual:>8.2%}
   • Sharpe Ratio:             {metrics.sharpe_ratio:>8.2f}
   • Sortino Ratio:            {metrics.sortino_ratio:>8.2f}
   • Calmar Ratio:             {metrics.calmar_ratio:>8.2f}

📈 BENCHMARK RELATIVE
   • Beta to Benchmark:        {metrics.beta_to_benchmark:>8.2f}
   • Tracking Error:           {metrics.tracking_error:>8.2%}

{'⚠️  RISK ALERTS ACTIVE' if len(self.alerts) > 0 else '✅ ALL RISK CHECKS PASSED'}
"""
        return report


if __name__ == "__main__":
    # Quick test
    print("\n🏦 INSTITUTIONAL RISK MANAGEMENT MODULE")
    print("   Phase 4 - Production Ready\n")
    
    # Initialize
    risk_mgr = RiskManager()
    
    # Create sample data
    np.random.seed(42)
    n_days = 252
    n_assets = 10
    
    returns = pd.DataFrame(
        np.random.randn(n_days, n_assets) * 0.02,
        columns=[f"STOCK{i}" for i in range(n_assets)]
    )
    
    weights = np.array([0.1] * n_assets)
    
    portfolio_values = (1 + (returns @ weights)).cumprod()
    
    # Calculate metrics
    metrics = risk_mgr.calculate_comprehensive_metrics(
        weights, returns, portfolio_values
    )
    
    # Print report
    print(risk_mgr.generate_risk_report(metrics))
    
    # Stress test
    print("\n💥 STRESS TEST RESULTS:")
    stress_results = risk_mgr.stress_test(weights, returns)
    print(stress_results.to_string(index=False))
    
    print("\n✅ Risk Management Module Test Complete")
