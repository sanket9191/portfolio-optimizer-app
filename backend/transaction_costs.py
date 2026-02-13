"""
Institutional-Grade Transaction Cost Module
Phase 4: Production-Ready Cost Modeling

Author: Senior VP - Quantitative Engineering
Purpose: Realistic transaction cost modeling, turnover optimization, min trade size
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
import logging

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TransactionCost:
    """Container for transaction cost breakdown"""
    total_cost: float
    commission: float
    bid_ask_spread: float
    market_impact: float
    slippage: float
    turnover: float
    trades_count: int
    

class TransactionCostModel:
    """
    Enterprise-grade transaction cost model.
    
    Features:
    - Commission costs (flat + percentage)
    - Bid-ask spread costs
    - Market impact (non-linear in trade size)
    - Slippage estimation
    - Turnover calculation and penalties
    - Minimum trade size enforcement
    - Liquidity-adjusted costs
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize transaction cost model.
        
        Args:
            config: Cost model parameters
                - commission_bps: Commission in basis points (default: 5)
                - flat_commission: Flat commission per trade (default: 0)
                - bid_ask_spread_bps: Bid-ask spread (default: 10)
                - market_impact_coef: Impact coefficient (default: 0.1)
                - min_trade_size: Minimum trade size (default: 0.02 = 2%)
                - max_turnover: Maximum allowed turnover (default: 3.0 = 300%)
        """
        default_config = {
            'commission_bps': 5.0,  # 5 bps = 0.05%
            'flat_commission': 0.0,  # No flat commission for now
            'bid_ask_spread_bps': 10.0,  # 10 bps = 0.10%
            'market_impact_coef': 0.10,  # Impact grows with turnover
            'slippage_bps': 3.0,  # 3 bps average slippage
            'min_trade_size': 0.02,  # 2% minimum trade
            'max_turnover': 3.0,  # 300% max annual turnover
            'adv_multiplier': 0.10,  # Trade up to 10% of ADV
        }
        
        self.config = {**default_config, **(config or {})}
        self.cost_history = []
        
    def calculate_total_cost(
        self,
        new_weights: np.ndarray,
        old_weights: np.ndarray,
        portfolio_value: float = 1000000.0,
        adv_values: Optional[np.ndarray] = None
    ) -> TransactionCost:
        """
        Calculate total transaction costs for rebalancing.
        
        This is the MAIN cost calculation function.
        
        Args:
            new_weights: Target portfolio weights
            old_weights: Current portfolio weights
            portfolio_value: Total portfolio value in currency
            adv_values: Average daily volume for each stock (optional)
            
        Returns:
            TransactionCost object with detailed breakdown
        """
        # Calculate turnover
        turnover = np.abs(new_weights - old_weights).sum()
        
        # Trade values
        trade_values = np.abs(new_weights - old_weights) * portfolio_value
        trades_count = (trade_values > 0).sum()
        
        # Commission costs
        commission = self._calculate_commission(trade_values)
        
        # Bid-ask spread costs
        bid_ask = self._calculate_bid_ask_spread(trade_values)
        
        # Market impact costs (non-linear)
        market_impact = self._calculate_market_impact(
            trade_values, turnover, adv_values
        )
        
        # Slippage
        slippage = self._calculate_slippage(trade_values)
        
        # Total cost as fraction of portfolio
        total_cost = (commission + bid_ask + market_impact + slippage) / portfolio_value
        
        result = TransactionCost(
            total_cost=total_cost,
            commission=commission / portfolio_value,
            bid_ask_spread=bid_ask / portfolio_value,
            market_impact=market_impact / portfolio_value,
            slippage=slippage / portfolio_value,
            turnover=turnover,
            trades_count=int(trades_count)
        )
        
        self.cost_history.append(result)
        
        return result
    
    def _calculate_commission(self, trade_values: np.ndarray) -> float:
        """
        Calculate commission costs.
        
        Commission = flat_fee + (trade_value * commission_rate)
        """
        commission_rate = self.config['commission_bps'] / 10000.0
        flat_commission = self.config['flat_commission']
        
        # Commission on each trade
        trade_commissions = trade_values * commission_rate + flat_commission
        
        return trade_commissions.sum()
    
    def _calculate_bid_ask_spread(self, trade_values: np.ndarray) -> float:
        """
        Calculate bid-ask spread costs.
        
        Spread cost = trade_value * (spread / 2)
        We pay half the spread on average.
        """
        spread_rate = self.config['bid_ask_spread_bps'] / 10000.0
        
        # Half spread cost
        spread_cost = trade_values * (spread_rate / 2.0)
        
        return spread_cost.sum()
    
    def _calculate_market_impact(
        self,
        trade_values: np.ndarray,
        turnover: float,
        adv_values: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate market impact costs (non-linear).
        
        Impact grows with:
        1. Trade size relative to ADV
        2. Overall turnover (market pressure)
        
        Model: impact = coefficient * trade_value * sqrt(turnover)
        """
        impact_coef = self.config['market_impact_coef']
        
        # Base impact proportional to sqrt(turnover)
        # This captures non-linear price impact
        impact = trade_values * impact_coef * np.sqrt(turnover)
        
        # If ADV provided, adjust for liquidity
        if adv_values is not None:
            # Increase impact if trade is large relative to ADV
            adv_multiplier = self.config['adv_multiplier']
            liquidity_adjustment = np.minimum(
                trade_values / (adv_values * adv_multiplier + 1e-10),
                2.0  # Cap at 2x
            )
            impact = impact * (1.0 + liquidity_adjustment)
        
        return impact.sum()
    
    def _calculate_slippage(self, trade_values: np.ndarray) -> float:
        """
        Calculate slippage costs.
        
        Slippage = difference between expected and executed price.
        """
        slippage_rate = self.config['slippage_bps'] / 10000.0
        
        slippage = trade_values * slippage_rate
        
        return slippage.sum()
    
    def optimize_with_turnover_penalty(
        self,
        expected_returns: np.ndarray,
        risk_matrix: np.ndarray,
        current_weights: np.ndarray,
        turnover_penalty: Optional[float] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Optimize portfolio considering transaction costs.
        
        Objective: Maximize (alpha - turnover_cost - risk_penalty)
        
        This should be integrated into the main optimizer.
        
        Args:
            expected_returns: Expected returns (alpha forecasts)
            risk_matrix: Covariance matrix
            current_weights: Current portfolio weights
            turnover_penalty: Cost per unit of turnover (bps)
            **kwargs: Additional optimizer arguments
            
        Returns:
            Optimized weights considering transaction costs
        """
        import cvxpy as cp
        
        n = len(expected_returns)
        w = cp.Variable(n)
        
        # Default turnover penalty based on config
        if turnover_penalty is None:
            # Estimate total cost per unit turnover
            total_bps = (
                self.config['commission_bps'] +
                self.config['bid_ask_spread_bps'] / 2.0 +
                self.config['slippage_bps']
            )
            turnover_penalty = total_bps / 10000.0
        
        # Turnover
        turnover = cp.norm(w - current_weights, 1)
        
        # Risk penalty (lambda parameter)
        risk_aversion = kwargs.get('risk_aversion', 1.0)
        
        # Objective: alpha - turnover_cost - risk
        objective = cp.Maximize(
            expected_returns @ w
            - turnover_penalty * turnover
            - risk_aversion * cp.quad_form(w, risk_matrix)
        )
        
        # Standard constraints
        constraints = [
            cp.sum(w) == 1,  # Fully invested
            w >= 0,  # Long-only
        ]
        
        # Add position limits if provided
        if 'max_weight' in kwargs:
            constraints.append(w <= kwargs['max_weight'])
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS)
        
        if w.value is None:
            logger.warning("⚠️  Optimization failed, returning current weights")
            return current_weights
        
        return w.value
    
    def enforce_min_trade_size(
        self,
        new_weights: np.ndarray,
        old_weights: np.ndarray,
        min_trade: Optional[float] = None
    ) -> np.ndarray:
        """
        Enforce minimum trade size to reduce noise trading.
        
        Don't trade if weight change < min_trade threshold.
        
        Args:
            new_weights: Target weights
            old_weights: Current weights
            min_trade: Minimum trade size (default from config)
            
        Returns:
            Adjusted weights with small trades removed
        """
        if min_trade is None:
            min_trade = self.config['min_trade_size']
        
        adjusted_weights = new_weights.copy()
        delta = new_weights - old_weights
        
        # Identify small trades
        small_trades = np.abs(delta) < min_trade
        
        # Keep old weights for small trades
        adjusted_weights[small_trades] = old_weights[small_trades]
        
        # Renormalize to sum to 1
        adjusted_weights = adjusted_weights / adjusted_weights.sum()
        
        # Count trades skipped
        trades_skipped = small_trades.sum()
        
        if trades_skipped > 0:
            logger.info(f"\n🚫 MIN TRADE SIZE FILTER:")
            logger.info(f"   • Trades skipped: {trades_skipped}")
            logger.info(f"   • Min trade size: {min_trade:.2%}")
        
        return adjusted_weights
    
    def estimate_annual_turnover(
        self,
        rebalance_frequency: str = 'monthly'
    ) -> float:
        """
        Estimate annual turnover based on rebalance frequency.
        
        Args:
            rebalance_frequency: 'daily', 'weekly', 'monthly', 'quarterly'
            
        Returns:
            Estimated annual turnover
        """
        if not self.cost_history:
            return 0.0
        
        # Average turnover per rebalance
        avg_turnover = np.mean([c.turnover for c in self.cost_history])
        
        # Rebalances per year
        rebalance_per_year = {
            'daily': 252,
            'weekly': 52,
            'monthly': 12,
            'quarterly': 4,
            'annual': 1
        }
        
        freq = rebalance_per_year.get(rebalance_frequency, 12)
        
        annual_turnover = avg_turnover * freq
        
        return annual_turnover
    
    def check_turnover_limit(
        self,
        new_weights: np.ndarray,
        old_weights: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Check if rebalance exceeds turnover limit.
        
        Returns:
            (passes_check, turnover)
        """
        turnover = np.abs(new_weights - old_weights).sum()
        
        max_turnover = self.config['max_turnover'] / 12  # Monthly limit
        
        passes = turnover <= max_turnover
        
        if not passes:
            logger.warning(
                f"⚠️  TURNOVER LIMIT BREACH: {turnover:.2f} > {max_turnover:.2f}"
            )
        
        return passes, turnover
    
    def generate_cost_report(self, cost: TransactionCost) -> str:
        """
        Generate formatted transaction cost report.
        """
        report = f"""
╔══════════════════════════════════════════════════════════╗
║           TRANSACTION COST ANALYSIS                      ║
╚══════════════════════════════════════════════════════════╝

💰 TOTAL TRANSACTION COST: {cost.total_cost*100:.3f} bps

📊 COST BREAKDOWN:
   • Commission:      {cost.commission*10000:>8.2f} bps
   • Bid-Ask Spread:  {cost.bid_ask_spread*10000:>8.2f} bps
   • Market Impact:   {cost.market_impact*10000:>8.2f} bps
   • Slippage:        {cost.slippage*10000:>8.2f} bps

🔄 TURNOVER METRICS:
   • Turnover:        {cost.turnover:>8.2%}
   • Trades executed: {cost.trades_count:>8d}
   • Max turnover:    {self.config['max_turnover']/12:>8.2%} (monthly)

📈 ANNUALIZED ESTIMATES (assuming monthly rebalance):
   • Annual cost:     {cost.total_cost * 12:>8.2%}
   • Annual turnover: {cost.turnover * 12:>8.2%}
"""
        return report


if __name__ == "__main__":
    # Quick test
    print("\n💰 TRANSACTION COST MODULE")
    print("   Phase 4 - Production Ready\n")
    
    # Initialize
    tc_model = TransactionCostModel()
    
    # Create sample portfolio transition
    n_stocks = 10
    old_weights = np.array([0.1] * n_stocks)
    new_weights = np.array([0.15, 0.12, 0.10, 0.08, 0.12, 0.11, 0.09, 0.08, 0.08, 0.07])
    
    portfolio_value = 1_000_000  # $1M portfolio
    
    # Calculate costs
    cost = tc_model.calculate_total_cost(
        new_weights, old_weights, portfolio_value
    )
    
    # Print report
    print(tc_model.generate_cost_report(cost))
    
    # Test min trade size enforcement
    print("\n🚫 TESTING MIN TRADE SIZE FILTER...")
    adjusted_weights = tc_model.enforce_min_trade_size(
        new_weights, old_weights, min_trade=0.03
    )
    
    print(f"   • Original turnover: {np.abs(new_weights - old_weights).sum():.2%}")
    print(f"   • Adjusted turnover: {np.abs(adjusted_weights - old_weights).sum():.2%}")
    
    print("\n✅ Transaction Cost Module Test Complete")
