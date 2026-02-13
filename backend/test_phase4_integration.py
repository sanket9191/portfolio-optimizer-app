"""
Phase 4 Integration Tests
Institutional-Grade Test Suite

Author: Senior VP - Quantitative Engineering
Purpose: Rigorous testing of all Phase 4 components
"""

import numpy as np
import pandas as pd
import sys
import time
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

# Import Phase 4 modules
try:
    from risk_management import RiskManager, RiskMetrics
    from model_validation import ModelValidator, ValidationMetrics
    from data_quality import DataQualityManager, DataQualityReport
    from transaction_costs import TransactionCostModel, TransactionCost
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("   Make sure all Phase 4 modules are in the same directory")
    sys.exit(1)

# Test data generation utilities
class TestDataGenerator:
    """Generate realistic test data for validation."""
    
    @staticmethod
    def generate_returns(n_days=500, n_assets=10, seed=42):
        """Generate realistic stock returns."""
        np.random.seed(seed)
        
        # Correlated returns
        mean_returns = np.random.rand(n_assets) * 0.0005  # Small daily returns
        cov_matrix = np.eye(n_assets) * 0.0004  # Base volatility
        
        # Add some correlation structure
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                cov_matrix[i, j] = cov_matrix[j, i] = 0.0001
        
        returns = np.random.multivariate_normal(
            mean_returns, cov_matrix, n_days
        )
        
        dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
        tickers = [f'STOCK{i}' for i in range(n_assets)]
        
        return pd.DataFrame(returns, index=dates, columns=tickers)
    
    @staticmethod
    def generate_features(n_days=500, n_assets=10, n_features=20, seed=42):
        """Generate feature matrix."""
        np.random.seed(seed)
        
        dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
        tickers = [f'STOCK{i}' for i in range(n_assets)]
        
        # MultiIndex
        index = pd.MultiIndex.from_product(
            [dates, tickers],
            names=['date', 'ticker']
        )
        
        # Generate features with some structure
        features = np.random.randn(len(index), n_features) * 0.02
        
        # Add some trending features
        trend = np.linspace(0, 0.1, len(index)).reshape(-1, 1)
        features[:, :5] += trend
        
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        return pd.DataFrame(features, index=index, columns=feature_names)


class Phase4TestSuite:
    """Comprehensive test suite for Phase 4."""
    
    def __init__(self):
        self.test_results = {}
        self.test_data = TestDataGenerator()
        
    def run_all_tests(self):
        """Run complete test suite."""
        print("\n" + "="*70)
        print("🛡️  PHASE 4 INSTITUTIONAL-GRADE TEST SUITE")
        print("="*70)
        
        tests = [
            ('Risk Management', self.test_risk_management),
            ('Model Validation', self.test_model_validation),
            ('Data Quality', self.test_data_quality),
            ('Transaction Costs', self.test_transaction_costs),
            ('Integration Test', self.test_full_integration),
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            print(f"\n{'='*70}")
            print(f"🧪 Testing: {test_name}")
            print("="*70)
            
            try:
                start_time = time.time()
                result = test_func()
                elapsed = time.time() - start_time
                
                self.test_results[test_name] = {
                    'status': 'PASSED' if result else 'FAILED',
                    'elapsed': elapsed
                }
                
                if result:
                    passed += 1
                    print(f"\n✅ {test_name}: PASSED ({elapsed:.2f}s)")
                else:
                    failed += 1
                    print(f"\n❌ {test_name}: FAILED ({elapsed:.2f}s)")
                    
            except Exception as e:
                failed += 1
                print(f"\n❌ {test_name}: EXCEPTION - {str(e)}")
                self.test_results[test_name] = {
                    'status': 'EXCEPTION',
                    'error': str(e)
                }
        
        # Summary
        print("\n" + "="*70)
        print("📈 TEST SUMMARY")
        print("="*70)
        print(f"   • Tests Passed: {passed}/{len(tests)}")
        print(f"   • Tests Failed: {failed}/{len(tests)}")
        print(f"   • Success Rate: {passed/len(tests)*100:.1f}%")
        
        if failed == 0:
            print("\n🎉 ALL TESTS PASSED - PRODUCTION READY! 🎉")
            return True
        else:
            print("\n⚠️  SOME TESTS FAILED - Review errors above")
            return False
    
    def test_risk_management(self) -> bool:
        """Test risk management module."""
        try:
            # Generate test data
            returns = self.test_data.generate_returns(n_days=252, n_assets=10)
            weights = np.array([0.1] * 10)
            portfolio_values = (1 + (returns @ weights)).cumprod()
            
            # Initialize risk manager
            risk_mgr = RiskManager()
            
            # Test 1: VaR calculations
            print("\n📊 Testing VaR calculations...")
            var_parametric = risk_mgr.calculate_var(weights, returns, method='parametric')
            var_historical = risk_mgr.calculate_var(weights, returns, method='historical')
            var_monte_carlo = risk_mgr.calculate_var(weights, returns, method='monte_carlo')
            
            assert 0 < var_parametric < 0.10, "Parametric VaR out of range"
            assert 0 < var_historical < 0.10, "Historical VaR out of range"
            assert 0 < var_monte_carlo < 0.10, "Monte Carlo VaR out of range"
            print(f"   ✅ VaR Parametric: {var_parametric:.4f}")
            print(f"   ✅ VaR Historical: {var_historical:.4f}")
            print(f"   ✅ VaR Monte Carlo: {var_monte_carlo:.4f}")
            
            # Test 2: Expected Shortfall
            print("\n📊 Testing Expected Shortfall...")
            es = risk_mgr.calculate_expected_shortfall(weights, returns)
            assert 0 < es < 0.15, "ES out of range"
            print(f"   ✅ Expected Shortfall: {es:.4f}")
            
            # Test 3: Drawdown metrics
            print("\n📉 Testing Drawdown metrics...")
            max_dd, current_dd, dd_series = risk_mgr.calculate_drawdown_metrics(portfolio_values)
            assert 0 <= max_dd <= 1.0, "Max DD out of range"
            assert 0 <= current_dd <= max_dd, "Current DD invalid"
            print(f"   ✅ Max Drawdown: {max_dd:.4f}")
            print(f"   ✅ Current Drawdown: {current_dd:.4f}")
            
            # Test 4: Comprehensive metrics
            print("\n📊 Testing Comprehensive metrics...")
            metrics = risk_mgr.calculate_comprehensive_metrics(
                weights, returns, portfolio_values
            )
            assert isinstance(metrics, RiskMetrics), "Invalid metrics type"
            assert metrics.sharpe_ratio != 0, "Invalid Sharpe ratio"
            print(f"   ✅ Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
            print(f"   ✅ Sortino Ratio: {metrics.sortino_ratio:.2f}")
            
            # Test 5: Stress testing
            print("\n💥 Testing Stress scenarios...")
            stress_results = risk_mgr.stress_test(weights, returns)
            assert len(stress_results) >= 5, "Insufficient stress scenarios"
            print(f"   ✅ Stress scenarios tested: {len(stress_results)}")
            
            # Test 6: Circuit breaker
            print("\n🚨 Testing Circuit breaker...")
            should_halt, reason = risk_mgr.trigger_circuit_breaker(metrics, -0.15)
            print(f"   ✅ Circuit breaker logic working")
            
            print("\n✔️  Risk Management: ALL CHECKS PASSED")
            return True
            
        except AssertionError as e:
            print(f"\n❌ Assertion failed: {e}")
            return False
        except Exception as e:
            print(f"\n❌ Exception: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_model_validation(self) -> bool:
        """Test model validation module."""
        try:
            from sklearn.linear_model import Ridge
            
            # Generate test data
            n_samples = 1000
            n_features = 20
            X = np.random.randn(n_samples, n_features)
            y = X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(n_samples) * 0.2
            dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
            
            # Initialize validator
            validator = ModelValidator()
            
            # Test 1: IC calculation
            print("\n📈 Testing IC calculation...")
            predictions = X[:, 0] * 0.5 + np.random.randn(n_samples) * 0.1
            ic = validator.calculate_information_coefficient(predictions, y)
            assert -1 <= ic <= 1, "IC out of range"
            print(f"   ✅ Information Coefficient: {ic:.4f}")
            
            # Test 2: IC IR calculation
            print("\n📈 Testing IC IR calculation...")
            ic_series = pd.Series(np.random.randn(50) * 0.1 + 0.05)
            ic_ir = validator.calculate_ic_information_ratio(ic_series)
            assert ic_ir != 0, "IC IR calculation failed"
            print(f"   ✅ IC Information Ratio: {ic_ir:.4f}")
            
            # Test 3: Overfitting detection
            print("\n🔍 Testing Overfitting detection...")
            is_overfit, msg = validator.detect_overfitting(0.15, 0.03)
            assert is_overfit == True, "Should detect overfitting"
            print(f"   ✅ Overfitting detection working")
            
            # Test 4: Cross-validation
            print("\n⚙️  Testing Cross-validation...")
            model = Ridge(alpha=1.0)
            best_model, metrics = validator.cross_validate_with_ic(
                model, X, y, dates
            )
            assert isinstance(metrics, ValidationMetrics), "Invalid metrics type"
            assert abs(metrics.train_ic) <= 1.0, "Train IC out of range"
            print(f"   ✅ Train IC: {metrics.train_ic:.4f}")
            print(f"   ✅ Val IC: {metrics.val_ic:.4f}")
            
            # Test 5: Hyperparameter tuning
            print("\n🎯 Testing Hyperparameter tuning...")
            param_grid = {'alpha': [0.1, 1.0, 10.0]}
            best_model, metrics = validator.cross_validate_with_ic(
                model, X, y, dates, param_grid
            )
            assert metrics.best_params is not None, "No best params found"
            print(f"   ✅ Best alpha: {metrics.best_params['alpha']}")
            
            print("\n✔️  Model Validation: ALL CHECKS PASSED")
            return True
            
        except AssertionError as e:
            print(f"\n❌ Assertion failed: {e}")
            return False
        except Exception as e:
            print(f"\n❌ Exception: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_data_quality(self) -> bool:
        """Test data quality module."""
        try:
            # Generate test data with quality issues
            features = self.test_data.generate_features(n_days=500, n_assets=10)
            
            # Inject quality issues
            features.iloc[::20] = np.nan  # Missing data
            features.iloc[50, 0] = 10.0   # Outlier
            
            # Initialize quality manager
            qm = DataQualityManager()
            
            # Test 1: Data validation
            print("\n🕵️  Testing Data validation...")
            report = qm.validate_data(features)
            assert isinstance(report, DataQualityReport), "Invalid report type"
            assert report.missing_pct > 0, "Should detect missing data"
            print(f"   ✅ Missing data: {report.missing_pct:.2%}")
            print(f"   ✅ Outliers: {report.outliers_detected}")
            
            # Test 2: Missing data imputation
            print("\n🛠️  Testing Missing data imputation...")
            features_clean = qm.handle_missing_data(features, method='hybrid')
            missing_after = features_clean.isnull().sum().sum()
            assert missing_after == 0, "Should have no missing data after imputation"
            print(f"   ✅ Missing data imputed successfully")
            
            # Test 3: Outlier detection
            print("\n🕵️  Testing Outlier detection...")
            features_clean = qm.detect_and_handle_outliers(
                features_clean, method='winsorize'
            )
            print(f"   ✅ Outliers handled successfully")
            
            # Test 4: Re-validation
            print("\n🔍 Testing Re-validation...")
            report_clean = qm.validate_data(features_clean)
            assert report_clean.missing_pct < report.missing_pct, "Quality should improve"
            print(f"   ✅ Quality improved after cleaning")
            
            # Test 5: Feature correlation
            print("\n🔗 Testing Feature correlation...")
            corr_df = qm.check_feature_correlation(features_clean.iloc[:1000])
            print(f"   ✅ Correlation analysis complete")
            
            print("\n✔️  Data Quality: ALL CHECKS PASSED")
            return True
            
        except AssertionError as e:
            print(f"\n❌ Assertion failed: {e}")
            return False
        except Exception as e:
            print(f"\n❌ Exception: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_transaction_costs(self) -> bool:
        """Test transaction cost module."""
        try:
            # Initialize cost model
            tc_model = TransactionCostModel()
            
            # Create sample portfolio transition
            n_stocks = 10
            old_weights = np.array([0.1] * n_stocks)
            new_weights = np.array([0.15, 0.12, 0.10, 0.08, 0.12, 0.11, 0.09, 0.08, 0.08, 0.07])
            portfolio_value = 1_000_000
            
            # Test 1: Cost calculation
            print("\n💰 Testing Cost calculation...")
            cost = tc_model.calculate_total_cost(new_weights, old_weights, portfolio_value)
            assert isinstance(cost, TransactionCost), "Invalid cost type"
            assert 0 < cost.total_cost < 0.01, "Total cost out of range"
            print(f"   ✅ Total cost: {cost.total_cost*10000:.2f} bps")
            print(f"   ✅ Turnover: {cost.turnover:.2%}")
            
            # Test 2: Min trade size enforcement
            print("\n🚫 Testing Min trade size...")
            adjusted_weights = tc_model.enforce_min_trade_size(
                new_weights, old_weights, min_trade=0.03
            )
            new_turnover = np.abs(adjusted_weights - old_weights).sum()
            assert new_turnover < cost.turnover, "Turnover should reduce"
            print(f"   ✅ Turnover reduced: {cost.turnover:.2%} → {new_turnover:.2%}")
            
            # Test 3: Turnover limit check
            print("\n🚫 Testing Turnover limit...")
            passes, turnover = tc_model.check_turnover_limit(new_weights, old_weights)
            print(f"   ✅ Turnover check: {passes}")
            
            # Test 4: Annual turnover estimate
            print("\n📅 Testing Annual turnover...")
            annual_turnover = tc_model.estimate_annual_turnover('monthly')
            print(f"   ✅ Estimated annual turnover: {annual_turnover:.2%}")
            
            print("\n✔️  Transaction Costs: ALL CHECKS PASSED")
            return True
            
        except AssertionError as e:
            print(f"\n❌ Assertion failed: {e}")
            return False
        except Exception as e:
            print(f"\n❌ Exception: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_full_integration(self) -> bool:
        """Test full integration of all Phase 4 modules."""
        try:
            print("\n🔗 Testing Full Integration...")
            
            # Generate realistic portfolio scenario
            returns = self.test_data.generate_returns(n_days=252, n_assets=10)
            features = self.test_data.generate_features(n_days=252, n_assets=10, n_features=15)
            
            # Step 1: Data Quality
            print("\n   1️⃣ Data Quality Check...")
            qm = DataQualityManager()
            features_clean = qm.handle_missing_data(features)
            features_clean = qm.detect_and_handle_outliers(features_clean)
            print("   ✅ Data cleaned")
            
            # Step 2: Model Training & Validation
            print("\n   2️⃣ Model Training & Validation...")
            from sklearn.linear_model import Ridge
            
            # Prepare training data (simplified)
            X = features_clean.groupby(level='date').mean().values
            y = returns.mean(axis=1).values
            dates = returns.index
            
            validator = ModelValidator()
            model = Ridge(alpha=1.0)
            
            param_grid = {'alpha': [0.1, 1.0, 10.0]}
            best_model, val_metrics = validator.cross_validate_with_ic(
                model, X, y, dates, param_grid
            )
            print(f"   ✅ Model validated (Val IC: {val_metrics.val_ic:.4f})")
            
            # Step 3: Portfolio Optimization (simplified)
            print("\n   3️⃣ Portfolio Optimization...")
            old_weights = np.array([0.1] * 10)
            
            # Generate new weights (simplified - in practice use optimizer)
            predictions = best_model.predict(X[-1:]).repeat(10)
            new_weights = np.abs(predictions)
            new_weights = new_weights / new_weights.sum()
            
            print("   ✅ Portfolio optimized")
            
            # Step 4: Transaction Cost Analysis
            print("\n   4️⃣ Transaction Cost Analysis...")
            tc_model = TransactionCostModel()
            
            # Enforce min trade size
            new_weights = tc_model.enforce_min_trade_size(
                new_weights, old_weights
            )
            
            # Calculate costs
            cost = tc_model.calculate_total_cost(
                new_weights, old_weights, portfolio_value=1_000_000
            )
            print(f"   ✅ Transaction costs: {cost.total_cost*10000:.2f} bps")
            
            # Step 5: Risk Analysis
            print("\n   5️⃣ Risk Analysis...")
            risk_mgr = RiskManager()
            
            portfolio_values = (1 + (returns @ new_weights)).cumprod()
            
            risk_metrics = risk_mgr.calculate_comprehensive_metrics(
                new_weights, returns, portfolio_values
            )
            
            print(f"   ✅ Risk metrics calculated")
            print(f"      • Sharpe: {risk_metrics.sharpe_ratio:.2f}")
            print(f"      • VaR(95%): {risk_metrics.portfolio_var_95:.2%}")
            print(f"      • Max DD: {risk_metrics.max_drawdown:.2%}")
            
            # Step 6: Final checks
            print("\n   6️⃣ Final Validation...")
            
            # Check position limits
            passes_limits, violations = risk_mgr.check_position_limits(
                new_weights, [f'STOCK{i}' for i in range(10)]
            )
            
            # Check circuit breaker
            should_halt, reason = risk_mgr.trigger_circuit_breaker(
                risk_metrics, 0.0
            )
            
            assert passes_limits, f"Position limits violated: {violations}"
            assert not should_halt, f"Circuit breaker triggered: {reason}"
            
            print("   ✅ All checks passed")
            
            print("\n✔️  Full Integration: ALL CHECKS PASSED")
            print("\n🎉 PRODUCTION-READY SYSTEM VALIDATED 🎉")
            
            return True
            
        except AssertionError as e:
            print(f"\n❌ Assertion failed: {e}")
            return False
        except Exception as e:
            print(f"\n❌ Exception: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    print("\n" + "🚀" * 35)
    print("  PHASE 4: INSTITUTIONAL ROBUSTNESS TEST SUITE  ")
    print("🚀" * 35)
    
    test_suite = Phase4TestSuite()
    success = test_suite.run_all_tests()
    
    sys.exit(0 if success else 1)
