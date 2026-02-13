"""
Institutional-Grade Model Validation Module
Phase 4: Production-Ready Model Quality Controls

Author: Senior VP - Quantitative Engineering
Purpose: Overfitting detection, hyperparameter optimization, feature selection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import make_scorer
from dataclasses import dataclass
import warnings
from scipy.stats import spearmanr
import logging

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Container for model validation metrics"""
    train_ic: float
    val_ic: float
    test_ic: float
    train_ic_ir: float  # IC Information Ratio
    val_ic_ir: float
    train_r2: float
    val_r2: float
    overfitting_score: float  # train_ic - val_ic
    hit_rate: float  # % of correct sign predictions
    feature_importance: Optional[Dict[str, float]] = None
    best_params: Optional[Dict[str, Any]] = None
    

class ModelValidator:
    """
    Institutional-grade model validation system.
    
    Features:
    - Time-series cross-validation (expanding window)
    - Hyperparameter optimization with IC-based scoring
    - Overfitting detection and alerts
    - Feature importance tracking and selection
    - Walk-forward IC stability analysis
    - Information Coefficient Information Ratio (IC IR)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize model validator.
        
        Args:
            config: Validation configuration
                - cv_n_splits: Number of CV folds (default: 5)
                - ic_threshold: Minimum acceptable IC (default: 0.03)
                - overfitting_threshold: Max train-val IC gap (default: 0.10)
                - ic_ir_threshold: Min IC IR for stability (default: 0.5)
                - feature_selection_threshold: Min importance (default: 0.01)
        """
        default_config = {
            'cv_n_splits': 5,
            'ic_threshold': 0.03,  # 3% minimum IC
            'overfitting_threshold': 0.10,  # 10% max gap
            'ic_ir_threshold': 0.5,  # IC/std(IC) > 0.5
            'feature_selection_threshold': 0.01,
            'min_train_samples': 252,  # Min 1 year data
        }
        
        self.config = {**default_config, **(config or {})}
        self.validation_history = []
        
    def calculate_information_coefficient(
        self,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
        method: str = 'spearman'
    ) -> float:
        """
        Calculate Information Coefficient (IC).
        
        IC = correlation(predicted_returns, actual_returns)
        
        Args:
            predictions: Predicted returns
            actual_returns: Realized returns
            method: 'spearman' (default, rank-based) or 'pearson'
            
        Returns:
            IC value (-1 to 1)
        """
        # Remove NaN values
        mask = ~(np.isnan(predictions) | np.isnan(actual_returns))
        pred_clean = predictions[mask]
        actual_clean = actual_returns[mask]
        
        if len(pred_clean) < 10:
            return 0.0
        
        if method == 'spearman':
            ic, _ = spearmanr(pred_clean, actual_clean)
        else:  # pearson
            ic = np.corrcoef(pred_clean, actual_clean)[0, 1]
        
        return ic if not np.isnan(ic) else 0.0
    
    def calculate_ic_information_ratio(
        self,
        ic_series: pd.Series
    ) -> float:
        """
        Calculate IC Information Ratio (IC IR).
        
        IC IR = mean(IC) / std(IC)
        
        Measures stability of alpha signal.
        Higher is better (>1.0 is excellent).
        """
        if len(ic_series) < 2:
            return 0.0
        
        mean_ic = ic_series.mean()
        std_ic = ic_series.std()
        
        if std_ic == 0:
            return 0.0
        
        return mean_ic / std_ic
    
    def detect_overfitting(
        self,
        train_ic: float,
        val_ic: float,
        threshold: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Detect model overfitting.
        
        Returns:
            (is_overfitting, diagnostic_message)
        """
        if threshold is None:
            threshold = self.config['overfitting_threshold']
        
        gap = train_ic - val_ic
        
        is_overfitting = gap > threshold
        
        if is_overfitting:
            message = f"""
⚠️  OVERFITTING DETECTED:
   Train IC: {train_ic:.4f}
   Val IC:   {val_ic:.4f}
   Gap:      {gap:.4f} (threshold: {threshold:.4f})
   
   Recommendations:
   • Increase regularization (alpha/lambda)
   • Reduce model complexity
   • Feature selection (remove weak features)
   • More training data
   • Ensemble methods for stability
"""
        else:
            message = f"✅ No overfitting detected (gap: {gap:.4f})"
        
        return is_overfitting, message
    
    def time_series_cv_split(
        self,
        dates: pd.DatetimeIndex,
        n_splits: Optional[int] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate time-series cross-validation splits.
        
        Uses expanding window (not rolling) to preserve data.
        
        Returns:
            List of (train_indices, val_indices) tuples
        """
        if n_splits is None:
            n_splits = self.config['cv_n_splits']
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Get indices
        splits = []
        for train_idx, val_idx in tscv.split(dates):
            # Ensure minimum training samples
            if len(train_idx) >= self.config['min_train_samples']:
                splits.append((train_idx, val_idx))
        
        return splits
    
    def cross_validate_with_ic(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        dates: pd.DatetimeIndex,
        param_grid: Optional[Dict] = None
    ) -> Tuple[Any, ValidationMetrics]:
        """
        Cross-validate model with IC-based scoring.
        
        This is the MAIN validation function.
        
        Args:
            model: sklearn-compatible model
            X: Features (n_samples, n_features)
            y: Target returns (n_samples,)
            dates: Date index for time-series splits
            param_grid: Hyperparameter grid for tuning
            
        Returns:
            (best_model, validation_metrics)
        """
        logger.info("\n" + "="*60)
        logger.info("🔍 STARTING CROSS-VALIDATION WITH IC SCORING")
        logger.info("="*60)
        
        # Get CV splits
        splits = self.time_series_cv_split(dates)
        logger.info(f"   • CV Splits: {len(splits)}")
        logger.info(f"   • Training samples: {len(X)}")
        logger.info(f"   • Features: {X.shape[1]}")
        
        # If no param grid, just validate current model
        if param_grid is None:
            return self._validate_single_model(
                model, X, y, dates, splits
            )
        
        # Otherwise, do hyperparameter tuning
        return self._hyperparameter_tuning(
            model, X, y, dates, splits, param_grid
        )
    
    def _validate_single_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        dates: pd.DatetimeIndex,
        splits: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[Any, ValidationMetrics]:
        """
        Validate a single model configuration.
        """
        train_ics = []
        val_ics = []
        train_r2s = []
        val_r2s = []
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            pred_train = model.predict(X_train)
            pred_val = model.predict(X_val)
            
            # Calculate IC
            train_ic = self.calculate_information_coefficient(pred_train, y_train)
            val_ic = self.calculate_information_coefficient(pred_val, y_val)
            
            train_ics.append(train_ic)
            val_ics.append(val_ic)
            
            # R-squared
            train_r2 = model.score(X_train, y_train)
            val_r2 = model.score(X_val, y_val)
            
            train_r2s.append(train_r2)
            val_r2s.append(val_r2)
            
            logger.info(f"   Fold {fold+1}: Train IC={train_ic:.4f}, Val IC={val_ic:.4f}")
        
        # Average metrics
        train_ic = np.mean(train_ics)
        val_ic = np.mean(val_ics)
        train_r2 = np.mean(train_r2s)
        val_r2 = np.mean(val_r2s)
        
        # IC IR
        train_ic_ir = self.calculate_ic_information_ratio(pd.Series(train_ics))
        val_ic_ir = self.calculate_ic_information_ratio(pd.Series(val_ics))
        
        # Final train on all data
        model.fit(X, y)
        pred_all = model.predict(X)
        test_ic = self.calculate_information_coefficient(pred_all, y)
        
        # Hit rate
        hit_rate = (np.sign(pred_all) == np.sign(y)).mean()
        
        # Overfitting check
        overfitting_score = train_ic - val_ic
        is_overfitting, overfit_msg = self.detect_overfitting(train_ic, val_ic)
        
        if is_overfitting:
            logger.warning(overfit_msg)
        else:
            logger.info(overfit_msg)
        
        # Feature importance (if available)
        feature_importance = self._extract_feature_importance(model, X)
        
        metrics = ValidationMetrics(
            train_ic=train_ic,
            val_ic=val_ic,
            test_ic=test_ic,
            train_ic_ir=train_ic_ir,
            val_ic_ir=val_ic_ir,
            train_r2=train_r2,
            val_r2=val_r2,
            overfitting_score=overfitting_score,
            hit_rate=hit_rate,
            feature_importance=feature_importance,
            best_params=None
        )
        
        logger.info("\n" + self._format_validation_report(metrics))
        
        return model, metrics
    
    def _hyperparameter_tuning(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        dates: pd.DatetimeIndex,
        splits: List[Tuple[np.ndarray, np.ndarray]],
        param_grid: Dict
    ) -> Tuple[Any, ValidationMetrics]:
        """
        Hyperparameter tuning with GridSearchCV and IC scoring.
        """
        logger.info("\n🎯 HYPERPARAMETER TUNING:")
        logger.info(f"   Parameter Grid: {param_grid}")
        
        # Create IC scorer
        ic_scorer = make_scorer(
            self._ic_scorer_func,
            greater_is_better=True
        )
        
        # GridSearchCV with time-series splits
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=splits,
            scoring=ic_scorer,
            n_jobs=-1,
            verbose=0,
            return_train_score=True
        )
        
        # Fit
        grid_search.fit(X, y)
        
        # Best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        logger.info(f"\n✅ Best Parameters Found:")
        for param, value in best_params.items():
            logger.info(f"   • {param}: {value}")
        
        # Now validate best model
        _, metrics = self._validate_single_model(
            best_model, X, y, dates, splits
        )
        
        # Add best params to metrics
        metrics.best_params = best_params
        
        return best_model, metrics
    
    def _ic_scorer_func(self, y_true, y_pred):
        """Custom scorer for GridSearchCV - returns IC."""
        return self.calculate_information_coefficient(y_pred, y_true)
    
    def _extract_feature_importance(
        self,
        model: Any,
        X: np.ndarray
    ) -> Optional[Dict[str, float]]:
        """
        Extract feature importance from model (if available).
        """
        importance = None
        
        # Tree-based models
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        
        # Linear models (use absolute coefficients)
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
            if importance.ndim > 1:
                importance = importance.flatten()
        
        if importance is not None:
            # Normalize to sum to 1
            importance = importance / importance.sum()
            
            # Return as dict (assuming feature names are f0, f1, ...)
            return {
                f'feature_{i}': float(imp)
                for i, imp in enumerate(importance)
            }
        
        return None
    
    def select_features(
        self,
        feature_importance: Dict[str, float],
        threshold: Optional[float] = None
    ) -> List[str]:
        """
        Select features above importance threshold.
        
        Returns:
            List of selected feature names
        """
        if threshold is None:
            threshold = self.config['feature_selection_threshold']
        
        selected = [
            feat for feat, imp in feature_importance.items()
            if imp >= threshold
        ]
        
        logger.info(f"\n🎯 FEATURE SELECTION:")
        logger.info(f"   • Total features: {len(feature_importance)}")
        logger.info(f"   • Selected features: {len(selected)}")
        logger.info(f"   • Threshold: {threshold:.4f}")
        
        return selected
    
    def _format_validation_report(self, metrics: ValidationMetrics) -> str:
        """
        Format validation metrics as readable report.
        """
        report = f"""
╔══════════════════════════════════════════════════════════╗
║           MODEL VALIDATION REPORT                        ║
╚══════════════════════════════════════════════════════════╝

📊 INFORMATION COEFFICIENT (IC)
   • Train IC:        {metrics.train_ic:>8.4f}
   • Validation IC:   {metrics.val_ic:>8.4f}
   • Test IC:         {metrics.test_ic:>8.4f}
   • IC Threshold:    {self.config['ic_threshold']:>8.4f}
   • Status: {'✅ PASS' if metrics.val_ic >= self.config['ic_threshold'] else '⚠️  WEAK SIGNAL'}

📈 IC INFORMATION RATIO (Stability)
   • Train IC IR:     {metrics.train_ic_ir:>8.4f}
   • Val IC IR:       {metrics.val_ic_ir:>8.4f}
   • IR Threshold:    {self.config['ic_ir_threshold']:>8.4f}
   • Status: {'✅ STABLE' if metrics.val_ic_ir >= self.config['ic_ir_threshold'] else '⚠️  UNSTABLE'}

🎯 OVERFITTING ANALYSIS
   • Train-Val Gap:   {metrics.overfitting_score:>8.4f}
   • Max Allowed Gap: {self.config['overfitting_threshold']:>8.4f}
   • Status: {'✅ NO OVERFIT' if metrics.overfitting_score <= self.config['overfitting_threshold'] else '⚠️  OVERFITTING'}

📊 MODEL FIT
   • Train R²:        {metrics.train_r2:>8.4f}
   • Validation R²:   {metrics.val_r2:>8.4f}
   • Hit Rate:        {metrics.hit_rate:>8.2%}

{'✅ MODEL VALIDATION PASSED' if self._passes_validation(metrics) else '⚠️  MODEL VALIDATION FAILED'}
"""
        return report
    
    def _passes_validation(self, metrics: ValidationMetrics) -> bool:
        """
        Check if model passes all validation criteria.
        """
        checks = [
            metrics.val_ic >= self.config['ic_threshold'],
            metrics.overfitting_score <= self.config['overfitting_threshold'],
            metrics.val_ic_ir >= self.config['ic_ir_threshold'],
        ]
        return all(checks)


if __name__ == "__main__":
    # Quick test
    print("\n🔍 MODEL VALIDATION MODULE")
    print("   Phase 4 - Production Ready\n")
    
    from sklearn.linear_model import Ridge
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(n_samples) * 0.2
    
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Initialize validator
    validator = ModelValidator()
    
    # Create model
    model = Ridge(alpha=1.0)
    
    # Parameter grid for tuning
    param_grid = {
        'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
    }
    
    # Cross-validate with hyperparameter tuning
    best_model, metrics = validator.cross_validate_with_ic(
        model, X, y, dates, param_grid
    )
    
    print("\n✅ Model Validation Test Complete")
