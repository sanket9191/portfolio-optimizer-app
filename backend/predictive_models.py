import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class PredictiveAlphaModel:
    """
    Institutional-grade cross-sectional return prediction model.
    
    Key Features:
    - Time-series aware cross-validation
    - Robust scaling to handle outliers
    - Multiple model types with regularization
    - Information Coefficient (IC) tracking
    - Feature importance analysis
    - Proper train/test split (no look-ahead)
    """
    
    MODEL_REGISTRY = {
        'ridge': Ridge,
        'lasso': Lasso,
        'elastic_net': ElasticNet,
        'random_forest': RandomForestRegressor,
        'gradient_boosting': GradientBoostingRegressor
    }
    
    def __init__(self, model_type='ridge', horizon_months=3, alpha=1.0):
        """
        Initialize predictive model.
        
        Parameters:
        -----------
        model_type : str
            Type of model: 'ridge', 'lasso', 'elastic_net', 'random_forest', 'gradient_boosting'
        horizon_months : int
            Forecasting horizon in months
        alpha : float
            Regularization strength (for linear models)
        """
        self.model_type = model_type
        self.horizon_months = horizon_months
        self.alpha = alpha
        
        # Initialize model
        self.model = self._initialize_model()
        
        # Use RobustScaler (handles outliers better than StandardScaler)
        self.scaler = RobustScaler()
        
        # Performance tracking
        self.training_history = []
        self.feature_importance = None
        
        print(f"\nInitialized {model_type} model for {horizon_months}-month ahead forecasting")
    
    def _initialize_model(self):
        """Initialize model based on type."""
        if self.model_type not in self.MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        ModelClass = self.MODEL_REGISTRY[self.model_type]
        
        if self.model_type == 'ridge':
            # Ridge with CV-selected alpha
            return ModelClass(alpha=self.alpha, fit_intercept=True)
        
        elif self.model_type == 'lasso':
            # LASSO for feature selection
            return ModelClass(alpha=self.alpha, fit_intercept=True, max_iter=10000)
        
        elif self.model_type == 'elastic_net':
            # Elastic Net (L1 + L2)
            return ModelClass(alpha=self.alpha, l1_ratio=0.5, fit_intercept=True, max_iter=10000)
        
        elif self.model_type == 'random_forest':
            # Random Forest with regularization
            return ModelClass(
                n_estimators=100,
                max_depth=5,  # Shallow trees to prevent overfitting
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
        
        elif self.model_type == 'gradient_boosting':
            # Gradient Boosting with conservative parameters
            return ModelClass(
                n_estimators=100,
                learning_rate=0.05,  # Slow learning
                max_depth=3,  # Very shallow trees
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=0.8,  # Stochastic boosting
                random_state=42
            )
    
    def prepare_training_data(self, features_df, price_df, rebalance_dates):
        """
        Prepare training dataset with proper forward returns.
        
        Critical: This ensures no look-ahead bias.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            Technical features with (date, ticker) multi-index
        price_df : pd.DataFrame
            Price data (wide format: dates x tickers)
        rebalance_dates : pd.DatetimeIndex
            Dates at which we compute features and need predictions
        
        Returns:
        --------
        X : np.ndarray
            Feature matrix (n_samples x n_features)
        y : np.ndarray
            Target forward returns
        dates : np.ndarray
            Corresponding dates for time-series split
        tickers : np.ndarray
            Corresponding tickers
        """
        X_list, y_list, date_list, ticker_list = [], [], [], []
        
        for rebal_date in rebalance_dates:
            # Get features at this date
            if rebal_date not in features_df.index.get_level_values('date'):
                continue
            
            features_at_date = features_df.loc[rebal_date]
            
            # Compute forward returns for each ticker
            for ticker in features_at_date.index:
                if ticker not in price_df.columns:
                    continue
                
                # Get forward return
                forward_return = self._compute_forward_return(
                    price_df[ticker],
                    rebal_date
                )
                
                if forward_return is None:
                    continue
                
                # Append to dataset
                X_list.append(features_at_date.loc[ticker].values)
                y_list.append(forward_return)
                date_list.append(rebal_date)
                ticker_list.append(ticker)
        
        X = np.array(X_list)
        y = np.array(y_list)
        dates = np.array(date_list)
        tickers = np.array(ticker_list)
        
        # Handle NaN values (fill with median)
        X = pd.DataFrame(X).fillna(pd.DataFrame(X).median()).values
        
        print(f"  Training data prepared: {len(X)} samples, {X.shape[1]} features")
        
        return X, y, dates, tickers
    
    def _compute_forward_return(self, price_series, start_date):
        """
        Compute forward return from start_date to start_date + horizon.
        
        Returns None if insufficient data.
        """
        try:
            # Get price at start date
            start_idx = price_series.index.get_loc(start_date)
            start_price = price_series.iloc[start_idx]
            
            # Approximate end date (horizon months ahead)
            # Use ~21 trading days per month
            horizon_days = self.horizon_months * 21
            end_idx = start_idx + horizon_days
            
            if end_idx >= len(price_series):
                return None
            
            end_price = price_series.iloc[end_idx]
            
            # Return
            forward_return = (end_price / start_price) - 1.0
            
            # Winsorize extreme returns (prevent outliers from dominating)
            return np.clip(forward_return, -0.5, 0.5)
        
        except Exception:
            return None
    
    def train_with_cross_validation(self, X, y, dates, n_splits=3):
        """
        Train model with time-series cross-validation.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target returns
        dates : np.ndarray
            Dates for proper time-series split
        n_splits : int
            Number of CV splits
        
        Returns:
        --------
        dict : Cross-validation results (IC, RMSE, R²)
        """
        print(f"\n  Cross-validating {self.model_type} model...")
        
        # Sort by date (critical for time-series split)
        sort_idx = np.argsort(dates)
        X_sorted = X[sort_idx]
        y_sorted = y[sort_idx]
        
        # Time-series split
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_ics, cv_rmses, cv_r2s = [], [], []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_sorted)):
            # Split data
            X_train, X_test = X_sorted[train_idx], X_sorted[test_idx]
            y_train, y_test = y_sorted[train_idx], y_sorted[test_idx]
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = self._initialize_model()
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            
            # Metrics
            ic = np.corrcoef(y_pred, y_test)[0, 1]  # Information Coefficient
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            cv_ics.append(ic)
            cv_rmses.append(rmse)
            cv_r2s.append(r2)
            
            print(f"    Fold {fold+1}/{n_splits}: IC={ic:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
        
        results = {
            'mean_ic': np.mean(cv_ics),
            'std_ic': np.std(cv_ics),
            'mean_rmse': np.mean(cv_rmses),
            'mean_r2': np.mean(cv_r2s)
        }
        
        print(f"\n  CV Results: IC={results['mean_ic']:.4f}±{results['std_ic']:.4f}, " +
              f"RMSE={results['mean_rmse']:.4f}, R²={results['mean_r2']:.4f}")
        
        return results
    
    def train(self, X, y):
        """
        Train model on full dataset (after CV validation).
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target returns
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        # Compute in-sample IC
        y_pred = self.model.predict(X_scaled)
        ic = np.corrcoef(y_pred, y)[0, 1]
        
        # Extract feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            self.feature_importance = np.abs(self.model.coef_)
        
        print(f"  Model trained. In-sample IC: {ic:.4f}")
        
        return ic
    
    def predict(self, X):
        """
        Generate return forecasts.
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Features for prediction
        
        Returns:
        --------
        np.ndarray : Predicted returns
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Handle NaN
        X = pd.DataFrame(X).fillna(0).values
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def get_feature_importance(self, feature_names=None, top_n=10):
        """
        Get top N most important features.
        
        Returns:
        --------
        pd.DataFrame : Feature importance rankings
        """
        if self.feature_importance is None:
            return None
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(self.feature_importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)


class EnsembleAlphaModel:
    """
    Ensemble of multiple alpha models for robustness.
    
    Combines predictions from multiple models to reduce model risk.
    """
    
    def __init__(self, model_configs, ensemble_method='mean'):
        """
        Initialize ensemble.
        
        Parameters:
        -----------
        model_configs : list of dict
            List of model configurations
            Example: [{'type': 'ridge', 'alpha': 1.0}, {'type': 'random_forest'}]
        ensemble_method : str
            'mean', 'median', or 'weighted'
        """
        self.models = []
        for config in model_configs:
            model = PredictiveAlphaModel(
                model_type=config.get('type', 'ridge'),
                horizon_months=config.get('horizon_months', 3),
                alpha=config.get('alpha', 1.0)
            )
            self.models.append(model)
        
        self.ensemble_method = ensemble_method
        self.weights = None  # For weighted ensemble
        
        print(f"\nInitialized ensemble with {len(self.models)} models")
    
    def prepare_training_data(self, features_df, price_df, rebalance_dates):
        """
        Prepare training data using the first model's method.
        All models in ensemble use the same training data.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            Technical features with (date, ticker) multi-index
        price_df : pd.DataFrame
            Price data (wide format: dates x tickers)
        rebalance_dates : pd.DatetimeIndex
            Dates at which we compute features and need predictions
        
        Returns:
        --------
        X, y, dates, tickers : Training data components
        """
        # Delegate to the first model's prepare_training_data method
        return self.models[0].prepare_training_data(features_df, price_df, rebalance_dates)
    
    def train_with_cross_validation(self, X, y, dates, n_splits=3):
        """Train ensemble with cross-validation and return aggregated results."""
        all_ics = []
        all_rmses = []
        all_r2s = []
        
        for i, model in enumerate(self.models):
            print(f"\nCross-validating model {i+1}/{len(self.models)}: {model.model_type}")
            cv_results = model.train_with_cross_validation(X, y, dates, n_splits)
            all_ics.append(cv_results['mean_ic'])
            all_rmses.append(cv_results['mean_rmse'])
            all_r2s.append(cv_results['mean_r2'])
        
        # Set weights based on IC (if weighted ensemble)
        if self.ensemble_method == 'weighted':
            ic_array = np.array(all_ics)
            ic_array = np.maximum(ic_array, 0)  # Only positive ICs get weight
            self.weights = ic_array / ic_array.sum() if ic_array.sum() > 0 else np.ones(len(all_ics)) / len(all_ics)
            print(f"\nEnsemble weights (based on CV IC): {self.weights}")
        
        # Return ensemble-level results
        ensemble_results = {
            'mean_ic': np.mean(all_ics),
            'std_ic': np.std(all_ics),
            'mean_rmse': np.mean(all_rmses),
            'mean_r2': np.mean(all_r2s)
        }
        
        print(f"\n  Ensemble CV Results: IC={ensemble_results['mean_ic']:.4f}±{ensemble_results['std_ic']:.4f}")
        
        return ensemble_results
    
    def train(self, X, y):
        """
        Train all models in ensemble on full dataset.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target returns
        
        Note: This method has the same signature as PredictiveAlphaModel.train()
        to maintain API consistency.
        """
        ics = []
        
        for i, model in enumerate(self.models):
            print(f"\nTraining model {i+1}/{len(self.models)}: {model.model_type} on full dataset")
            ic = model.train(X, y)
            ics.append(ic)
        
        # Update weights if using weighted ensemble (based on in-sample IC)
        if self.ensemble_method == 'weighted' and self.weights is None:
            ic_array = np.array(ics)
            ic_array = np.maximum(ic_array, 0)  # Only positive ICs get weight
            self.weights = ic_array / ic_array.sum() if ic_array.sum() > 0 else np.ones(len(ics)) / len(ics)
            print(f"\nEnsemble weights (based on in-sample IC): {self.weights}")
        
        mean_ic = np.mean(ics)
        print(f"\n  Ensemble trained. Mean in-sample IC: {mean_ic:.4f}")
        return mean_ic
    
    def predict(self, X):
        """Generate ensemble predictions."""
        predictions = np.array([model.predict(X) for model in self.models])
        
        if self.ensemble_method == 'mean':
            return predictions.mean(axis=0)
        elif self.ensemble_method == 'median':
            return np.median(predictions, axis=0)
        elif self.ensemble_method == 'weighted':
            return (predictions.T @ self.weights).flatten()
        else:
            return predictions.mean(axis=0)
    
    def get_feature_importance(self, feature_names=None, top_n=10):
        """Aggregate feature importance across all models."""
        # Get feature importance from models that support it
        importance_list = []
        for model in self.models:
            if model.feature_importance is not None:
                importance_list.append(model.feature_importance)
        
        if len(importance_list) == 0:
            return None
        
        # Average importance across models
        avg_importance = np.mean(importance_list, axis=0)
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(avg_importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': avg_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
