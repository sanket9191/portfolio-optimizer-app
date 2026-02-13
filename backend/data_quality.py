"""
Institutional-Grade Data Quality Module
Phase 4: Production-Ready Data Validation & Cleaning

Author: Senior VP - Quantitative Engineering  
Purpose: Data quality checks, missing data handling, outlier detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats.mstats import winsorize
from dataclasses import dataclass
import warnings
import logging

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Container for data quality metrics"""
    total_rows: int
    total_cols: int
    missing_pct: float
    duplicate_rows: int
    outliers_detected: int
    stale_tickers: List[str]
    problematic_features: List[str]
    data_staleness_days: int
    passes_quality_checks: bool
    issues: List[str]


class DataQualityManager:
    """
    Enterprise-grade data quality management.
    
    Features:
    - Missing data detection and imputation
    - Outlier detection (Z-score, IQR, MAD)
    - Winsorization for extreme values
    - Data staleness checks
    - Cross-sectional consistency validation
    - Feature correlation analysis
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize data quality manager.
        
        Args:
            config: Quality control parameters
                - max_missing_pct: Max % missing allowed (default: 0.20)
                - max_staleness_days: Max days old for data (default: 7)
                - outlier_std_threshold: Z-score for outliers (default: 4.0)
                - winsorize_limits: Winsorization limits (default: (0.01, 0.99))
                - min_observations: Min data points per feature (default: 252)
        """
        default_config = {
            'max_missing_pct': 0.20,  # 20% max missing
            'max_staleness_days': 7,   # Max 7 days old
            'outlier_std_threshold': 4.0,  # 4-sigma outliers
            'winsorize_limits': (0.01, 0.99),  # 1% and 99% percentiles
            'min_observations': 252,   # Min 1 year data
            'forward_fill_limit': 5,   # Max 5 days forward fill
            'high_correlation_threshold': 0.95,  # Flag highly correlated features
        }
        
        self.config = {**default_config, **(config or {})}
        self.quality_reports = []
        
    def validate_data(
        self,
        data: pd.DataFrame,
        data_date: Optional[pd.Timestamp] = None
    ) -> DataQualityReport:
        """
        Comprehensive data quality validation.
        
        This is the MAIN quality check function.
        
        Args:
            data: DataFrame to validate (can be features or returns)
            data_date: Date of data (if None, uses latest date in index)
            
        Returns:
            DataQualityReport with all quality metrics
        """
        logger.info("\n" + "="*60)
        logger.info("🔍 DATA QUALITY VALIDATION")
        logger.info("="*60)
        
        issues = []
        
        # Basic stats
        total_rows = len(data)
        total_cols = len(data.columns)
        
        logger.info(f"   • Rows: {total_rows:,}")
        logger.info(f"   • Columns: {total_cols}")
        
        # Check 1: Missing data
        missing_pct = data.isnull().sum().sum() / (total_rows * total_cols)
        logger.info(f"\n🕵️  Missing Data: {missing_pct:.2%}")
        
        if missing_pct > self.config['max_missing_pct']:
            issues.append(f"High missing data: {missing_pct:.2%} > {self.config['max_missing_pct']:.2%}")
        
        # Check 2: Duplicate rows
        duplicate_rows = data.duplicated().sum()
        logger.info(f"   • Duplicate rows: {duplicate_rows}")
        
        if duplicate_rows > 0:
            issues.append(f"Found {duplicate_rows} duplicate rows")
        
        # Check 3: Outliers
        outliers_detected = self._count_outliers(data)
        logger.info(f"   • Outliers detected: {outliers_detected}")
        
        # Check 4: Data staleness
        if data_date is None:
            if isinstance(data.index, pd.DatetimeIndex):
                data_date = data.index.max()
            elif isinstance(data.index, pd.MultiIndex) and isinstance(data.index.levels[0], pd.DatetimeIndex):
                data_date = data.index.get_level_values(0).max()
            else:
                data_date = pd.Timestamp.now()
        
        staleness_days = (pd.Timestamp.now() - data_date).days
        logger.info(f"\n📅 Data Staleness: {staleness_days} days old")
        
        if staleness_days > self.config['max_staleness_days']:
            issues.append(
                f"Data is stale: {staleness_days} days > {self.config['max_staleness_days']} days"
            )
        
        # Check 5: Features with too much missing data
        problematic_features = []
        col_missing = data.isnull().mean()
        for col in col_missing.index:
            if col_missing[col] > self.config['max_missing_pct']:
                problematic_features.append(col)
        
        if problematic_features:
            logger.info(f"\n⚠️  Problematic Features ({len(problematic_features)}):")
            for feat in problematic_features[:10]:  # Show first 10
                logger.info(f"   • {feat}: {col_missing[feat]:.2%} missing")
            issues.append(f"{len(problematic_features)} features with >{self.config['max_missing_pct']:.0%} missing")
        
        # Check 6: Stale tickers (if MultiIndex with ticker level)
        stale_tickers = []
        if isinstance(data.index, pd.MultiIndex):
            stale_tickers = self._detect_stale_tickers(data)
            if stale_tickers:
                logger.info(f"\n⚠️  Stale Tickers ({len(stale_tickers)}): {stale_tickers[:5]}")
                issues.append(f"{len(stale_tickers)} tickers with stale/missing data")
        
        # Final verdict
        passes_quality_checks = len(issues) == 0
        
        report = DataQualityReport(
            total_rows=total_rows,
            total_cols=total_cols,
            missing_pct=missing_pct,
            duplicate_rows=duplicate_rows,
            outliers_detected=outliers_detected,
            stale_tickers=stale_tickers,
            problematic_features=problematic_features,
            data_staleness_days=staleness_days,
            passes_quality_checks=passes_quality_checks,
            issues=issues
        )
        
        logger.info("\n" + self._format_quality_report(report))
        
        self.quality_reports.append(report)
        
        return report
    
    def handle_missing_data(
        self,
        data: pd.DataFrame,
        method: str = 'hybrid'
    ) -> pd.DataFrame:
        """
        Intelligent missing data imputation.
        
        Args:
            data: DataFrame with missing values
            method: 'ffill', 'median', 'hybrid' (default)
            
        Returns:
            DataFrame with imputed values
        """
        logger.info(f"\n🛠️  MISSING DATA IMPUTATION (method={method})")
        
        data_clean = data.copy()
        
        if method == 'ffill':
            # Forward fill with limit
            if isinstance(data.index, pd.MultiIndex):
                data_clean = data_clean.groupby(level='ticker').ffill(
                    limit=self.config['forward_fill_limit']
                )
            else:
                data_clean = data_clean.ffill(limit=self.config['forward_fill_limit'])
        
        elif method == 'median':
            # Cross-sectional median
            if isinstance(data.index, pd.MultiIndex):
                for col in data_clean.columns:
                    data_clean[col] = data_clean.groupby(level='date')[col].transform(
                        lambda x: x.fillna(x.median())
                    )
            else:
                data_clean = data_clean.fillna(data_clean.median())
        
        elif method == 'hybrid':
            # 1. Forward fill with limit
            if isinstance(data.index, pd.MultiIndex):
                data_clean = data_clean.groupby(level='ticker').ffill(
                    limit=self.config['forward_fill_limit']
                )
                
                # 2. Cross-sectional median for remaining
                for col in data_clean.columns:
                    data_clean[col] = data_clean.groupby(level='date')[col].transform(
                        lambda x: x.fillna(x.median())
                    )
            else:
                data_clean = data_clean.ffill(limit=self.config['forward_fill_limit'])
                data_clean = data_clean.fillna(data_clean.median())
        
        # Final: Fill any remaining NaNs with 0
        data_clean = data_clean.fillna(0)
        
        missing_before = data.isnull().sum().sum()
        missing_after = data_clean.isnull().sum().sum()
        
        logger.info(f"   • Missing before: {missing_before:,}")
        logger.info(f"   • Missing after: {missing_after:,}")
        logger.info(f"   • Imputed: {missing_before - missing_after:,}")
        
        return data_clean
    
    def detect_and_handle_outliers(
        self,
        data: pd.DataFrame,
        method: str = 'winsorize',
        detection: str = 'zscore'
    ) -> pd.DataFrame:
        """
        Detect and handle outliers.
        
        Args:
            data: DataFrame to clean
            method: 'winsorize', 'clip', or 'remove'
            detection: 'zscore', 'iqr', or 'mad'
            
        Returns:
            DataFrame with outliers handled
        """
        logger.info(f"\n🕵️  OUTLIER DETECTION & HANDLING")
        logger.info(f"   • Detection method: {detection}")
        logger.info(f"   • Handling method: {method}")
        
        data_clean = data.copy()
        
        if method == 'winsorize':
            # Winsorize each column
            for col in data_clean.columns:
                if data_clean[col].dtype in ['float64', 'int64']:
                    data_clean[col] = winsorize(
                        data_clean[col].values,
                        limits=self.config['winsorize_limits'],
                        nan_policy='omit'
                    )
        
        elif method == 'clip':
            # Clip based on detection method
            for col in data_clean.columns:
                if data_clean[col].dtype in ['float64', 'int64']:
                    if detection == 'zscore':
                        z_scores = np.abs(stats.zscore(data_clean[col], nan_policy='omit'))
                        threshold = self.config['outlier_std_threshold']
                        outliers = z_scores > threshold
                    
                    elif detection == 'iqr':
                        Q1 = data_clean[col].quantile(0.25)
                        Q3 = data_clean[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower = Q1 - 3 * IQR
                        upper = Q3 + 3 * IQR
                        outliers = (data_clean[col] < lower) | (data_clean[col] > upper)
                    
                    # Clip outliers to reasonable range
                    if outliers.any():
                        p1 = data_clean[col].quantile(self.config['winsorize_limits'][0])
                        p99 = data_clean[col].quantile(self.config['winsorize_limits'][1])
                        data_clean[col] = data_clean[col].clip(lower=p1, upper=p99)
        
        outliers_removed = self._count_outliers(data) - self._count_outliers(data_clean)
        logger.info(f"   • Outliers handled: {outliers_removed}")
        
        return data_clean
    
    def _count_outliers(self, data: pd.DataFrame) -> int:
        """
        Count outliers using Z-score method.
        """
        outliers = 0
        
        for col in data.columns:
            if data[col].dtype in ['float64', 'int64']:
                try:
                    z_scores = np.abs(stats.zscore(data[col].dropna()))
                    outliers += (z_scores > self.config['outlier_std_threshold']).sum()
                except:
                    continue
        
        return outliers
    
    def _detect_stale_tickers(
        self,
        data: pd.DataFrame,
        min_data_points: Optional[int] = None
    ) -> List[str]:
        """
        Detect tickers with insufficient or stale data.
        """
        if min_data_points is None:
            min_data_points = self.config['min_observations']
        
        stale_tickers = []
        
        if isinstance(data.index, pd.MultiIndex):
            # Count data points per ticker
            ticker_counts = data.groupby(level='ticker').size()
            
            for ticker, count in ticker_counts.items():
                if count < min_data_points:
                    stale_tickers.append(ticker)
        
        return stale_tickers
    
    def check_feature_correlation(
        self,
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Check for highly correlated features.
        
        Returns:
            DataFrame of highly correlated feature pairs
        """
        logger.info("\n🔗 FEATURE CORRELATION ANALYSIS")
        
        corr_matrix = features.corr().abs()
        
        # Get upper triangle (avoid duplicates)
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find highly correlated pairs
        high_corr = []
        threshold = self.config['high_correlation_threshold']
        
        for column in upper.columns:
            high_corr_features = upper.index[upper[column] > threshold].tolist()
            for feat in high_corr_features:
                high_corr.append({
                    'feature_1': column,
                    'feature_2': feat,
                    'correlation': corr_matrix.loc[column, feat]
                })
        
        if high_corr:
            high_corr_df = pd.DataFrame(high_corr).sort_values(
                'correlation', ascending=False
            )
            
            logger.info(f"   • Highly correlated pairs: {len(high_corr)}")
            logger.info("\n   Top 5 correlated pairs:")
            for _, row in high_corr_df.head(5).iterrows():
                logger.info(f"      {row['feature_1']} <-> {row['feature_2']}: {row['correlation']:.3f}")
            
            return high_corr_df
        else:
            logger.info("   ✅ No highly correlated features found")
            return pd.DataFrame()
    
    def _format_quality_report(self, report: DataQualityReport) -> str:
        """
        Format data quality report.
        """
        status = "✅ PASSED" if report.passes_quality_checks else "⚠️  FAILED"
        
        issues_str = "\n".join([f"   • {issue}" for issue in report.issues])
        if not issues_str:
            issues_str = "   • None"
        
        result = f"""
╔══════════════════════════════════════════════════════════╗
║           DATA QUALITY REPORT - {status}             ║
╚══════════════════════════════════════════════════════════╝

📊 DATA OVERVIEW
   • Total rows: {report.total_rows:,}
   • Total columns: {report.total_cols}
   • Missing data: {report.missing_pct:.2%}
   • Duplicate rows: {report.duplicate_rows}
   • Outliers detected: {report.outliers_detected}

📅 DATA FRESHNESS
   • Data age: {report.data_staleness_days} days
   • Max allowed: {self.config['max_staleness_days']} days
   • Status: {'✅ Fresh' if report.data_staleness_days <= self.config['max_staleness_days'] else '⚠️  Stale'}

⚠️  ISSUES DETECTED ({len(report.issues)}):
{issues_str}
"""
        return result


if __name__ == "__main__":
    # Quick test
    print("\n🔍 DATA QUALITY MODULE")
    print("   Phase 4 - Production Ready\n")
    
    # Create sample data with quality issues
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    
    # Create MultiIndex data
    index = pd.MultiIndex.from_product(
        [dates, tickers],
        names=['date', 'ticker']
    )
    
    data = pd.DataFrame(
        np.random.randn(len(index), 10) * 0.02,
        index=index,
        columns=[f'feature_{i}' for i in range(10)]
    )
    
    # Inject quality issues
    # 1. Missing data
    data.iloc[::20] = np.nan
    
    # 2. Outliers
    data.iloc[50, 0] = 10.0  # Extreme outlier
    data.iloc[100, 1] = -8.0
    
    # Initialize quality manager
    qm = DataQualityManager()
    
    # Validate
    report = qm.validate_data(data)
    
    # Clean
    print("\n🧹 CLEANING DATA...")
    data_clean = qm.handle_missing_data(data)
    data_clean = qm.detect_and_handle_outliers(data_clean)
    
    # Validate again
    print("\n🔍 RE-VALIDATING CLEANED DATA...")
    report_clean = qm.validate_data(data_clean)
    
    print("\n✅ Data Quality Test Complete")
