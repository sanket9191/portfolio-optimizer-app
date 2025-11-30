import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def perform_clustering(df, n_clusters=4):
    """
    Perform K-means clustering on the latest month's data
    
    Parameters:
    - df: DataFrame with features (multi-index: date, ticker)
    - n_clusters: Number of clusters
    
    Returns:
    - Dictionary with cluster labels, centroids, and metrics
    """
    try:
        # Get the latest month's data
        latest_date = df.index.get_level_values('date').max()
        latest_data = df.loc[latest_date]
        
        print(f"Clustering data from {latest_date} with {len(latest_data)} stocks")
        
        # Select features for clustering (exclude return columns)
        feature_cols = [col for col in latest_data.columns if not col.startswith('return')]
        X = latest_data[feature_cols].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform K-means
        kmeans = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            n_init=10,
            random_state=42
        )
        labels = kmeans.fit_predict(X_scaled)
        
        # Calculate silhouette score
        silhouette = silhouette_score(X_scaled, labels)
        
        # Get cluster statistics
        cluster_stats = []
        for i in range(n_clusters):
            cluster_mask = labels == i
            cluster_stocks = latest_data.index[cluster_mask].tolist()
            cluster_stats.append({
                'cluster_id': int(i),
                'n_stocks': int(np.sum(cluster_mask)),
                'stocks': cluster_stocks,
                'avg_rsi': float(latest_data.loc[cluster_mask, 'rsi'].mean()),
                'avg_volatility': float(latest_data.loc[cluster_mask, 'garman_klass_vol'].mean())
            })
        
        results = {
            'n_clusters': n_clusters,
            'silhouette_score': float(silhouette),
            'labels': labels.tolist(),
            'cluster_stats': cluster_stats,
            'tickers': latest_data.index.tolist()
        }
        
        print(f"Clustering complete. Silhouette score: {silhouette:.3f}")
        
        return results
        
    except Exception as e:
        print(f"Error in clustering: {str(e)}")
        raise