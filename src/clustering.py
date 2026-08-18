"""
================================================================================
Unsupervised K-Means Clustering & Metric Optimization Engine
================================================================================
Author: AI & Data Science Engineer
Purpose:
  - Multi-metric evaluation: Inertia (Elbow), Silhouette, Davies-Bouldin, Calinski-Harabasz
  - Automated or parametric cluster selection
  - Behavioral customer persona profiling & revenue attribution
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA


class CustomerClusterEngine:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.best_k = None
        self.model = None
        self.pca = PCA(n_components=2, random_state=random_state)
        self.evaluation_results = {}

    def evaluate_clusters(self, X: np.ndarray, k_range: range = range(2, 9)) -> pd.DataFrame:
        """
        Runs K-Means across a range of clusters and computes 4 mathematical cluster quality metrics:
          1. WCSS / Inertia: Intra-cluster variance (lower is tighter)
          2. Silhouette Score: Distance separation [-1, +1] (higher is better)
          3. Davies-Bouldin Index: Ratio of within to between cluster distances (lower is better)
          4. Calinski-Harabasz: Variance ratio criterion (higher is better)
        """
        metrics = []
        
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=15, max_iter=300)
            labels = km.fit_predict(X)
            
            wcss = km.inertia_
            sil = silhouette_score(X, labels)
            db = davies_bouldin_score(X, labels)
            ch = calinski_harabasz_score(X, labels)
            
            metrics.append({
                "K": k,
                "WCSS": round(wcss, 2),
                "Silhouette": round(sil, 4),
                "Davies_Bouldin": round(db, 4),
                "Calinski_Harabasz": round(ch, 2)
            })
            
        eval_df = pd.DataFrame(metrics)
        self.evaluation_results = eval_df
        
        # Determine highest silhouette K as default optimal
        self.best_k = int(eval_df.loc[eval_df["Silhouette"].idxmax(), "K"])
        return eval_df

    def fit_predict(self, X: np.ndarray, k: int = None) -> np.ndarray:
        """
        Fits K-Means with specified or optimal K and computes 2D PCA projections.
        """
        target_k = k or self.best_k or 3
        self.model = KMeans(n_clusters=target_k, random_state=self.random_state, n_init=15, max_iter=300)
        labels = self.model.fit_predict(X)
        self.pca.fit(X)
        self.best_k = target_k
        return labels

    def get_pca_projections(self, X: np.ndarray) -> np.ndarray:
        """
        Projects high-dimensional standardized space to 2D for plotting.
        """
        return self.pca.transform(X)

    @property
    def explained_variance_ratio(self) -> float:
        """Returns the cumulative explained variance of the 2D PCA projection."""
        if not hasattr(self.pca, 'explained_variance_ratio_'):
            return 0.0
        return float(np.sum(self.pca.explained_variance_ratio_) * 100)

    def generate_cluster_profiles(self, df: pd.DataFrame, cluster_col: str = "Cluster") -> pd.DataFrame:
        """
        Generates executive summary profiles for each cluster including mean RFM stats,
        customer share, and total revenue contribution.
        """
        # Compute global population medians for persona thresholds
        global_monetary_median = df['Monetary'].median()
        global_frequency_median = df['Frequency'].median()
        global_recency_median = df['Recency'].median()
        
        summary = df.groupby(cluster_col).agg(
            CustomerCount=("CustomerID", "count"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgMonetary=("Monetary", "mean"),
            TotalMonetary=("Monetary", "sum")
        ).reset_index()
        
        total_rev = df["Monetary"].sum()
        total_cust = len(df)
        
        summary["CustomerSharePct"] = round((summary["CustomerCount"] / total_cust) * 100, 1)
        summary["RevenueSharePct"] = round((summary["TotalMonetary"] / total_rev) * 100, 1)
        
        # Assign Descriptive Personas
        def label_persona(row):
            if row["AvgMonetary"] > global_monetary_median * 1.25 and row["AvgFrequency"] >= global_frequency_median:
                return "VIP Champions (High Frequency & Spend)"
            elif row["AvgRecency"] > global_recency_median * 1.2:
                return "At Risk / Churning (High Recency, Low Activity)"
            else:
                return "Active Prospects (Recent Engagement, Moderate Volume)"
                
        summary["Persona"] = summary.apply(label_persona, axis=1)
        return summary
