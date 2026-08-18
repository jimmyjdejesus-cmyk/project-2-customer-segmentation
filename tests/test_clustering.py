"""
================================================================================
Unit Tests for Clustering & Preprocessing Pipelines
================================================================================
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import CustomerDataPreprocessor
from src.clustering import CustomerClusterEngine


@pytest.fixture
def sample_rfm_df():
    """Generates synthetic RFM dataset for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "CustomerID": [f"CUST-{i:03d}" for i in range(1, 101)],
        "Recency": np.random.exponential(scale=40, size=100),
        "Frequency": np.random.poisson(lam=3, size=100) + 1,
        "Monetary": np.random.exponential(scale=200, size=100) + 20.0
    })


def test_preprocessor_skewness_handling(sample_rfm_df):
    """Verifies that preprocessor detects skew and scales features."""
    preprocessor = CustomerDataPreprocessor()
    features = ["Recency", "Frequency", "Monetary"]
    
    skew_scores = preprocessor.check_skewness(sample_rfm_df, features)
    assert len(skew_scores) == 3
    
    scaled_array, df_clean = preprocessor.fit_transform(sample_rfm_df, features)
    assert scaled_array.shape == (100, 3)
    # Scaled data should have approx zero mean and unit variance
    assert np.allclose(scaled_array.mean(axis=0), 0, atol=1e-2)
    assert np.allclose(scaled_array.std(axis=0), 1, atol=1e-2)


def test_clustering_evaluation_metrics(sample_rfm_df):
    """Verifies calculation of Elbow, Silhouette, and Davies-Bouldin metrics."""
    preprocessor = CustomerDataPreprocessor()
    scaled_array, _ = preprocessor.fit_transform(sample_rfm_df, ["Recency", "Frequency", "Monetary"])
    
    engine = CustomerClusterEngine(random_state=42)
    eval_df = engine.evaluate_clusters(scaled_array, k_range=range(2, 5))
    
    assert len(eval_df) == 3
    assert set(eval_df.columns) == {"K", "WCSS", "Silhouette", "Davies_Bouldin", "Calinski_Harabasz"}
    assert (eval_df["Silhouette"] >= -1.0).all() and (eval_df["Silhouette"] <= 1.0).all()
    assert (eval_df["Davies_Bouldin"] >= 0).all()


def test_fit_predict_and_pca(sample_rfm_df):
    """Verifies K-Means label generation and 2D PCA dimensionality reduction."""
    preprocessor = CustomerDataPreprocessor()
    scaled_array, _ = preprocessor.fit_transform(sample_rfm_df, ["Recency", "Frequency", "Monetary"])
    
    engine = CustomerClusterEngine(random_state=42)
    labels = engine.fit_predict(scaled_array, k=3)
    
    assert len(labels) == 100
    assert set(labels) == {0, 1, 2}
    
    pca_coords = engine.get_pca_projections(scaled_array)
    assert pca_coords.shape == (100, 2)
    assert engine.explained_variance_ratio > 50.0


def test_cluster_profiling(sample_rfm_df):
    """Verifies aggregation of persona profiles and revenue attribution."""
    sample_rfm_df["Cluster"] = np.random.choice([0, 1, 2], size=len(sample_rfm_df))
    engine = CustomerClusterEngine()
    profiles = engine.generate_cluster_profiles(sample_rfm_df, cluster_col="Cluster")
    
    assert len(profiles) == 3
    assert "RevenueSharePct" in profiles.columns
    assert "Persona" in profiles.columns
    assert np.isclose(profiles["RevenueSharePct"].sum(), 100.0, atol=1.0)
