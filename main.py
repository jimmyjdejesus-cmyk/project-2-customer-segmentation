"""
================================================================================
End-to-End Customer Segmentation & Sentiment Intelligence Pipeline
================================================================================
Author: AI & Data Science Engineer
Usage:
  python main.py --samples 600 --clusters 3 --output_dir ./reports
================================================================================
"""

import os
import argparse
import numpy as np
import pandas as pd
from src.preprocessing import CustomerDataPreprocessor
from src.clustering import CustomerClusterEngine
from src.nlp import CustomerSentimentEngine
from src.visualization import CustomerSegmentationPlotter
from src.data_generation import generate_synthetic_data


def run_pipeline(num_samples: int = 600, k: int = 3, output_dir: str = "./reports"):
    """Executes the full data science clustering and NLP intelligence pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print("STARTING CUSTOMER SEGMENTATION & SENTIMENT INTELLIGENCE PIPELINE")
    print("=" * 70)
    
    # 1. Generate & Load Data
    print(f"\n[1/5] Ingesting customer transaction records (n={num_samples})...")
    df = generate_synthetic_data(num_samples=num_samples)
    
    # 2. Preprocessing & Skewness Correction
    print("\n[2/5] Assessing feature skewness & fitting normalization pipeline...")
    features = ["Recency", "Frequency", "Monetary"]
    preprocessor = CustomerDataPreprocessor()
    skew_scores = preprocessor.check_skewness(df, features)
    print(f"      Initial Skewness: {skew_scores}")
    
    X_scaled, df_transformed = preprocessor.fit_transform(df, features)
    print(f"      Features transformed and standardized. Scaled shape: {X_scaled.shape}")
    
    # 3. Clustering Evaluation & Model Fit
    print(f"\n[3/5] Evaluating cluster configurations and fitting K-Means (K={k})...")
    cluster_engine = CustomerClusterEngine()
    eval_df = cluster_engine.evaluate_clusters(X_scaled)
    print(f"      Evaluation Metrics Summary:\n{eval_df.to_string(index=False)}")
    
    cluster_labels = cluster_engine.fit_predict(X_scaled, k=k)
    df["Cluster"] = cluster_labels
    pca_coords = cluster_engine.get_pca_projections(X_scaled)
    print(f"      2D PCA Cumulative Explained Variance: {cluster_engine.explained_variance_ratio:.2f}%")
    
    # 4. NLP Sentiment & Topic Extraction
    print("\n[4/5] Running NLTK VADER sentiment classification & N-gram extraction...")
    sentiment_engine = CustomerSentimentEngine()
    df_analyzed = sentiment_engine.analyze_sentiment(df)
    cluster_topics = sentiment_engine.extract_cluster_topics(df_analyzed, cluster_col="Cluster")
    
    # 5. Profile Aggregation & Report Generation
    print("\n[5/5] Synthesizing behavioral personas & exporting analytical figures...")
    profiles = cluster_engine.generate_cluster_profiles(df_analyzed)
    
    print("\n" + "=" * 70)
    print("EXECUTIVE PERSONA PROFILES & REVENUE CONTRIBUTION")
    print("=" * 70)
    for _, row in profiles.iterrows():
        c_id = row["Cluster"]
        topics = ", ".join(cluster_topics.get(c_id, []))
        print(f"\n[CLUSTER {c_id}] -> {row['Persona']}")
        print(f"  - Customer Share: {row['CustomerSharePct']}% ({row['CustomerCount']} customers)")
        print(f"  - Revenue Share:  {row['RevenueSharePct']}% (${row['TotalMonetary']:,.2f})")
        print(f"  - Mean RFM:       Recency={row['AvgRecency']:.1f}d | Freq={row['AvgFrequency']:.1f} orders | Spend=${row['AvgMonetary']:,.2f}")
        print(f"  - Keyphrase Themes: {topics}")
        
    # Generate Visualizations
    plotter = CustomerSegmentationPlotter()
    plotter.plot_cluster_evaluation(eval_df, save_path=os.path.join(output_dir, "cluster_evaluation.png"))
    plotter.plot_pca_clusters(df_analyzed, pca_coords, save_path=os.path.join(output_dir, "pca_clusters.png"))
    plotter.plot_sentiment_breakdown(df_analyzed, save_path=os.path.join(output_dir, "sentiment_distribution.png"))
    
    # Save CSV Report
    df_analyzed.to_csv(os.path.join(output_dir, "segmented_customers_with_sentiment.csv"), index=False)
    print(f"\n✓ Artifacts successfully saved to '{output_dir}'.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer Segmentation & Sentiment Analysis Pipeline")
    parser.add_argument("--samples", type=int, default=600, help="Number of customer samples to generate")
    parser.add_argument("--clusters", type=int, default=3, help="Number of K-Means clusters")
    parser.add_argument("--output_dir", type=str, default="./reports", help="Directory to save plots and reports")
    args = parser.parse_args()
    
    run_pipeline(num_samples=args.samples, k=args.clusters, output_dir=args.output_dir)
