"""
================================================================================
Visualization & Plotting Diagnostics Module
================================================================================
Author: AI & Data Science Engineer
Purpose:
  - Generates publication-ready diagnostic charts
  - Elbow, Silhouette, PCA scatter, and stacked sentiment distributions
================================================================================
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class CustomerSegmentationPlotter:
    def __init__(self):
        sns.set_theme(style="whitegrid")
        plt.rcParams["font.sans-serif"] = "Arial"

    def plot_cluster_evaluation(self, eval_df: pd.DataFrame, save_path: str = None):
        """Plots the Elbow Method (WCSS) and Silhouette score curves side-by-side."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(eval_df["K"], eval_df["WCSS"], marker="o", color="#0284c7", linewidth=2)
        ax1.set_title("Elbow Method: Within-Cluster Sum of Squares (Inertia)", fontweight="bold")
        ax1.set_xlabel("Cluster Count (K)")
        ax1.set_ylabel("WCSS / Inertia")
        
        ax2.plot(eval_df["K"], eval_df["Silhouette"], marker="s", color="#059669", linewidth=2)
        ax2.set_title("Silhouette Analysis (Separation Score)", fontweight="bold")
        ax2.set_xlabel("Cluster Count (K)")
        ax2.set_ylabel("Silhouette Coefficient")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        return fig

    def plot_pca_clusters(self, df: pd.DataFrame, pca_coords: np.ndarray, cluster_col: str = "Cluster", save_path: str = None):
        """Plots 2D Principal Component projections with labeled cluster boundaries."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        plot_df = df.copy()
        plot_df["PCA1"] = pca_coords[:, 0]
        plot_df["PCA2"] = pca_coords[:, 1]
        
        sns.scatterplot(
            data=plot_df,
            x="PCA1",
            y="PCA2",
            hue=cluster_col,
            palette="Set2",
            s=60,
            alpha=0.85,
            ax=ax
        )
        
        ax.set_title("Customer Segmentation in 2D Principal Component Space", fontweight="bold")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        return fig

    def plot_sentiment_breakdown(self, df: pd.DataFrame, cluster_col: str = "Cluster", sentiment_col: str = "Sentiment_Label", save_path: str = None):
        """Plots stacked sentiment percentage bars for each cluster."""
        crosstab = pd.crosstab(df[cluster_col], df[sentiment_col], normalize="index") * 100
        
        fig, ax = plt.subplots(figsize=(9, 5))
        crosstab.plot(
            kind="bar",
            stacked=True,
            color={"Positive": "#059669", "Neutral": "#94a3b8", "Negative": "#e11d48"},
            ax=ax
        )
        
        ax.set_title("Review Sentiment Distribution Across Customer Clusters (%)", fontweight="bold")
        ax.set_ylabel("Percentage of Cluster Reviews (%)")
        ax.set_xlabel("Customer Cluster")
        plt.xticks(rotation=0)
        plt.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        return fig
