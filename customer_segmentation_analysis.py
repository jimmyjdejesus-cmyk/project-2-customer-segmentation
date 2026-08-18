# %% [markdown]
# # Data Science Intelligence: Customer RFM Segmentation & NLP Topic Mining
# 
# **Author**: AI & Data Science Engineer  
# **Objective**: 
# 1. Standardize and normalize skewed customer Recency, Frequency, and Monetary (RFM) distributions.
# 2. Benchmark unsupervised K-Means clustering across **WCSS (Elbow)**, **Silhouette Score**, **Davies-Bouldin Index**, and **Calinski-Harabasz Criterion**.
# 3. Project clusters into 2D **Principal Component Analysis (PCA)** space.
# 4. Mine customer reviews using **NLTK VADER Sentiment Analysis** and **TF-IDF N-Gram Topic Extraction**.
# 5. Formulate actionable customer persona profiles and retention strategies.

# %%
# System Imports
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to import path
sys.path.insert(0, os.path.abspath("."))

from src.preprocessing import CustomerDataPreprocessor
from src.clustering import CustomerClusterEngine
from src.nlp import CustomerSentimentEngine
from src.visualization import CustomerSegmentationPlotter

# %% [markdown]
# ## 1. Ingest Customer Transaction & Review Dataset
# We generate a dataset of 600 customer profiles simulating realistic e-commerce purchase patterns and associated product reviews.

# %%
from src.data_generation import generate_synthetic_data

np.random.seed(42)
num_samples = 600

df_customers = generate_synthetic_data(num_samples=num_samples, seed=42)

print("Dataset Preview (First 5 Rows):")
print(df_customers.head())

# %% [markdown]
# ## 2. Feature Preprocessing & Skewness Correction
# E-commerce monetary spend and recency distributions typically exhibit heavy right-skewness. We calculate Fisher-Pearson skewness and apply $\log(1 + x)$ transformations prior to zero-mean unit-variance standard scaling.

# %%
features = ["Recency", "Frequency", "Monetary"]
preprocessor = CustomerDataPreprocessor(skew_threshold=0.75)

skew_initial = preprocessor.check_skewness(df_customers, features)
print(f"Initial Skewness Coefficients: {skew_initial}")

X_scaled, df_transformed = preprocessor.fit_transform(df_customers, features)
print(f"Transformed Features applied log1p: {preprocessor.transformed_features}")
print(f"Standardized Feature Matrix Shape: {X_scaled.shape}")

# %% [markdown]
# ## 3. Comprehensive Clustering Optimization & Validation
# We evaluate cluster counts $K \in [2, 8]$ across four complementary mathematical separation criteria.

# %%
cluster_engine = CustomerClusterEngine(random_state=42)
eval_df = cluster_engine.evaluate_clusters(X_scaled, k_range=range(2, 9))

print("Clustering Metrics Across K:")
print(eval_df.to_string(index=False))

plotter = CustomerSegmentationPlotter()
fig_eval = plotter.plot_cluster_evaluation(eval_df)
plt.show()

# %% [markdown]
# ## 4. Fit Optimal Model & 2D PCA Projections
# We fit $K = 3$ clusters and compute Principal Component Analysis (PCA) projections to visually verify cluster separation boundaries.

# %%
optimal_k = 3
cluster_labels = cluster_engine.fit_predict(X_scaled, k=optimal_k)
df_customers["Cluster"] = cluster_labels

pca_coords = cluster_engine.get_pca_projections(X_scaled)
print(f"Cumulative Explained Variance (2D PCA): {cluster_engine.explained_variance_ratio:.2f}%")

fig_pca = plotter.plot_pca_clusters(df_customers, pca_coords, cluster_col="Cluster")
plt.show()

# %% [markdown]
# ## 5. Natural Language Processing (NLP) Sentiment Analysis
# We analyze review polarity using the NLTK VADER lexicon to determine emotional valence and calculate semantic alignment with RFM clusters.

# %%
sentiment_engine = CustomerSentimentEngine()
df_analyzed = sentiment_engine.analyze_sentiment(df_customers, text_col="ReviewText")

fig_sentiment = plotter.plot_sentiment_breakdown(df_analyzed, cluster_col="Cluster", sentiment_col="Sentiment_Label")
plt.show()

# %% [markdown]
# ## 6. Qualitative TF-IDF Topic & Keyphrase Extraction
# Extract dominant N-grams per cluster to uncover root causes of satisfaction vs. churn friction.

# %%
cluster_topics = sentiment_engine.extract_cluster_topics(df_analyzed, cluster_col="Cluster", text_col="Cleaned_ReviewText", top_n=4)

profiles = cluster_engine.generate_cluster_profiles(df_analyzed)

print("=" * 80)
print("FINAL PERSONA PROFILES & QUALITATIVE THEMES")
print("=" * 80)
for _, row in profiles.iterrows():
    c_id = row["Cluster"]
    topics_str = ", ".join(cluster_topics.get(c_id, []))
    print(f"\n[CLUSTER {c_id}] -> {row['Persona']}")
    print(f"  • Customer Share: {row['CustomerSharePct']}% ({row['CustomerCount']} users)")
    print(f"  • Revenue Share:  {row['RevenueSharePct']}% (${row['TotalMonetary']:,.2f})")
    print(f"  • Avg RFM:        Recency={row['AvgRecency']:.1f}d | Freq={row['AvgFrequency']:.1f} orders | Spend=${row['AvgMonetary']:,.2f}")
    print(f"  • Driver Themes:  {topics_str}")
print("=" * 80)
