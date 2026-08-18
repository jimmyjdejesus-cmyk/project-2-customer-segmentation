"""
================================================================================
Feature Preprocessing & Skewness Normalization Pipeline
================================================================================
Author: AI & Data Science Engineer
Purpose:
  - Detects distribution skewness in RFM features
  - Applies logarithmic transformations (log1p) to normalize right-skewed data
  - Standardizes variables via StandardScaler for Euclidean distance models
================================================================================
"""

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler


class CustomerDataPreprocessor:
    def __init__(self, skew_threshold: float = 0.75):
        """
        Args:
            skew_threshold: Minimum absolute skewness coefficient to trigger log transformation.
        """
        self.skew_threshold = skew_threshold
        self.transformed_features = []
        self.scaler = StandardScaler()
        self.feature_names = []

    def check_skewness(self, df: pd.DataFrame, features: list) -> dict:
        """
        Calculates the Fisher-Pearson skewness score for numeric features.
        """
        skew_dict = {}
        for feat in features:
            s_val = float(skew(df[feat].dropna()))
            skew_dict[feat] = round(s_val, 3)
        return skew_dict

    def fit_transform(self, df: pd.DataFrame, features: list) -> tuple:
        """
        Applies log1p transformation to skewed features and fits StandardScaler.
        
        Returns:
            scaled_array: np.ndarray of normalized features.
            transformed_df: pd.DataFrame containing transformed values.
        """
        self.feature_names = features
        df_clean = df[features].copy()
        self.medians = df_clean.median()
        df_clean = df_clean.fillna(self.medians)
        
        # 1. Evaluate skewness and apply log1p where appropriate
        skew_scores = self.check_skewness(df_clean, features)
        self.transformed_features = []
        
        for feat, score in skew_scores.items():
            if score >= self.skew_threshold and (df_clean[feat] >= 0).all():
                df_clean[feat] = np.log1p(df_clean[feat])
                self.transformed_features.append(feat)
                
        # 2. Standardize to Zero Mean and Unit Variance
        scaled_array = self.scaler.fit_transform(df_clean)
        
        return scaled_array, df_clean

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms new out-of-sample customer data using fitted transformations.
        """
        df_clean = df[self.feature_names].copy()
        if hasattr(self, 'medians'):
            df_clean = df_clean.fillna(self.medians)
        for feat in self.transformed_features:
            df_clean[feat] = np.log1p(df_clean[feat])
        return self.scaler.transform(df_clean)
