"""
================================================================================
NLP Sentiment Analysis & Qualitative N-Gram Topic Extraction
================================================================================
Author: AI & Data Science Engineer
Purpose:
  - Lexicon-based Sentiment Scoring (NLTK VADER)
  - Bi-gram & Tri-gram Keyphrase Topic Extraction per Customer Cluster
  - Semantic correlation between RFM quantitative clusters and review texts
================================================================================
"""

import re
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class CustomerSentimentEngine:
    def __init__(self):
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        self.sia = SentimentIntensityAnalyzer()

    def clean_text(self, text: str) -> str:
        """Standardizes text casing and removes anomalous punctuation."""
        if not isinstance(text, str):
            return ""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        return text

    def analyze_sentiment(self, df: pd.DataFrame, text_col: str = "ReviewText") -> pd.DataFrame:
        """
        Calculates compound, positive, neutral, and negative VADER sentiment scores.
        Categorizes sentiment into Positive, Neutral, or Negative labels.
        """
        df_out = df.copy()
        
        df_out["Cleaned_ReviewText"] = df_out[text_col].apply(self.clean_text)
        
        # Calculate Polarity Scores
        scores = df_out[text_col].apply(lambda x: self.sia.polarity_scores(str(x)))
        df_out["Sentiment_Compound"] = scores.apply(lambda s: s["compound"])
        df_out["Sentiment_Pos"] = scores.apply(lambda s: s["pos"])
        df_out["Sentiment_Neu"] = scores.apply(lambda s: s["neu"])
        df_out["Sentiment_Neg"] = scores.apply(lambda s: s["neg"])
        
        def assign_label(compound: float) -> str:
            if compound >= 0.05:
                return "Positive"
            elif compound <= -0.05:
                return "Negative"
            else:
                return "Neutral"
                
        df_out["Sentiment_Label"] = df_out["Sentiment_Compound"].apply(assign_label)
        return df_out

    def extract_cluster_topics(self, df: pd.DataFrame, cluster_col: str = "Cluster", text_col: str = "ReviewText", top_n: int = 5) -> dict:
        """
        Extracts top N TF-IDF bi-grams and tri-grams per cluster to uncover
        specific qualitative drivers (e.g. 'broken item', 'fast delivery').
        """
        cluster_topics = {}
        unique_clusters = sorted(df[cluster_col].unique())
        
        for c in unique_clusters:
            cluster_texts = df[df[cluster_col] == c][text_col].dropna().tolist()
            if not cluster_texts:
                cluster_topics[c] = []
                continue
                
            # Fit TF-IDF on cluster texts using 1-to-3 ngrams
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 3),
                max_features=25
            )
            
            try:
                tfidf_matrix = vectorizer.fit_transform(cluster_texts)
                feature_names = vectorizer.get_feature_names_out()
                scores = tfidf_matrix.sum(axis=0).A1
                
                top_indices = scores.argsort()[::-1][:top_n]
                top_terms = [feature_names[i] for i in top_indices]
                cluster_topics[c] = top_terms
            except ValueError:
                # Fallback for small sample clusters
                from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
                words = " ".join(cluster_texts).lower().split()
                filtered = [w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 2]
                common = [w for w, _ in Counter(filtered).most_common(top_n)]
                cluster_topics[c] = common
                
        return cluster_topics
