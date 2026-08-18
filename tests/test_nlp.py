"""
================================================================================
Unit Tests for NLP Sentiment & Topic Extraction
================================================================================
"""

import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.nlp import CustomerSentimentEngine


@pytest.fixture
def sample_reviews_df():
    """Generates synthetic review texts for NLP sentiment testing."""
    return pd.DataFrame({
        "CustomerID": ["CUST-001", "CUST-002", "CUST-003"],
        "Cluster": [0, 1, 2],
        "ReviewText": [
            "Outstanding quality, excellent customer support, and fast delivery!",
            "Average product, packaging was slightly broken but it works okay.",
            "Horrible experience, completely defective charging unit and delayed delivery."
        ]
    })


def test_sentiment_classification(sample_reviews_df):
    """Verifies VADER polarity classification into positive, neutral, and negative labels."""
    engine = CustomerSentimentEngine()
    df_res = engine.analyze_sentiment(sample_reviews_df, text_col="ReviewText")
    
    assert "Sentiment_Compound" in df_res.columns
    assert "Sentiment_Label" in df_res.columns
    
    # Check expected polarity mapping
    assert df_res.loc[0, "Sentiment_Label"] == "Positive"
    assert df_res.loc[2, "Sentiment_Label"] == "Negative"
    assert df_res.loc[0, "Sentiment_Compound"] > df_res.loc[2, "Sentiment_Compound"]


def test_topic_ngram_extraction(sample_reviews_df):
    """Verifies TF-IDF N-gram topic keyphrase extraction per cluster."""
    engine = CustomerSentimentEngine()
    topics = engine.extract_cluster_topics(sample_reviews_df, cluster_col="Cluster", text_col="ReviewText", top_n=2)
    
    assert len(topics) == 3
    for c_id, term_list in topics.items():
        assert isinstance(term_list, list)
        assert len(term_list) > 0
