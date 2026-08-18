import pandas as pd
from src.data_generation import generate_synthetic_data

def test_generate_synthetic_data():
    num_samples = 100
    df = generate_synthetic_data(num_samples=num_samples, seed=42)
    
    # Check shape
    assert df.shape[0] == num_samples
    
    # Check columns
    expected_cols = ["CustomerID", "Recency", "Frequency", "Monetary", "ReviewText"]
    for col in expected_cols:
        assert col in df.columns
        
    # Check constraints
    assert (df["Recency"] >= 0).all()
    assert (df["Frequency"] >= 1).all()
    assert (df["Monetary"] > 0).all()
