import numpy as np
import pandas as pd
from src.preprocessing import CustomerDataPreprocessor

def test_preprocessing_out_of_sample_and_missing_values():
    # Setup data
    np.random.seed(42)
    df_train = pd.DataFrame({
        "Recency": [10, 20, 30, 40, 50],
        "Frequency": [1, 2, 1, 2, 1],
        "Monetary": [100.0, 200.0, 150.0, 300.0, 1000.0]
    })
    features = ["Recency", "Frequency", "Monetary"]
    
    preprocessor = CustomerDataPreprocessor(skew_threshold=0.5)
    X_train_scaled, df_train_transformed = preprocessor.fit_transform(df_train, features)
    
    # Test Medians
    assert hasattr(preprocessor, 'medians')
    assert preprocessor.medians["Recency"] == 30.0
    
    # Test Out of sample transform with missing values
    df_test = pd.DataFrame({
        "Recency": [np.nan],
        "Frequency": [np.nan],
        "Monetary": [np.nan]
    })
    X_test_scaled = preprocessor.transform(df_test)
    
    # Ensure no NaN in output
    assert not np.isnan(X_test_scaled).any()
    
    # Test that applying transform on train data matches fit_transform output
    X_train_scaled_2 = preprocessor.transform(df_train)
    np.testing.assert_array_almost_equal(X_train_scaled, X_train_scaled_2)
