import numpy as np
import pandas as pd

def generate_synthetic_data(num_samples: int = 600, seed: int = 42) -> pd.DataFrame:
    """Generates synthetic RFM metrics with correlated textual customer reviews."""
    np.random.seed(seed)
    
    recency = np.random.exponential(scale=55, size=num_samples)
    frequency = np.random.poisson(lam=4, size=num_samples) + 1
    monetary = frequency * np.random.normal(loc=50, scale=18, size=num_samples)
    monetary = np.clip(monetary, 8.0, None)
    
    positive_reviews = [
        "Incredible product quality! Delivered in 2 days and exceeded all my expectations.",
        "Flawless customer service. They resolved my return in under 3 minutes.",
        "Very high quality craftsmanship. Will definitely purchase again for the team.",
        "Worth every penny. Outstanding build and intuitive setup process."
    ]
    neutral_reviews = [
        "Acceptable performance. It functions as described, but packaging was slightly dented.",
        "Standard delivery time. Product is fine for routine everyday use.",
        "A bit pricey for what it delivers, but overall decent experience.",
        "Does the job adequately. No major complaints, but nothing exceptional."
    ]
    negative_reviews = [
        "Completely defective. The charging unit broke after 3 days of normal use.",
        "Unacceptable delivery delay. Took 3 weeks to arrive and support was unreachable.",
        "Extremely disappointed with build quality. Cheap plastic that feels fragile.",
        "Do not recommend. Horrible return policy and unhelpful customer support."
    ]
    
    reviews = []
    for r, f, m in zip(recency, frequency, monetary):
        score = (f * 2.5 + m / 25.0) - r
        if score > 40:
            reviews.append(np.random.choice(positive_reviews))
        elif score < -25:
            reviews.append(np.random.choice(negative_reviews))
        else:
            reviews.append(np.random.choice(neutral_reviews))
            
    df = pd.DataFrame({
        "CustomerID": [f"CUST-{i:04d}" for i in range(1, num_samples + 1)],
        "Recency": np.round(recency, 1),
        "Frequency": frequency,
        "Monetary": np.round(monetary, 2),
        "ReviewText": reviews
    })
    return df
