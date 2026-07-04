import pandas as pd
import numpy as np
from pathlib import Path
import os

np.random.seed(42)

def generate_data(n_samples=5000, for_training=True):
    policy_ids = [f"POL-{i:06d}" for i in range(1, n_samples + 1)]
    
    age = np.random.randint(18, 85, n_samples)
    
    # Vehicle age is loosely correlated with age
    vehicle_age = []
    for a in age:
        if a < 25:
            vehicle_age.append(np.random.choice(["< 1 Year", "1-2 Year"], p=[0.7, 0.3]))
        else:
            vehicle_age.append(np.random.choice(["< 1 Year", "1-2 Year", "> 2 Years"], p=[0.2, 0.5, 0.3]))
            
    vehicle_damage = np.random.choice(["Yes", "No"], p=[0.5, 0.5], size=n_samples)
    
    annual_premium = np.random.normal(30000, 10000, n_samples)
    annual_premium = np.clip(annual_premium, 2000, 100000).round(2)
    
    policy_tenure = np.random.randint(1, 120, n_samples)
    
    past_claims_count = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.6, 0.2, 0.1, 0.05, 0.03, 0.02], size=n_samples)
    
    credit_score = np.random.normal(650, 100, n_samples)
    credit_score = np.clip(credit_score, 300, 850).round(0)
    
    df = pd.DataFrame({
        "policy_id": policy_ids,
        "age": age,
        "vehicle_age": vehicle_age,
        "vehicle_damage": vehicle_damage,
        "annual_premium": annual_premium,
        "policy_tenure": policy_tenure,
        "past_claims_count": past_claims_count,
        "credit_score": credit_score
    })
    
    if for_training:
        # Create a synthetic target variable based on some rules
        base_prob = 0.1
        
        prob = base_prob + \
               (df["age"] < 25) * 0.15 + \
               (df["age"] > 65) * 0.10 + \
               (df["vehicle_damage"] == "Yes") * 0.25 + \
               (df["past_claims_count"] * 0.12) - \
               ((df["credit_score"] - 650) / 1000)
               
        prob = np.clip(prob, 0.01, 0.99)
        
        # Add some noise
        prob = prob + np.random.normal(0, 0.05, n_samples)
        prob = np.clip(prob, 0.01, 0.99)
        
        df["claim_status"] = np.random.binomial(1, prob)
        
    return df

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    
    train_df = generate_data(5000, for_training=True)
    train_path = base_dir / "data" / "raw" / "insurance_data.csv"
    train_df.to_csv(train_path, index=False)
    print(f"Saved training data to {train_path}")
    
    sample_df = generate_data(20, for_training=False)
    sample_csv_path = base_dir / "data" / "sample" / "sample_insurance_data.csv"
    sample_excel_path = base_dir / "data" / "sample" / "sample_insurance_data.xlsx"
    
    sample_df.to_csv(sample_csv_path, index=False)
    sample_df.to_excel(sample_excel_path, index=False)
    print(f"Saved sample data to {sample_csv_path} and .xlsx")
