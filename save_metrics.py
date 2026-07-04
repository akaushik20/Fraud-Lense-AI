"""
Extract and save model metrics to JSON for Gradio app
"""
import json
import pickle
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.helper import load_ieee_cis, encode_categoricals

# Load model
with open('outputs/models/xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load encoders
with open('outputs/models/encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Load features
with open('outputs/models/features.txt', 'r') as f:
    features = [line.strip() for line in f if line.strip()]

# Load data
df = load_ieee_cis()

# Prepare test set (last 20% based on time)
df = df.sort_values('TransactionDT').reset_index(drop=True)
split_idx = int(len(df) * 0.8)
test_df = df.iloc[split_idx:]

X_test = test_df[features].copy()
y_test = test_df['isFraud']

# Encode categoricals
X_test_encoded, _ = encode_categoricals(X_test, encoders)

# Get predictions
y_pred_proba = model.predict_proba(X_test_encoded)[:, 1]

# Calculate AUC
auc_score = roc_auc_score(y_test, y_pred_proba)

# Calculate fraud rate from full training data
fraud_rate = df['isFraud'].mean() * 100  # Convert to percentage

# Prepare metrics dictionary
metrics = {
    "auc_score": round(auc_score, 4),
    "fraud_rate_percent": round(fraud_rate, 2),
    "total_transactions": len(df),
    "original_features": 434,
    "selected_features": len(features)
}

# Save to JSON
with open('outputs/models/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("Metrics saved to outputs/models/metrics.json")
print(json.dumps(metrics, indent=2))
