"""
Pre-compute SHAP feature importance for landing page chart
Computes mean absolute SHAP values on 200-row training sample
"""
import json
import pickle
import yaml
import pandas as pd
import numpy as np
import shap
from src.helper import load_ieee_cis, encode_categoricals, INTERPRETABLE_FEATURES

# Load model
with open('outputs/models/xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load encoders
with open('outputs/models/encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Load features from YAML
with open('outputs/models/features.yaml', 'r') as f:
    feature_config = yaml.safe_load(f)
    all_features = feature_config['numeric'] + feature_config['categorical']
    categorical_features = feature_config['categorical']

# Load data and sample 200 rows from training set
print("Loading training data...")
df = load_ieee_cis()
df = df.sort_values('TransactionDT').reset_index(drop=True)
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]

# Sample 200 rows for SHAP computation
print("Sampling 200 rows for SHAP computation...")
sample_df = train_df.sample(n=min(200, len(train_df)), random_state=42)
X_sample = sample_df[all_features].copy()

# Encode categorical features
if categorical_features:
    X_sample, _ = encode_categoricals(X_sample, encoders=encoders, columns=categorical_features)

# Create SHAP explainer
print("Creating SHAP explainer...")
explainer = shap.TreeExplainer(model, X_sample)

# Compute SHAP values
print("Computing SHAP values...")
shap_values = explainer.shap_values(X_sample)

# Handle SHAP output format (for binary classification)
if isinstance(shap_values, list):
    shap_values = shap_values[1]  # Positive class (fraud)

# Calculate mean absolute SHAP for each feature
mean_abs_shap = np.abs(shap_values).mean(axis=0)
mean_shap = shap_values.mean(axis=0)  # For direction

# Create DataFrame for ALL features, tagging which ones are interpretable
shap_df = pd.DataFrame({
    'feature': all_features,
    'mean_abs_shap': mean_abs_shap,
    'mean_shap': mean_shap,
    'is_interpretable': [f in INTERPRETABLE_FEATURES for f in all_features]
})
shap_df = shap_df.sort_values('mean_abs_shap', ascending=False)


def to_feature_dicts(df):
    """Convert a SHAP DataFrame slice into JSON-serializable records"""
    records = []
    for _, row in df.iterrows():
        records.append({
            'feature': row['feature'],
            'mean_abs_shap': float(row['mean_abs_shap']),
            'mean_shap': float(row['mean_shap']),
            'direction': 'increases_risk' if row['mean_shap'] > 0 else 'decreases_risk',
            'is_interpretable': bool(row['is_interpretable'])
        })
    return records


# Full SHAP results for every feature (interpretable and non-interpretable)
all_feature_importance = to_feature_dicts(shap_df)

# Subset used for the Gradio chart: interpretable features only, top 15
print(f"\nFiltering to interpretable features (from {len(shap_df)} total)...")
interpretable_shap = shap_df[shap_df['is_interpretable']]
print(f"Found {len(interpretable_shap)} interpretable features")

top_features = interpretable_shap.nlargest(15, 'mean_abs_shap')
feature_importance = to_feature_dicts(top_features)

# Save to JSON
output = {
    'sample_size': len(X_sample),
    'total_features': len(all_features),
    'interpretable_features': len(interpretable_shap),
    'top_features_count': len(feature_importance),
    'features': feature_importance,       # interpretable-only, top 15 (used by Gradio chart)
    'all_features': all_feature_importance  # every feature, tagged with is_interpretable
}

output_path = 'outputs/models/shap_feature_importance.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✓ SHAP feature importance saved to: {output_path}")
print(f"\nTop 5 features by importance:")
for i, feat in enumerate(feature_importance[:5], 1):
    direction = "↑" if feat['direction'] == 'increases_risk' else "↓"
    print(f"  {i}. {feat['feature']:20s} {direction} {feat['mean_abs_shap']:.4f}")
