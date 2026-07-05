"""
Pre-compute default feature values for the single-transaction explainer form.
Users only fill in the interpretable features; every other feature falls
back to its median (numeric) or mode (categorical) from the training data.
"""
import json
import yaml
import pandas as pd
from src.helper import load_ieee_cis, INTERPRETABLE_FEATURES

# Load features from YAML
with open('outputs/models/features.yaml', 'r') as f:
    feature_config = yaml.safe_load(f)
    numeric_features = feature_config['numeric']
    categorical_features = feature_config['categorical']
    all_features = numeric_features + categorical_features

# Load data and use the same 80% train split as training/SHAP scripts
print("Loading training data...")
df = load_ieee_cis()
df = df.sort_values('TransactionDT').reset_index(drop=True)
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]


def default_for(feature):
    """Median for numeric features, mode for categorical features"""
    series = train_df[feature]
    if feature in categorical_features:
        mode = series.mode(dropna=True)
        return str(mode.iloc[0]) if not mode.empty else 'unknown'
    median = series.median()
    return float(median) if pd.notna(median) else 0.0


print("Computing default values for all features...")
feature_defaults = {feature: default_for(feature) for feature in all_features}

output_path = 'outputs/models/feature_defaults.json'
with open(output_path, 'w') as f:
    json.dump(feature_defaults, f, indent=2)

print(f"\nFeature defaults saved to: {output_path}")
print(f"  {len(feature_defaults)} features total "
      f"({len(all_features) - len(INTERPRETABLE_FEATURES)} defaulted, "
      f"{len(INTERPRETABLE_FEATURES)} overridden by the form)")
