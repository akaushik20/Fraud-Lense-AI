import os
import sys
import pandas as pd
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.helper import load_ieee_cis

OUTPUT_DIR = 'outputs/feature_selection'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def identify_constant_features(df):
    """Identify features with only one unique value (constant across all rows)"""
    constant_features = []
    for col in df.columns:
        if col == 'isFraud':  # Skip target variable
            continue
        if df[col].nunique(dropna=False) == 1:
            constant_features.append(col)
    return constant_features


def identify_high_missing_features(df, threshold=0.90):
    """Identify features with missing values above threshold"""
    missing_pct = df.isnull().mean()
    high_missing = missing_pct[missing_pct > threshold].index.tolist()
    
    # Don't remove target variable
    if 'isFraud' in high_missing:
        high_missing.remove('isFraud')
    
    return high_missing, missing_pct


def remove_correlated_features(df, target='isFraud', threshold=0.95):
    """
    Remove highly correlated features (correlation > threshold).
    For each correlated pair, keep the feature with higher correlation to target.
    
    Args:
        df: DataFrame with features and target
        target: Target variable name
        threshold: Correlation threshold (default 0.95)
    
    Returns:
        to_drop: List of features to remove
        corr_pairs: List of tuples (feat1, feat2, correlation) for reporting
    """
    print(f"  Computing correlation matrix for {len(df.columns)-1} features...")
    
    # Select only numeric features (exclude target)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)
    
    # Compute correlation matrix
    corr_matrix = df[numeric_cols].corr().abs()
    
    # Get upper triangle of correlation matrix (avoid duplicates)
    # Create boolean mask: True for upper triangle, False elsewhere (k=1 excludes diagonal)
    mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    # Apply mask to keep only upper triangle values, rest become NaN
    upper_tri = corr_matrix.where(mask)
    
    # Find pairs with correlation > threshold
    corr_pairs = []
    for col in upper_tri.columns:
        high_corr = upper_tri[col][upper_tri[col] > threshold]
        for other_col in high_corr.index:
            corr_pairs.append((col, other_col, upper_tri[col][other_col]))
    
    print(f"  Found {len(corr_pairs)} feature pairs with correlation > {threshold}")
    
    if len(corr_pairs) == 0:
        return [], []
    
    # For each pair, decide which feature to drop
    # Keep the one with higher correlation to target
    to_drop = set()
    target_corr = df[numeric_cols + [target]].corr()[target].abs()
    
    for feat1, feat2, corr_val in corr_pairs:
        # Skip if already marked for removal
        if feat1 in to_drop or feat2 in to_drop:
            continue
        
        # Compare correlation with target
        corr1 = target_corr.get(feat1, 0)
        corr2 = target_corr.get(feat2, 0)
        
        # Drop the feature with lower target correlation
        if corr1 >= corr2:
            to_drop.add(feat2)
        else:
            to_drop.add(feat1)
    
    return list(to_drop), corr_pairs


def remove_low_variance_features(df, threshold=0.01):
    """
    Remove features with variance below threshold.
    
    Args:
        df: DataFrame with numeric features
        threshold: Minimum variance required (default 0.01)
    
    Returns:
        to_drop: List of low-variance features to remove
    """
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    variances = df[numeric_cols].var()
    low_var = variances[variances < threshold].index.tolist()
    
    return low_var


def select_by_xgboost_importance(df, features, target='isFraud', top_n=150, sample_size=100000):
    """
    Select top N features using XGBoost feature importance.
    Uses minimal preprocessing (categorical encoding only).
    
    Args:
        df: Full DataFrame with features and target
        features: List of feature names to consider
        target: Target variable name
        top_n: Number of top features to keep
        sample_size: Number of rows to sample for speed (None for all)
    
    Returns:
        selected_features: List of top N features by importance
        importance_df: DataFrame with all features and their importance scores
    """
    from xgboost import XGBClassifier
    from src.helper import encode_categoricals
    
    print(f"  Preparing data for XGBoost...")
    
    # Sample for speed if needed
    if sample_size and len(df) > sample_size:
        # Stratified sampling: sample from each class proportionally
        fraud_rate = df[target].mean()
        n_fraud = int(sample_size * fraud_rate)
        n_normal = sample_size - n_fraud
        
        fraud_df = df[df[target] == 1].sample(n=min(n_fraud, (df[target] == 1).sum()), random_state=42)
        normal_df = df[df[target] == 0].sample(n=n_normal, random_state=42)
        df_sample = pd.concat([fraud_df, normal_df], axis=0).sample(frac=1, random_state=42)
        print(f"  Using stratified sample of {len(df_sample):,} rows (fraud rate: {df_sample[target].mean():.4f})")
    else:
        df_sample = df
    
    # Select features and encode categoricals
    X = df_sample[features].copy()
    y = df_sample[target]
    
    X_encoded, _ = encode_categoricals(X)
    
    # Calculate scale_pos_weight for imbalance
    fraud_rate = y.mean()
    scale_weight = (1 - fraud_rate) / fraud_rate
    print(f"  Fraud rate: {fraud_rate:.2%}, scale_pos_weight: {scale_weight:.1f}")
    
    # Train XGBoost
    print(f"  Training XGBoost on {len(features)} features...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_weight,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_encoded, y)
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    selected_features = importance_df.head(top_n)['feature'].tolist()
    
    return selected_features, importance_df


def save_feature_importance_csv(feature_records, filepath):
    """Save a complete feature report as a single CSV file."""
    rows = []
    for feature, info in feature_records.items():
        rows.append({
            'feature': feature,
            'status': info.get('status', 'selected'),
            'removal_reason': info.get('removal_reason', ''),
            'missing_pct': round(info.get('missing_pct', 0.0) * 100, 2),
            'importance_score': info.get('importance_score', ''),
        })
    df_out = pd.DataFrame(rows).sort_values(
        ['status', 'importance_score'], ascending=[True, False]
    )
    df_out.to_csv(filepath, index=False)


def save_features_yaml(df, features, filepath):
    """Save features in YAML format with numeric/categorical split"""
    # Separate numeric and categorical features
    numeric_features = [f for f in features if df[f].dtype != 'object']
    categorical_features = [f for f in features if df[f].dtype == 'object']
    
    # Create configuration dictionary
    config = {
        'total_features': len(features),
        'numeric_count': len(numeric_features),
        'categorical_count': len(categorical_features),
        'numeric': sorted(numeric_features),
        'categorical': sorted(categorical_features)
    }
    
    # Save as YAML
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


if __name__ == '__main__':
    print("Loading IEEE-CIS fraud detection dataset...")
    df = load_ieee_cis()
    
    print(f"\nOriginal dataset: {df.shape}")
    print(f"Total features: {len(df.columns) - 1} (excluding target)")

    # Initialize feature records: one entry per non-target feature
    all_features = [col for col in df.columns if col != 'isFraud']
    feature_records = {f: {'status': 'selected', 'removal_reason': '', 'missing_pct': 0.0, 'importance_score': ''} for f in all_features}

    # Step 1: Identify constant features
    print("\n" + "=" * 80)
    print("STEP 1: Identifying constant features...")
    print("=" * 80)
    constant_features = identify_constant_features(df)
    print(f"Found {len(constant_features)} constant features")
    if constant_features:
        print("Examples:", constant_features[:5])
    for f in constant_features:
        feature_records[f]['status'] = 'removed'
        feature_records[f]['removal_reason'] = 'constant'

    # Step 2: Identify high missing value features
    print("\n" + "=" * 80)
    print("STEP 2: Identifying features with >90% missing values...")
    print("=" * 80)
    high_missing_features, missing_pct = identify_high_missing_features(df, threshold=0.90)
    print(f"Found {len(high_missing_features)} features with >90% missing values")
    if high_missing_features:
        print("Examples:", high_missing_features[:5])
    for f in all_features:
        feature_records[f]['missing_pct'] = missing_pct.get(f, 0.0)
    for f in high_missing_features:
        feature_records[f]['status'] = 'removed'
        feature_records[f]['removal_reason'] = 'high_missing'

    # Step 3: Remove highly correlated features
    print("\n" + "=" * 80)
    print("STEP 3: Removing highly correlated features (>0.95)...")
    print("=" * 80)
    
    # First, remove already identified features before correlation analysis
    initial_removed = set(constant_features) | set(high_missing_features)
    df_for_corr = df.drop(columns=list(initial_removed), errors='ignore')
    
    corr_removed_features, corr_pairs = remove_correlated_features(df_for_corr, threshold=0.95)
    print(f"Found {len(corr_removed_features)} features to remove due to high correlation")
    if corr_removed_features:
        print("Examples:", corr_removed_features[:5])
    for f in corr_removed_features:
        feature_records[f]['status'] = 'removed'
        feature_records[f]['removal_reason'] = 'high_correlation'

    # Combine all removed features
    removed_features = initial_removed | set(corr_removed_features)
    print(f"\n{'=' * 80}")
    print(f"TOTAL FEATURES TO REMOVE: {len(removed_features)}")
    print(f"{'=' * 80}")
    
    # Get remaining features (excluding target)
    selected_features = [feat for feat in all_features if feat not in removed_features]
    
    print(f"Features remaining: {len(selected_features)}")
    print(f"Reduction: {len(removed_features) / len(all_features) * 100:.1f}%")
    
    # Step 4: Remove low variance features (optional - uncomment to enable)
    print("\n" + "=" * 80)
    print("STEP 4: Removing low variance features...")
    print("=" * 80)
    df_remaining = df[selected_features + ['isFraud']]
    low_var_features = remove_low_variance_features(df_remaining, threshold=0.01)
    print(f"Found {len(low_var_features)} low-variance features")
    for f in low_var_features:
        feature_records[f]['status'] = 'removed'
        feature_records[f]['removal_reason'] = 'low_variance'
    selected_features = [f for f in selected_features if f not in low_var_features]

    # Step 5: XGBoost feature importance (optional - uncomment to enable)
    print("\n" + "=" * 80)
    print("STEP 5: Selecting top features by XGBoost importance...")
    print("=" * 80)
    top_features, importance_df = select_by_xgboost_importance(
        df, selected_features, target='isFraud', top_n=150, sample_size=100000
    )
    print(f"Selected top {len(top_features)} features by importance")
    # Merge importance scores into records; features not in top_n are low_importance
    importance_map = dict(zip(importance_df['feature'], importance_df['importance']))
    for f in selected_features:
        score = importance_map.get(f, 0.0)
        feature_records[f]['importance_score'] = score
        if f not in top_features:
            feature_records[f]['status'] = 'removed'
            feature_records[f]['removal_reason'] = 'low_importance'
    selected_features = top_features

    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS...")
    print("=" * 80)

    save_feature_importance_csv(
        feature_records,
        os.path.join(OUTPUT_DIR, 'feature_importance_scores.csv')
    )
    print(f"✓ Feature report saved to: feature_importance_scores.csv")

    save_features_yaml(
        df,
        selected_features,
        os.path.join(OUTPUT_DIR, 'selected_features.yaml')
    )
    print(f"✓ Selected features saved to: selected_features.yaml")

    print("\n" + "=" * 80)
    print("FEATURE SELECTION COMPLETE")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"  1. Review: outputs/feature_selection/feature_importance_scores.csv")
    print(f"  2. Use selected features: outputs/feature_selection/selected_features.yaml")
    print(f"  3. Run model training: python model/train.py")
