import os
import sys
import pandas as pd
import numpy as np

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
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
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


def save_feature_list(features, filepath, header=""):
    """Save list of features to text file"""
    with open(filepath, 'w') as f:
        if header:
            f.write(f"# {header}\n")
            f.write(f"# Total: {len(features)} features\n\n")
        for feat in sorted(features):
            f.write(f"{feat}\n")


def save_feature_report(df, constant_feats, high_missing_feats, missing_pct, 
                       corr_removed_feats=None, corr_pairs=None):
    """Generate detailed report of feature selection"""
    report_path = os.path.join(OUTPUT_DIR, 'feature_selection_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("FEATURE SELECTION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Original dataset shape: {df.shape}\n")
        f.write(f"Total features: {len(df.columns)}\n")
        f.write(f"Target variable: isFraud\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("CONSTANT FEATURES (removed)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Count: {len(constant_feats)}\n\n")
        for feat in sorted(constant_feats):
            unique_val = df[feat].dropna().unique()
            val_str = str(unique_val[0]) if len(unique_val) > 0 else "All NaN"
            f.write(f"  {feat}: {val_str}\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write(f"HIGH MISSING VALUE FEATURES (>90% missing, removed)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Count: {len(high_missing_feats)}\n\n")
        for feat in sorted(high_missing_feats):
            pct = missing_pct[feat] * 100
            f.write(f"  {feat}: {pct:.2f}% missing\n")
        
        # Add correlation analysis section if provided
        if corr_removed_feats is not None and corr_pairs is not None:
            f.write("\n" + "-" * 80 + "\n")
            f.write(f"HIGHLY CORRELATED FEATURES (>0.95 correlation, removed)\n")
            f.write("-" * 80 + "\n")
            f.write(f"Count: {len(corr_removed_feats)}\n")
            f.write(f"Total correlated pairs found: {len(corr_pairs)}\n\n")
            
            if len(corr_pairs) > 0:
                f.write("Sample correlated pairs (showing up to 20):\n")
                for feat1, feat2, corr_val in sorted(corr_pairs, key=lambda x: x[2], reverse=True)[:20]:
                    dropped = feat1 if feat1 in corr_removed_feats else feat2
                    kept = feat2 if feat1 in corr_removed_feats else feat1
                    f.write(f"  {feat1} <-> {feat2}: {corr_val:.3f} -> Dropped: {dropped}, Kept: {kept}\n")
                
                if len(corr_pairs) > 20:
                    f.write(f"  ... and {len(corr_pairs) - 20} more pairs\n")
        
        # Features removed (union of all)
        removed_features = set(constant_feats) | set(high_missing_feats)
        if corr_removed_feats:
            removed_features |= set(corr_removed_feats)
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Features removed (constant): {len(constant_feats)}\n")
        f.write(f"Features removed (>90% missing): {len(high_missing_feats)}\n")
        if corr_removed_feats is not None:
            f.write(f"Features removed (high correlation): {len(corr_removed_feats)}\n")
        f.write(f"Total features removed: {len(removed_features)}\n")
        f.write(f"Features remaining: {len(df.columns) - len(removed_features) - 1}\n")  # -1 for target
        f.write(f"\nRemaining features saved to: selected_features.txt\n")
    
    print(f"\nDetailed report saved to: {report_path}")


if __name__ == '__main__':
    print("Loading IEEE-CIS fraud detection dataset...")
    df = load_ieee_cis()
    
    print(f"\nOriginal dataset: {df.shape}")
    print(f"Total features: {len(df.columns) - 1} (excluding target)")
    
    # Step 1: Identify constant features
    print("\n" + "=" * 80)
    print("STEP 1: Identifying constant features...")
    print("=" * 80)
    constant_features = identify_constant_features(df)
    print(f"Found {len(constant_features)} constant features")
    if constant_features:
        print("Examples:", constant_features[:5])
    
    # Step 2: Identify high missing value features
    print("\n" + "=" * 80)
    print("STEP 2: Identifying features with >90% missing values...")
    print("=" * 80)
    high_missing_features, missing_pct = identify_high_missing_features(df, threshold=0.90)
    print(f"Found {len(high_missing_features)} features with >90% missing values")
    if high_missing_features:
        print("Examples:", high_missing_features[:5])
    
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
    
    # Combine all removed features
    removed_features = initial_removed | set(corr_removed_features)
    print(f"\n{'=' * 80}")
    print(f"TOTAL FEATURES TO REMOVE: {len(removed_features)}")
    print(f"{'=' * 80}")
    
    # Get remaining features (excluding target)
    all_features = [col for col in df.columns if col != 'isFraud']
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
    selected_features = [f for f in selected_features if f not in low_var_features]
    
    # Step 5: XGBoost feature importance (optional - uncomment to enable)
    print("\n" + "=" * 80)
    print("STEP 5: Selecting top features by XGBoost importance...")
    print("=" * 80)
    top_features, importance_df = select_by_xgboost_importance(
        df, selected_features, target='isFraud', top_n=150, sample_size=100000
    )
    print(f"Selected top {len(top_features)} features by importance")
    importance_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance_scores.csv'), index=False)
    print(f"✓ Importance scores saved")
    selected_features = top_features
    
    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS...")
    print("=" * 80)
    
    save_feature_list(
        constant_features,
        os.path.join(OUTPUT_DIR, 'removed_constant_features.txt'),
        "Features removed: Constant values"
    )
    print(f"✓ Constant features saved to: removed_constant_features.txt")
    
    save_feature_list(
        high_missing_features,
        os.path.join(OUTPUT_DIR, 'removed_high_missing_features.txt'),
        "Features removed: >90% missing values"
    )
    print(f"✓ High missing features saved to: removed_high_missing_features.txt")
    
    save_feature_list(
        corr_removed_features,
        os.path.join(OUTPUT_DIR, 'removed_correlated_features.txt'),
        "Features removed: High correlation (>0.95)"
    )
    print(f"✓ Correlated features saved to: removed_correlated_features.txt")
    
    save_feature_list(
        sorted(removed_features),
        os.path.join(OUTPUT_DIR, 'all_removed_features.txt'),
        "All features removed (constant OR >90% missing OR highly correlated)"
    )
    print(f"✓ All removed features saved to: all_removed_features.txt")
    
    save_feature_list(
        selected_features,
        os.path.join(OUTPUT_DIR, 'selected_features.txt'),
        "Selected features for training (after removing constant, high-missing, and correlated features)"
    )
    print(f"✓ Selected features saved to: selected_features.txt")
    
    # Generate detailed report
    save_feature_report(df, constant_features, high_missing_features, missing_pct,
                       corr_removed_features, corr_pairs)
    
    print("\n" + "=" * 80)
    print("FEATURE SELECTION COMPLETE")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"  1. Review the report: outputs/feature_selection/feature_selection_report.txt")
    print(f"  2. Use selected features: outputs/feature_selection/selected_features.txt")
    print(f"  3. Update train.py to load selected features")
