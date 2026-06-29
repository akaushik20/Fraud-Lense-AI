import os
import sys
import pickle
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.helper import load_ieee_cis, encode_categoricals

OUTPUT_DIR = 'outputs/models'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_selected_features(filepath='outputs/feature_selection/selected_features.txt'):
    """Load feature list from file"""
    features = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                features.append(line)
    return features


def prepare_data(df, features, target='isFraud', test_size=0.2):
    """Prepare train/test split with minimal preprocessing"""
    # Sort by time for time-based split (TransactionDT is time delta)
    df = df.sort_values('TransactionDT').reset_index(drop=True)
    
    # Split point (last 20% for test)
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # Separate features and target
    X_train = train_df[features].copy()
    y_train = train_df[target]
    X_test = test_df[features].copy()
    y_test = test_df[target]
    
    # Encode categoricals (fit on train, transform both)
    X_train_encoded, encoders = encode_categoricals(X_train)
    X_test_encoded, _ = encode_categoricals(X_test, encoders)
    
    return X_train_encoded, X_test_encoded, y_train, y_test, encoders


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost with imbalance handling"""
    # Calculate scale_pos_weight for imbalance
    fraud_rate = y_train.mean()
    scale_weight = (1 - fraud_rate) / fraud_rate
    
    print(f"Training set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    print(f"Fraud rate (train): {fraud_rate:.2%}")
    print(f"Scale pos weight: {scale_weight:.1f}")
    
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='auc',
        enable_categorical=False  # Disable categorical splits for SHAP compatibility
    )
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )
    
    return model


def save_artifacts(model, encoders, features, X_train_sample=None):
    """Save model, encoders, feature list, and SHAP background data"""
    # Save model
    model_path = os.path.join(OUTPUT_DIR, 'xgboost_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved: {model_path}")
    
    # Save encoders
    encoder_path = os.path.join(OUTPUT_DIR, 'encoders.pkl')
    with open(encoder_path, 'wb') as f:
        pickle.dump(encoders, f)
    print(f"✓ Encoders saved: {encoder_path}")
    
    # Save feature list
    feature_path = os.path.join(OUTPUT_DIR, 'features.txt')
    with open(feature_path, 'w') as f:
        for feat in features:
            f.write(f"{feat}\n")
    print(f"✓ Features saved: {feature_path}")
    
    # Save SHAP background data (for explanations)
    if X_train_sample is not None:
        background_path = os.path.join(OUTPUT_DIR, 'shap_background.pkl')
        with open(background_path, 'wb') as f:
            pickle.dump(X_train_sample, f)
        print(f"✓ SHAP background saved: {background_path}")


if __name__ == "__main__":
    print("=" * 80)
    print("FRAUD DETECTION MODEL TRAINING")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading data...")
    df = load_ieee_cis()
    
    # Load selected features
    print("\n2. Loading selected features...")
    features = load_selected_features()
    print(f"Using {len(features)} features")
    
    # Prepare data
    print("\n3. Preparing train/test split...")
    X_train, X_test, y_train, y_test, encoders = prepare_data(df, features)
    
    # Train model
    print("\n4. Training XGBoost model...")
    model = train_xgboost(X_train, y_train, X_test, y_test)
    
    # Save artifacts
    print("\n5. Saving model artifacts...")
    # Sample 100 rows from training set for SHAP background
    X_train_sample = X_train.sample(n=min(100, len(X_train)), random_state=42)
    save_artifacts(model, encoders, features, X_train_sample)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)