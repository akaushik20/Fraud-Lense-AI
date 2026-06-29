import os
import pickle
import pandas as pd
import numpy as np
import shap
from src.helper import INTERPRETABLE_FEATURES, encode_categoricals


def load_model_artifacts(model_dir='outputs/models'):
    """Load trained model, encoders, and feature list"""
    # Load model
    model_path = os.path.join(model_dir, 'xgboost_model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load encoders
    encoder_path = os.path.join(model_dir, 'encoders.pkl')
    with open(encoder_path, 'rb') as f:
        encoders = pickle.load(f)
    
    # Load feature list
    feature_path = os.path.join(model_dir, 'features.txt')
    features = []
    with open(feature_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                features.append(line)
    
    return model, encoders, features


def create_shap_explainer(model, X_background):
    """
    Initialize SHAP TreeExplainer with background data
    
    Args:
        model: Trained XGBoost model
        X_background: Sample of training data (e.g., 100 rows)
    
    Returns:
        explainer: SHAP TreeExplainer object
    """
    # Use tree_path_dependent for XGBoost compatibility (handles categorical splits)
    explainer = shap.TreeExplainer(model, X_background, feature_perturbation='tree_path_dependent')
    return explainer


def get_prediction_explanation(model, explainer, encoders, transaction_df, features):
    """
    Get top 5 interpretable feature drivers for a single prediction
    
    Args:
        model: Trained XGBoost model
        explainer: SHAP explainer
        encoders: Dictionary of LabelEncoders
        transaction_df: DataFrame with single transaction (raw features)
        features: List of feature names used in training
    
    Returns:
        dict with prediction, is_fraud flag, and top 5 drivers
    """
    # Select only features used in training
    X = transaction_df[features].copy()
    
    # Encode categoricals using trained encoders
    X_encoded, _ = encode_categoricals(X, encoders)
    
    # Get prediction probability
    pred_proba = model.predict_proba(X_encoded)[0, 1]  # Probability of fraud (class 1)
    is_fraud = pred_proba >= 0.5
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X_encoded)
    
    # Handle SHAP output format (might be array or list of arrays)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # For binary classification, index 1 is positive class
    
    # Create DataFrame with feature names and SHAP values
    shap_df = pd.DataFrame({
        'feature': features,
        'shap_value': shap_values[0]  # First (and only) row
    })
    
    # Filter to interpretable features only
    interpretable_shap = shap_df[shap_df['feature'].isin(INTERPRETABLE_FEATURES)]
    
    # Sort by absolute SHAP value (impact magnitude) and take top 5
    top_drivers = interpretable_shap.reindex(
        interpretable_shap['shap_value'].abs().sort_values(ascending=False).index
    ).head(5)
    
    # Format drivers with direction and magnitude
    drivers = []
    for _, row in top_drivers.iterrows():
        drivers.append({
            'feature': row['feature'],
            'impact': abs(row['shap_value']),
            'direction': 'increases_risk' if row['shap_value'] > 0 else 'decreases_risk',
            'shap_value': row['shap_value']
        })
    
    return {
        'prediction': pred_proba,
        'is_fraud': is_fraud,
        'drivers': drivers
    }


def format_explanation(explanation):
    """Pretty print explanation for console output"""
    print("=" * 80)
    print("FRAUD PREDICTION EXPLANATION")
    print("=" * 80)
    print(f"\nFraud Probability: {explanation['prediction']:.1%}")
    print(f"Decision: {'FRAUD' if explanation['is_fraud'] else 'LEGITIMATE'}")
    
    print(f"\nTop 5 Risk Drivers:")
    print("-" * 80)
    for i, driver in enumerate(explanation['drivers'], 1):
        sign = '+' if driver['direction'] == 'increases_risk' else '-'
        direction_text = 'increases' if driver['direction'] == 'increases_risk' else 'decreases'
        print(f"{i}. {driver['feature']:20s} {sign} {driver['impact']:.4f}  ({direction_text} fraud risk)")
    print("=" * 80)


class FraudExplainer:
    """
    Production-ready fraud explainer (serving layer component)
    
    WHY A CLASS:
      - Load artifacts once at startup (expensive ~1-2s)
      - Reuse for many transactions (each call <100ms)
      - Standard pattern for Gradio/HuggingFace Spaces
      - Matches sklearn.Pipeline, shap.Explainer, transformers.pipeline
    
    Usage:
      explainer = FraudExplainer()  # Load once
      result = explainer.explain(transaction)  # Fast, reusable
    """
    
    def __init__(self, model_dir='outputs/models'):
        """
        Load model, encoders, SHAP background once
        After init, each explain() call is fast (<100ms)
        """
        print("Loading fraud detection model...")
        
        # Load model, encoders, features
        self.model, self.encoders, self.features = load_model_artifacts(model_dir)
        print(f"✓ Model loaded with {len(self.features)} features")
        
        # Load SHAP background data
        background_path = os.path.join(model_dir, 'shap_background.pkl')
        with open(background_path, 'rb') as f:
            self.background_data = pickle.load(f)
        print(f"✓ SHAP background loaded ({len(self.background_data)} samples)")
        
        # Create SHAP explainer
        self.explainer = create_shap_explainer(self.model, self.background_data)
        print("✓ SHAP explainer ready\n")
    
    def explain(self, transaction_dict):
        """
        Explain a single transaction (fast <100ms, reusable)
        
        Args:
            transaction_dict: Dict with all training features
                             {'TransactionAmt': 450.0, 'ProductCD': 'W', ...}
        
        Returns:
            dict with prediction (float), is_fraud (bool), drivers (list of top 5)
        """
        # Convert dict to DataFrame
        transaction_df = pd.DataFrame([transaction_dict])
        
        # Get explanation
        explanation = get_prediction_explanation(
            self.model, 
            self.explainer, 
            self.encoders, 
            transaction_df, 
            self.features
        )
        
        return explanation
