import gradio as gr
import json
from pathlib import Path
import pandas as pd

# Load metrics from JSON file
def load_metrics():
    """Load pre-computed model metrics from JSON"""
    metrics_path = Path("outputs/models/metrics.json")
    
    # Default values if file doesn't exist yet
    default_metrics = {
        "auc_score": 0.89,
        "fraud_rate_percent": 3.5,
        "total_transactions": 590540,
        "original_features": 434,
        "selected_features": 90
    }
    
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            return json.load(f)
    else:
        print(f"Warning: {metrics_path} not found. Using default values.")
        print("Run 'python save_metrics.py' to generate actual metrics.")
        return default_metrics

def load_shap_data():
    """Load pre-computed SHAP feature importance from JSON"""
    shap_path = Path("outputs/models/shap_feature_importance.json")
    
    if shap_path.exists():
        with open(shap_path, 'r') as f:
            data = json.load(f)
            # Convert to DataFrame for plotting
            df = pd.DataFrame(data['features'])
            # Only show interpretable features on the chart
            df = df[df['is_interpretable'] == True]
            # Sort by mean_abs_shap descending (already sorted, but explicit)
            df = df.sort_values('mean_abs_shap', ascending=False)
            return df
    else:
        print(f"Warning: {shap_path} not found.")
        print("Run 'python compute_shap_importance.py' to generate SHAP data.")
        return pd.DataFrame()  # Empty DataFrame

# Simple function that will be called when user interacts with the app
def predict_fraud():
    """Placeholder function for fraud prediction"""
    return "Fraud detection system ready!"

if __name__ == "__main__":
    # Load metrics and SHAP data at startup
    metrics = load_metrics()
    shap_df = load_shap_data()
    
    # Create a Blocks interface - this gives us full control over layout
    with gr.Blocks() as demo:
        
        # ROW 1: Header + one-liner context
        gr.Markdown("# FraudLens v2 — Fraud Investigation System")
        gr.Markdown(f"*Trained on {metrics['total_transactions']:,} real e-commerce transactions · IEEE-CIS Dataset · XGBoost + SHAP*")
        
        # ROW 2: 4 metric cards side by side
        # gr.Row() arranges components horizontally
        with gr.Row():
            # Card 1: AUC Score
            with gr.Column():
                gr.Markdown("### AUC Score")
                gr.Markdown(f"# {metrics['auc_score']}")
                gr.Markdown("*How well the model separates fraud from legitimate*")
            
            # Card 2: Fraud Rate
            with gr.Column():
                gr.Markdown("### Fraud Rate (training data)")
                gr.Markdown(f"# {metrics['fraud_rate_percent']}%")
                gr.Markdown("*Class imbalance the model was built to handle*")
            
            # Card 3: Total Transactions
            with gr.Column():
                gr.Markdown("### Total transactions trained on")
                gr.Markdown(f"# {metrics['total_transactions']:,}")
                gr.Markdown("*Scale of real data*")
            
            # Card 4: Features Used
            with gr.Column():
                gr.Markdown("### Features used")
                gr.Markdown(f"# {metrics['original_features']} → {metrics['selected_features']}")
                gr.Markdown("*Full model power, explainable subset*")
        
        # ROW 3: SHAP Feature Importance Chart
        if not shap_df.empty:
            gr.Markdown("### Top Features Driving Fraud Predictions")
            gr.Markdown("*Based on mean absolute SHAP values · Red = increases fraud risk · Green = decreases risk*")
            
            gr.BarPlot(
                value=shap_df,
                x="mean_abs_shap",
                y="feature",
                color="direction",
                color_map={
                    "increases_risk": "#ef4444",  # red
                    "decreases_risk": "#22c55e"   # green
                },
                show_label=False,
                height=500,
                x_title="Mean Absolute SHAP Value",
                y_title="Feature"
            )
        
    
    # Launch the web interface
    demo.launch()