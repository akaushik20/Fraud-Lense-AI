import gradio as gr
import json
from pathlib import Path
import pandas as pd

from src.explainer import FraudExplainer
from src.helper import INTERPRETABLE_FEATURES

CATEGORICAL_INTERPRETABLE_FEATURES = [
    f for f in INTERPRETABLE_FEATURES if f != 'TransactionAmt'
]

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

def load_feature_defaults():
    """Load pre-computed default values for non-interpretable features"""
    defaults_path = Path("outputs/models/feature_defaults.json")

    if defaults_path.exists():
        with open(defaults_path, 'r') as f:
            return json.load(f)
    else:
        print(f"Warning: {defaults_path} not found.")
        print("Run 'python compute_feature_defaults.py' to generate feature defaults.")
        return {}


def build_transaction(amt, product_cd, p_emaildomain, r_emaildomain, device_type, device_info):
    """Overlay the 6 user-facing form inputs on top of the pre-computed defaults"""
    transaction = feature_defaults.copy()
    transaction.update({
        'TransactionAmt': amt,
        'ProductCD': product_cd,
        'P_emaildomain': p_emaildomain,
        'R_emaildomain': r_emaildomain,
        'DeviceType': device_type,
        'DeviceInfo': device_info,
    })
    return transaction


def explain_transaction(amt, product_cd, p_emaildomain, r_emaildomain, device_type, device_info):
    """Score one transaction and return a risk label + SHAP driver chart"""
    transaction = build_transaction(
        amt, product_cd, p_emaildomain, r_emaildomain, device_type, device_info
    )
    result = fraud_explainer.explain(transaction)

    risk_label = {
        "Likely Fraud": float(result['prediction']),
        "Likely Legitimate": float(1 - result['prediction']),
    }

    drivers_df = pd.DataFrame(result['drivers'])

    return risk_label, drivers_df


if __name__ == "__main__":
    # Load metrics and SHAP data at startup
    metrics = load_metrics()
    shap_df = load_shap_data()

    # Load the single-transaction explainer once (model, encoders, SHAP background)
    fraud_explainer = FraudExplainer()
    feature_defaults = load_feature_defaults()

    # Dropdown choices come from the fitted encoders, so every option is a
    # category the model actually saw during training
    categorical_choices = {
        feature: sorted(fraud_explainer.encoders[feature].classes_.tolist())
        for feature in CATEGORICAL_INTERPRETABLE_FEATURES
        if feature in fraud_explainer.encoders
    }

    # Create a Blocks interface - this gives us full control over layout
    with gr.Blocks() as demo:

        # ROW 1: Header + one-liner context
        gr.Markdown("# FraudLens v2 — Fraud Investigation System")
        gr.Markdown(f"*Trained on {metrics['total_transactions']:,} real e-commerce transactions · IEEE-CIS Dataset · XGBoost + SHAP*")

        with gr.Tabs():
            with gr.Tab("Model Overview"):
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
                    gr.Markdown("*How much does each feature matter in determining if a transaction is fraudulent*")

                    gr.BarPlot(
                        value=shap_df,
                        x="mean_abs_shap",
                        y="feature",
                        show_label=False,
                        height=500,
                        x_title="Feature Importance (Mean Absolute SHAP)",
                        y_title="Feature"
                    )

            with gr.Tab("Explain a Transaction"):
                gr.Markdown("### Score a single transaction")
                gr.Markdown("*Fill in what you know — everything else uses a typical training-data default*")

                with gr.Row():
                    amt_input = gr.Number(label="Transaction Amount ($)", value=100.0)
                    product_input = gr.Dropdown(
                        label="Product Code",
                        choices=categorical_choices.get('ProductCD', []),
                    )
                    device_type_input = gr.Dropdown(
                        label="Device Type",
                        choices=categorical_choices.get('DeviceType', []),
                    )

                with gr.Row():
                    p_email_input = gr.Dropdown(
                        label="Purchaser Email Domain",
                        choices=categorical_choices.get('P_emaildomain', []),
                    )
                    r_email_input = gr.Dropdown(
                        label="Recipient Email Domain",
                        choices=categorical_choices.get('R_emaildomain', []),
                    )
                    device_info_input = gr.Dropdown(
                        label="Device Info",
                        choices=categorical_choices.get('DeviceInfo', []),
                    )

                explain_button = gr.Button("Check Transaction", variant="primary")

                with gr.Row():
                    risk_output = gr.Label(label="Fraud Risk")
                    drivers_output = gr.BarPlot(
                        x="shap_value",
                        y="feature",
                        color="direction",
                        color_map={
                            "increases_risk": "#ef4444",  # red
                            "decreases_risk": "#22c55e"   # green
                        },
                        show_label=False,
                        height=350,
                        x_title="SHAP Value (← decreases risk | increases risk →)",
                        y_title="Feature",
                        label="Top Drivers for This Transaction",
                    )

                explain_button.click(
                    fn=explain_transaction,
                    inputs=[
                        amt_input, product_input, p_email_input,
                        r_email_input, device_type_input, device_info_input,
                    ],
                    outputs=[risk_output, drivers_output],
                )

    # Launch the web interface
    demo.launch()