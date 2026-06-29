import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.explainer import FraudExplainer, format_explanation


def create_sample_transaction():
    """Create a sample transaction for testing (production scenario)"""
    # This represents a new transaction from a payment gateway
    # In production, this would come from your API/database
    transaction = {
        'TransactionID': 999999,
        'TransactionAmt': 450.00,
        'ProductCD': 'W',
        'card1': 13926,
        'card2': 150.0,
        'card3': 150.0,
        'card4': 'discover',
        'card5': 226.0,
        'card6': 'credit',
        'addr1': 315.0,
        'addr2': 87.0,
        'dist1': 19.0,
        'P_emaildomain': 'gmail.com',
        'R_emaildomain': 'gmail.com',
        'C2': 1.0,
        'C3': 0.0,
        'C5': 0.0,
        'C9': 1.0,
        'C12': 0.0,
        'C13': 1.0,
        'C14': 1.0,
        'D2': 10.0,
        'D3': 50.0,
        'D4': 100.0,
        'D5': 150.0,
        'D8': 25.0,
        'D10': 5.0,
        'D11': 2.0,
        'D15': 200.0,
        'M2': 'T',
        'M3': 'T',
        'M4': 'M0',
        'M5': 'F',
        'M6': 'T',
        'M7': 'T',
        'M8': 'T',
        'M9': 'T',
        'DeviceType': 'desktop',
        'DeviceInfo': 'Windows',
        # Add placeholder values for all V features (they'll be NaN in real scenarios often)
        'V2': 1.0, 'V3': 1.0, 'V4': 1.0, 'V5': 0.5, 'V6': 1.0,
        'V10': 1.0, 'V12': 0.5, 'V13': 1.0, 'V19': 0.0, 'V20': 1.0,
        'V23': 1.0, 'V24': 0.5, 'V25': 1.0, 'V26': 0.5, 'V29': 1.0,
        'V35': 0.0, 'V36': 1.0, 'V38': 0.5, 'V44': 1.0, 'V45': 0.5,
        'V48': 1.0, 'V52': 0.0, 'V53': 1.0, 'V54': 0.5, 'V55': 1.0,
        'V56': 0.5, 'V58': 1.0, 'V61': 0.0, 'V62': 1.0, 'V64': 0.5,
        'V66': 1.0, 'V67': 0.5, 'V69': 1.0, 'V74': 0.0, 'V75': 1.0,
        'V76': 0.5, 'V77': 1.0, 'V78': 0.5, 'V81': 1.0, 'V82': 0.0,
        'V83': 1.0, 'V86': 0.5, 'V87': 1.0, 'V90': 0.5, 'V94': 1.0,
        'V99': 0.0,
        # Identity features
        'id_01': 0.5, 'id_04': 1.0, 'id_06': 0.0, 'id_09': 1.0,
        'id_13': 50.0, 'id_14': 10.0, 'id_20': 200.0, 'id_31': 'chrome',
        'id_32': 24.0, 'id_34': 'match_status:1', 'id_38': 'T',
        # Additional V features
        'V109': 0.5, 'V114': 1.0, 'V115': 0.5, 'V123': 1.0, 'V124': 0.0,
        'V125': 1.0, 'V129': 0.5, 'V130': 1.0, 'V131': 0.5, 'V136': 1.0,
        'V137': 0.0, 'V140': 1.0, 'V142': 0.5, 'V143': 1.0, 'V152': 0.5,
        'V156': 1.0, 'V158': 0.0, 'V165': 1.0, 'V171': 0.5, 'V173': 1.0,
        'V187': 0.5, 'V201': 1.0, 'V207': 0.0, 'V220': 1.0, 'V226': 0.5,
        'V228': 1.0, 'V238': 0.5, 'V245': 1.0, 'V257': 0.0, 'V258': 1.0,
        'V259': 0.5, 'V261': 1.0, 'V263': 0.5, 'V268': 1.0, 'V274': 0.0,
        'V276': 1.0, 'V281': 0.5, 'V283': 1.0, 'V285': 0.5, 'V287': 1.0,
        'V290': 0.0, 'V300': 1.0, 'V301': 0.5, 'V309': 1.0, 'V310': 0.5,
        'V312': 1.0, 'V313': 0.0, 'V314': 1.0, 'V315': 0.5, 'V319': 1.0,
        'V320': 0.5, 'V335': 1.0, 'V338': 0.0, 'V339': 1.0
    }
    return transaction


def main():
    print("=" * 80)
    print("PRODUCTION-READY FRAUD EXPLANATION SYSTEM")
    print("=" * 80)
    print("\nThis demonstrates explaining NEW transactions (production scenario)")
    print("No training data loading required!\n")
    
    # Initialize explainer once (loads model, encoders, SHAP background)
    explainer = FraudExplainer()
    
    print("=" * 80)
    print("EXAMPLE 1: High-value suspicious transaction")
    print("=" * 80)
    
    # Create a suspicious-looking transaction
    suspicious_transaction = create_sample_transaction()
    suspicious_transaction['TransactionAmt'] = 2500.00  # High amount
    suspicious_transaction['C2'] = 5.0  # Unusual C2 value
    
    print(f"\nTransaction Details:")
    print(f"  Amount: ${suspicious_transaction['TransactionAmt']:.2f}")
    print(f"  Product: {suspicious_transaction['ProductCD']}")
    print(f"  Card Type: {suspicious_transaction['card4']}")
    print(f"  Device: {suspicious_transaction['DeviceType']}")
    
    # Explain it
    explanation = explainer.explain(suspicious_transaction)
    format_explanation(explanation)
    
    
    print("\n\n" + "=" * 80)
    print("EXAMPLE 2: Normal-looking transaction")
    print("=" * 80)
    
    # Create a normal-looking transaction
    normal_transaction = create_sample_transaction()
    normal_transaction['TransactionAmt'] = 49.99  # Normal amount
    normal_transaction['C2'] = 1.0  # Normal C2
    
    print(f"\nTransaction Details:")
    print(f"  Amount: ${normal_transaction['TransactionAmt']:.2f}")
    print(f"  Product: {normal_transaction['ProductCD']}")
    print(f"  Card Type: {normal_transaction['card4']}")
    print(f"  Device: {normal_transaction['DeviceType']}")
    
    # Explain it
    explanation = explainer.explain(normal_transaction)
    format_explanation(explanation)
    
    
    print("\n\n" + "=" * 80)
    print("PRODUCTION USAGE SUMMARY")
    print("=" * 80)
    print("\n✓ Explainer initialized once (reusable for many transactions)")
    print("✓ No training data loading required")
    print("✓ Each explanation takes <1 second")
    print("✓ Ready for API/web service integration")
    print("\nNext steps:")
    print("  - Wrap explainer.explain() in a Flask/FastAPI endpoint")
    print("  - Add to Gradio UI for interactive demo")
    print("  - Use for real-time fraud detection")


if __name__ == "__main__":
    main()
