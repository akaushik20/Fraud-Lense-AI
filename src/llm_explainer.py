import os
from huggingface_hub import InferenceClient

# ── Model config ──────────────────────────────────────────────────
# Phase 1: Mistral 7B via HF Inference API
# Phase 2: swap MODEL_ID to your fine-tuned adapter on HF Hub
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

client = InferenceClient(
    model=MODEL_ID,
    token=os.environ.get("HF_TOKEN")
)

# ── Prompt ────────────────────────────────────────────────────────
EXPLANATION_PROMPT = """You are a financial compliance officer writing
a regulatory explanation for a flagged transaction.

Write exactly 3-4 sentences in plain English. Rules:
- Be factual and specific to the factors listed below
- Do not accuse the customer of fraud
- Do not reference any factor not listed below
- Use clear language suitable for a compliance report

Transaction details:
- Risk score: {risk_score}
- Transaction amount: ${amount}
- Product type: {product}

Key risk factors identified:
{drivers}

Explanation:"""

def generate_explanation(shap_drivers: list,
                         risk_score: float,
                         transaction: dict) -> str:
    """
    Convert SHAP drivers into a plain-English regulatory explanation.

    Args:
        shap_drivers: list of dicts from FraudExplainer.explain()
                      each with keys: feature, impact, direction, shap_value
        risk_score:   float 0-1 from model.predict_proba()
        transaction:  dict of raw user inputs from the Gradio form

    Returns:
        str: 3-4 sentence ECOA-compliant explanation
    """
    # Format the drivers into readable lines
    drivers_text = "\n".join([
        f"- {d['feature']}: {d['direction']} "
        f"(impact score: {d['shap_value']:.3f})"
        for d in shap_drivers
    ])

    prompt = EXPLANATION_PROMPT.format(
        risk_score=f"{risk_score:.1%}",
        amount=transaction.get("TransactionAmt", "unknown"),
        product=transaction.get("ProductCD", "unknown"),
        drivers=drivers_text
    )

    try:
        response = client.text_generation(
            prompt,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=False,
        )
        return response.strip()

    except Exception as e:
        # Graceful fallback — app still works if LLM call fails; log detail, don't expose it in the UI
        print(f"LLM explanation failed: {e}")
        return (
            f"This transaction received a risk score of {risk_score:.1%}. "
            f"The primary contributing factors were: "
            f"{', '.join([d['feature'] for d in shap_drivers[:3]])}. "
            f"Manual review is recommended."
        )