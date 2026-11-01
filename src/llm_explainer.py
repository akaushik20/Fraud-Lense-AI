import os
from huggingface_hub import InferenceClient

from src.helper import FEATURE_DISPLAY_NAMES

# ── Model config ──────────────────────────────────────────────────
# Phase 1: Mistral 7B via HF Inference API
# Phase 2: swap MODEL_ID to your fine-tuned adapter on HF Hub
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

client = InferenceClient(
    model=MODEL_ID,
    token=os.environ.get("HF_TOKEN")
)

# ── Prompt ────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a fraud-operations analyst writing a case note for a \
compliance reviewer. Rules:
- Use ONLY the facts provided. Never invent numbers, thresholds, history, or causes.
- Describe model signals, not customer intent. Do not accuse anyone of fraud.
- Do not recommend any action other than the one given under RECOMMENDED ACTION.
- Write one plain paragraph of 5-6 sentences, no bullets, suitable for a compliance report."""

EXPLANATION_PROMPT = """Model risk score: {risk_score} ({tier} risk)

Signals the model weighed:
{drivers}

RECOMMENDED ACTION: {action}

Write the note in this order:
1. State the score and the risk tier.
2. Describe the factors that raised the score, then the factors that lowered it, naming each value and its strength. If a group is "none", say no factors pushed in that direction.
3. Add one sentence that inputs not collected in this demo were set to typical training-data values.
4. End with the recommended action, worded as given."""

# ── Decision logic (code decides; the LLM only narrates) ──────────
# HIGH_RISK_THRESHOLD matches the 0.5 "Likely Fraud" cut-off in app.py/explainer.py
HIGH_RISK_THRESHOLD = 0.1
MODERATE_RISK_THRESHOLD = 0.03

ACTION_BY_TIER = {
    "LOW": "This score alone does not indicate a need for additional review.",
    "MEDIUM": "Secondary review is suggested.",
    "HIGH": "Manual review is recommended before the transaction is approved.",
}

STRONG_SHAP = 0.5
MODERATE_SHAP = 0.2


def risk_tier(risk_score: float) -> str:
    if risk_score >= HIGH_RISK_THRESHOLD:
        return "HIGH"
    if risk_score >= MODERATE_RISK_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def signal_strength(shap_value: float) -> str:
    magnitude = abs(shap_value)
    if magnitude >= STRONG_SHAP:
        return "strong"
    if magnitude >= MODERATE_SHAP:
        return "moderate"
    return "slight"


def format_value(feature: str, value) -> str:
    if feature == "TransactionAmt" and isinstance(value, (int, float)):
        return f"${value:,.2f}"
    return str(value)


def build_facts(shap_drivers: list, transaction: dict) -> str:
    """Render all drivers in two groups: factors that raised and factors that lowered the score."""
    raised = [d for d in shap_drivers if d["shap_value"] > 0]
    lowered = [d for d in shap_drivers if d["shap_value"] <= 0]
    return (
        f"Factors that raised the risk score:\n{format_group(raised, transaction)}\n\n"
        f"Factors that lowered the risk score:\n{format_group(lowered, transaction)}"
    )


def format_group(drivers: list, transaction: dict) -> str:
    if not drivers:
        return "none"
    return "\n".join(format_driver_line(d, transaction) for d in drivers)


def format_driver_line(driver: dict, transaction: dict) -> str:
    feature = driver["feature"]
    name = FEATURE_DISPLAY_NAMES.get(feature, feature).lower()
    value = format_value(feature, transaction.get(feature, "unknown"))
    strength = signal_strength(driver["shap_value"])
    return f"- {name} = {value} ({strength})"


def build_messages(shap_drivers: list, risk_score: float, transaction: dict) -> list:
    tier = risk_tier(risk_score)
    user_prompt = EXPLANATION_PROMPT.format(
        risk_score=f"{risk_score:.1%}",
        tier=tier,
        drivers=build_facts(shap_drivers, transaction),
        action=ACTION_BY_TIER[tier],
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

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
        str: 5-6 sentence case note for a compliance reviewer
    """
    messages = build_messages(shap_drivers, risk_score, transaction)

    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=300,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        # Graceful fallback — app still works if LLM call fails; log detail, don't expose it in the UI
        print(f"LLM explanation failed: {e!r}")
        return build_fallback_explanation(shap_drivers, risk_score)


def build_fallback_explanation(shap_drivers: list, risk_score: float) -> str:
    top_names = [
        FEATURE_DISPLAY_NAMES.get(d["feature"], d["feature"]).lower()
        for d in shap_drivers[:3]
    ]
    return (
        f"This transaction received a risk score of {risk_score:.1%} "
        f"({risk_tier(risk_score)} risk). "
        f"The strongest signals were: {', '.join(top_names)}. "
        f"{ACTION_BY_TIER[risk_tier(risk_score)]}"
    )