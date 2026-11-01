import os
import re
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
- Write one plain paragraph of 5-6 sentences, no bullets, suitable for a compliance report.
- Use factor names exactly as written."""

EXPLANATION_PROMPT = """Model risk score: {risk_score} ({tier} risk)

Signals the model weighed:
{drivers}

RECOMMENDED ACTION: {action}

{steps}"""

STANDARD_STEPS = """Write the note in this order:
1. State the score and the risk tier.
2. Describe the factors that raised the score, then the factors that lowered it, naming each value and its strength. If a group is "none", say no factors pushed in that direction.
3. Add one sentence that inputs not collected in this demo were set to typical training-data values.
4. End with the recommended action, worded as given."""

# Used for MEDIUM/HIGH when at least one factor raised the score
ELEVATED_STEPS = """Write the note in this order:
1. State the score and the risk tier.
2. Spend most of the note on the factors under KEY RISK DRIVERS, strongest first. Name every one of them, with its value and strength. Mention any "Other factors that raised the score" briefly.
3. In one sentence, mention the factors that lowered the score as partial offsets. If there are none, skip this.
4. Add one sentence that inputs not collected in this demo were set to typical training-data values.
5. End with the recommended action, worded as given."""

# ── Decision logic (code decides; the LLM only narrates) ──────────
# Note tiers; the UI label in app.py uses its own 0.5 "Likely Fraud" cut-off
HIGH_RISK_THRESHOLD = 0.1
MODERATE_RISK_THRESHOLD = 0.03
ELEVATED_TIERS = {"MEDIUM", "HIGH"}
MAX_KEY_DRIVERS = 3

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


def raised_drivers(shap_drivers: list) -> list:
    """Risk-raising drivers (positive SHAP), strongest first."""
    raised = [d for d in shap_drivers if d["shap_value"] > 0]
    return sorted(raised, key=lambda d: d["shap_value"], reverse=True)


def lowered_drivers(shap_drivers: list) -> list:
    return [d for d in shap_drivers if d["shap_value"] <= 0]


def select_key_drivers(shap_drivers: list, tier: str) -> list:
    """Top risk-raising drivers to highlight; empty for LOW risk."""
    if tier not in ELEVATED_TIERS:
        return []
    return raised_drivers(shap_drivers)[:MAX_KEY_DRIVERS]


def build_facts(shap_drivers: list, transaction: dict, key_drivers: list) -> str:
    """Render drivers in groups; with key_drivers, the top risk-raisers get their own section."""
    lowered_section = (
        f"Factors that lowered the risk score:\n"
        f"{format_group(lowered_drivers(shap_drivers), transaction)}"
    )
    if not key_drivers:
        raised_section = (
            f"Factors that raised the risk score:\n"
            f"{format_group(raised_drivers(shap_drivers), transaction)}"
        )
        return f"{raised_section}\n\n{lowered_section}"

    key_features = {d["feature"] for d in key_drivers}
    others = [d for d in raised_drivers(shap_drivers) if d["feature"] not in key_features]
    sections = [
        "KEY RISK DRIVERS (raised the score, strongest first):\n"
        f"{format_group(key_drivers, transaction)}"
    ]
    if others:
        sections.append(
            f"Other factors that raised the score:\n{format_group(others, transaction)}"
        )
    sections.append(lowered_section)
    return "\n\n".join(sections)


def display_name(feature: str) -> str:
    return FEATURE_DISPLAY_NAMES.get(feature, feature).lower()


def bold_key_drivers(text: str, key_drivers: list) -> str:
    """Wrap each key driver's name in markdown bold wherever the note mentions it."""
    for driver in key_drivers:
        pattern = rf"(?<!\*)\b({re.escape(display_name(driver['feature']))})\b(?!\*)"
        text = re.sub(pattern, r"**\1**", text, flags=re.IGNORECASE)
    return text


def format_group(drivers: list, transaction: dict) -> str:
    if not drivers:
        return "none"
    return "\n".join(format_driver_line(d, transaction) for d in drivers)


def format_driver_line(driver: dict, transaction: dict) -> str:
    feature = driver["feature"]
    value = format_value(feature, transaction.get(feature, "unknown"))
    strength = signal_strength(driver["shap_value"])
    return f"- {display_name(feature)} = {value} ({strength})"


def build_messages(shap_drivers: list, risk_score: float, transaction: dict) -> list:
    tier = risk_tier(risk_score)
    key_drivers = select_key_drivers(shap_drivers, tier)
    user_prompt = EXPLANATION_PROMPT.format(
        risk_score=f"{risk_score:.1%}",
        tier=tier,
        drivers=build_facts(shap_drivers, transaction, key_drivers),
        action=ACTION_BY_TIER[tier],
        steps=ELEVATED_STEPS if key_drivers else STANDARD_STEPS,
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
        note = response.choices[0].message.content.strip()
        key_drivers = select_key_drivers(shap_drivers, risk_tier(risk_score))
        return bold_key_drivers(note, key_drivers)

    except Exception as e:
        # Graceful fallback — app still works if LLM call fails; log detail, don't expose it in the UI
        print(f"LLM explanation failed: {e!r}")
        return build_fallback_explanation(shap_drivers, risk_score)


def build_fallback_explanation(shap_drivers: list, risk_score: float) -> str:
    tier = risk_tier(risk_score)
    raised = ", ".join(display_name(d["feature"]) for d in raised_drivers(shap_drivers)) or "none"
    note = (
        f"This transaction received a risk score of {risk_score:.1%} ({tier} risk). "
        f"Factors that increased the risk score: {raised}. "
        f"{ACTION_BY_TIER[tier]}"
    )
    return bold_key_drivers(note, select_key_drivers(shap_drivers, tier))