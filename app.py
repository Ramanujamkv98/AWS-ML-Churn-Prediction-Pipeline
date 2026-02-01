import os
import streamlit as st

# Make sure inference.py can find the saved artifacts
os.environ["MODEL_DIR"] = "/tmp/model"

from inference import predict

st.set_page_config(page_title="Churn Risk Predictor", layout="centered")
st.title("Churn Risk Predictor")
st.caption("Select a model and input feature values to score churn risk.")

# -----------------------
# Model controls
# -----------------------
model_choice = st.selectbox("Choose Model", ["xgb", "logreg"], index=0)
threshold = st.slider("Decision Threshold", 0.05, 0.95, 0.35, 0.01)

st.subheader("Inputs")

# -----------------------
# Inputs
# -----------------------
tokens_per_session_7d = st.number_input("tokens_per_session_7d", value=300.0, min_value=0.0)

sessions_7d = st.number_input("sessions_7d", value=8, min_value=0)

active_days_7d = st.number_input("active_days_7d (0–7)", value=3, min_value=0, max_value=7)

avg_turns_per_session_7d = st.number_input("avg_turns_per_session_7d", value=8.0, min_value=0.0)

error_rate_7d = st.number_input("error_rate_7d (0–1)", value=0.05, min_value=0.0, max_value=1.0)

avg_latency_ms_7d = st.number_input("avg_latency_ms_7d", value=600.0, min_value=0.0)

sessions_trend_7d = st.number_input("sessions_trend_7d", value=0.0)

tokens_trend_7d = st.number_input("tokens_trend_7d", value=0.0)

model_switch_rate_7d = st.number_input("model_switch_rate_7d (0–1)", value=0.10, min_value=0.0, max_value=1.0)

activated_800 = st.selectbox("activated_800", [0, 1], index=0)

obs_month = st.number_input("obs_month (1–12)", value=1, min_value=1, max_value=12)

obs_dow = st.number_input("obs_dow (0=Mon ... 6=Sun)", value=2, min_value=0, max_value=6)

# NOTE: Your model expects gpt_4.1 / gpt_4o / other one-hots.
# We map UI selections into those buckets.
primary_model_7d = st.selectbox(
    "primary_model_7d",
    ["gpt-4o", "gpt-4.1", "gpt-4o-mini", "gpt-3.5", "other"],
    index=0
)

# -----------------------
# Build payload (pre-validation)
# -----------------------
payload = {
    "tokens_per_session_7d": float(tokens_per_session_7d),
    "sessions_7d": int(sessions_7d),
    "active_days_7d": int(active_days_7d),
    "avg_turns_per_session_7d": float(avg_turns_per_session_7d),
    "error_rate_7d": float(error_rate_7d),
    "avg_latency_ms_7d": float(avg_latency_ms_7d),
    "sessions_trend_7d": float(sessions_trend_7d),
    "tokens_trend_7d": float(tokens_trend_7d),
    "model_switch_rate_7d": float(model_switch_rate_7d),
    "activated_800": int(activated_800),
    "obs_month": int(obs_month),
    "obs_dow": int(obs_dow),
    "primary_model_7d": str(primary_model_7d),
}

# -----------------------
# Validation / auto-fix (prevents unrealistic combos)
# -----------------------
fixes = []

# sessions vs active days
if payload["sessions_7d"] > 0 and payload["active_days_7d"] == 0:
    payload["active_days_7d"] = 1
    fixes.append("active_days_7d set to 1 because sessions_7d > 0")

if payload["active_days_7d"] > 0 and payload["sessions_7d"] == 0:
    payload["sessions_7d"] = payload["active_days_7d"]
    fixes.append("sessions_7d set to active_days_7d because active_days_7d > 0")

# if no sessions, per-session metrics should be 0
if payload["sessions_7d"] == 0:
    if payload["avg_turns_per_session_7d"] != 0.0:
        payload["avg_turns_per_session_7d"] = 0.0
        fixes.append("avg_turns_per_session_7d set to 0 because sessions_7d = 0")

    if payload["tokens_per_session_7d"] != 0.0:
        payload["tokens_per_session_7d"] = 0.0
        fixes.append("tokens_per_session_7d set to 0 because sessions_7d = 0")

    if payload["model_switch_rate_7d"] != 0.0:
        payload["model_switch_rate_7d"] = 0.0
        fixes.append("model_switch_rate_7d set to 0 because sessions_7d = 0")

# display fixes (if any)
if fixes:
    st.info("Auto-fixes applied to keep the input profile realistic:\n- " + "\n- ".join(fixes))

# Show payload preview
with st.expander("Show request payload"):
    st.json(payload)

# -----------------------
# Predict
# -----------------------
if st.button("Predict"):
    out = predict(payload, model_choice=model_choice, threshold=threshold)

    st.subheader("Result")
    st.metric("Churn Probability", out["churn_probability"])
    st.write("Prediction:", "🚨Churn Risk" if out["churn_prediction"] else "✅Stable")

    with st.expander("Show full model output"):
        st.json(out)
