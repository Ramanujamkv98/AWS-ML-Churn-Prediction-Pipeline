import streamlit as st

from inference import predict


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Churn Risk Predictor",
    layout="centered",
)

st.title("Churn Risk Predictor")
st.caption(
    "Enter a user's recent activity to estimate their "
    "probability of churning within 14 days."
)


# --------------------------------------------------
# Model controls
# --------------------------------------------------
model_choice = st.selectbox(
    "Choose model",
    ["xgb", "logreg"],
    index=0,
)

threshold = st.slider(
    "Decision threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.35,
    step=0.01,
)

st.subheader("User activity inputs")


# --------------------------------------------------
# Behavioral inputs
# --------------------------------------------------
tokens_per_session_7d = st.number_input(
    "Average tokens per session (last 7 days)",
    min_value=0.0,
    value=300.0,
)

sessions_7d = st.number_input(
    "Sessions in the last 7 days",
    min_value=0,
    value=8,
    step=1,
)

active_days_7d = st.number_input(
    "Active days in the last 7 days",
    min_value=0,
    max_value=7,
    value=3,
    step=1,
)

avg_turns_per_session_7d = st.number_input(
    "Average turns per session",
    min_value=0.0,
    value=8.0,
)

error_rate_7d = st.number_input(
    "Error rate (0–1)",
    min_value=0.0,
    max_value=1.0,
    value=0.05,
    step=0.01,
)

avg_latency_ms_7d = st.number_input(
    "Average latency in milliseconds",
    min_value=0.0,
    value=600.0,
)

sessions_trend_7d = st.number_input(
    "Session trend",
    value=0.0,
)

tokens_trend_7d = st.number_input(
    "Token usage trend",
    value=0.0,
)

model_switch_rate_7d = st.number_input(
    "Model switch rate (0–1)",
    min_value=0.0,
    max_value=1.0,
    value=0.10,
    step=0.01,
)

primary_model_7d = st.selectbox(
    "Primary model used",
    [
        "gpt-4o",
        "gpt-4.1",
        "gpt-4o-mini",
        "gpt-3.5",
        "other",
    ],
    index=0,
)

obs_month = st.number_input(
    "Observation month (1–12)",
    min_value=1,
    max_value=12,
    value=1,
    step=1,
)

obs_dow = st.number_input(
    "Observation day of week (0=Monday, 6=Sunday)",
    min_value=0,
    max_value=6,
    value=2,
    step=1,
)


# --------------------------------------------------
# Build prediction payload
# --------------------------------------------------
payload = {
    "tokens_per_session_7d": float(tokens_per_session_7d),
    "sessions_7d": int(sessions_7d),
    "active_days_7d": int(active_days_7d),
    "avg_turns_per_session_7d": float(
        avg_turns_per_session_7d
    ),
    "error_rate_7d": float(error_rate_7d),
    "avg_latency_ms_7d": float(avg_latency_ms_7d),
    "sessions_trend_7d": float(sessions_trend_7d),
    "tokens_trend_7d": float(tokens_trend_7d),
    "model_switch_rate_7d": float(
        model_switch_rate_7d
    ),
    "primary_model_7d": str(primary_model_7d),
    "obs_month": int(obs_month),
    "obs_dow": int(obs_dow),
}


# --------------------------------------------------
# Validate and correct inconsistent inputs
# --------------------------------------------------
fixes = []

if (
    payload["sessions_7d"] > 0
    and payload["active_days_7d"] == 0
):
    payload["active_days_7d"] = 1
    fixes.append(
        "Active days was changed to 1 because sessions are greater than 0."
    )

if (
    payload["active_days_7d"] > 0
    and payload["sessions_7d"] == 0
):
    payload["sessions_7d"] = payload["active_days_7d"]
    fixes.append(
        "Sessions was changed to match active days."
    )

if payload["sessions_7d"] == 0:
    if payload["avg_turns_per_session_7d"] != 0:
        payload["avg_turns_per_session_7d"] = 0.0
        fixes.append(
            "Average turns was changed to 0 because sessions are 0."
        )

    if payload["tokens_per_session_7d"] != 0:
        payload["tokens_per_session_7d"] = 0.0
        fixes.append(
            "Tokens per session was changed to 0 because sessions are 0."
        )

    if payload["model_switch_rate_7d"] != 0:
        payload["model_switch_rate_7d"] = 0.0
        fixes.append(
            "Model switch rate was changed to 0 because sessions are 0."
        )


# activated_800 was engineered during training as:
# tokens_per_session_7d >= 800
payload["activated_800"] = int(
    payload["tokens_per_session_7d"] >= 800
)


if fixes:
    st.info(
        "Automatic input corrections:\n\n- "
        + "\n- ".join(fixes)
    )


with st.expander("Show model input payload"):
    st.json(payload)


# --------------------------------------------------
# Generate prediction
# --------------------------------------------------
if st.button(
    "Predict churn risk",
    type="primary",
    use_container_width=True,
):
    try:
        output = predict(
            payload,
            model_choice=model_choice,
            threshold=threshold,
        )

        probability = output["churn_probability"]
        prediction = output["churn_prediction"]

        st.subheader("Prediction result")

        st.metric(
            "Churn probability",
            f"{probability:.1%}",
        )

        if prediction:
            st.error(
                "Churn risk detected at the selected threshold."
            )
        else:
            st.success(
                "The user is classified as stable at the selected threshold."
            )

        with st.expander("Show complete model output"):
            st.json(output)

    except Exception as error:
        st.error(
            "The prediction could not be generated. "
            "Verify that all three model artifacts are available "
            "inside the models folder."
        )

        with st.expander("Technical error details"):
            st.exception(error)
