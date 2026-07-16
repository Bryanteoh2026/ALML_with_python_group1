"""Interactive Streamlit website for Telco customer churn prediction."""

from pathlib import Path

import joblib
import streamlit as st

from project_utils import prepare_single_customer


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_churn_model.joblib"


st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="wide",
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def make_recommendations(data: dict, churn_probability: float) -> list[str]:
    recommendations = []

    if data["Contract"] == "Month-to-month":
        recommendations.append("Offer a discounted one-year contract or loyalty reward.")
    if data["PaymentMethod"] == "Electronic check":
        recommendations.append("Encourage automatic bank or credit-card payment.")
    if data["TechSupport"] == "No":
        recommendations.append("Provide a free technical-support trial.")
    if data["OnlineSecurity"] == "No":
        recommendations.append("Bundle online security at a promotional price.")
    if data["tenure"] <= 12:
        recommendations.append("Use an early-customer onboarding and follow-up campaign.")
    if data["MonthlyCharges"] >= 80:
        recommendations.append("Review the plan and offer a lower-cost or better-value bundle.")
    if not recommendations and churn_probability >= 0.50:
        recommendations.append("Contact the customer for a personalised retention discussion.")
    if churn_probability < 0.50:
        recommendations.append("Maintain service quality and offer a renewal reminder before contract end.")

    return recommendations


st.title("📡 Telco Customer Churn Prediction Website")
st.write(
    "Select a customer's service and account conditions. The trained machine-learning model "
    "will estimate whether the customer is likely to discontinue the service."
)

if not MODEL_PATH.exists():
    st.error("The trained model is missing. Run 01_data_cleaning.py and 02_model_training_analysis.py first.")
    st.stop()

model = load_model()

with st.form("customer_form"):
    st.subheader("Customer Profile")
    profile_col1, profile_col2, profile_col3 = st.columns(3)

    with profile_col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])

    with profile_col2:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    with profile_col3:
        monthly_charges = st.slider("Monthly Charges", 18.0, 120.0, 70.0, 0.5)
        estimated_total = monthly_charges * tenure
        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=9000.0,
            value=float(round(estimated_total, 2)),
            step=10.0,
        )
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"],
        )

    st.subheader("Internet and Optional Services")
    service_col1, service_col2, service_col3 = st.columns(3)

    with service_col1:
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

    with service_col2:
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

    with service_col3:
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)

if submitted:
    customer_data = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    customer_df = prepare_single_customer(customer_data)
    probability = float(model.predict_proba(customer_df)[0, 1])
    prediction = int(probability >= 0.50)

    st.divider()
    result_col1, result_col2 = st.columns([1, 2])

    with result_col1:
        st.metric("Estimated Churn Probability", f"{probability:.1%}")
        if probability >= 0.70:
            st.error("High churn risk")
        elif probability >= 0.40:
            st.warning("Medium churn risk")
        else:
            st.success("Low churn risk")

    with result_col2:
        if prediction == 1:
            st.subheader("Prediction: Customer may discontinue the service")
        else:
            st.subheader("Prediction: Customer is likely to continue the service")

        st.write("Suggested marketing or retention actions:")
        for recommendation in make_recommendations(customer_data, probability):
            st.write(f"• {recommendation}")

    st.caption(
        "This prediction is a decision-support estimate, not a guarantee. Human review and fair treatment are still required."
    )
