# =========================================================
# SIMPLE FLASK WEBSITE FOR CHURN PREDICTION
# =========================================================

import json
import os

import joblib
import pandas as pd
from flask import Flask, render_template, request


BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_FOLDER, "models", "best_model.joblib")
INFO_FILE = os.path.join(BASE_FOLDER, "models", "model_information.json")

app = Flask(__name__)


# Fields shown on the website.
# The groups make the form easier to understand.
FIELD_GROUPS = [
    {
        "title": "1. Usage & Billing",
        "icon": "📱",
        "description": "Tell us how the customer uses and pays for the service.",
        "numeric_fields": [
            ("MonthlyRevenue", "Monthly Revenue ($)", "0.01", "Example: 55.90"),
            ("MonthlyMinutes", "Monthly Minutes", "1", "Example: 420"),
            ("TotalRecurringCharge", "Recurring Charge ($)", "0.01", "Example: 48.00"),
            ("OverageMinutes", "Overage Minutes", "1", "Example: 12"),
            ("RoamingCalls", "Roaming Calls", "0.1", "Example: 1.5"),
        ],
        "select_fields": [],
    },
    {
        "title": "2. Service Experience",
        "icon": "🛠️",
        "description": "Add information about service quality and account history.",
        "numeric_fields": [
            ("DroppedCalls", "Dropped Calls", "0.1", "Example: 3"),
            ("CustomerCareCalls", "Customer Care Calls", "0.1", "Example: 2"),
            ("MonthsInService", "Months in Service", "1", "Example: 18"),
            ("CurrentEquipmentDays", "Current Phone Age (Days)", "1", "Example: 365"),
        ],
        "select_fields": [
            ("HandsetWebCapable", "Web-Capable Handset"),
            ("HandsetRefurbished", "Refurbished Handset"),
        ],
    },
    {
        "title": "3. Retention & Profile",
        "icon": "🎯",
        "description": "Finish with retention activity and basic customer profile details.",
        "numeric_fields": [
            ("RetentionCalls", "Retention Calls", "1", "Example: 0"),
            ("RetentionOffersAccepted", "Retention Offers Accepted", "1", "Example: 0"),
            ("IncomeGroup", "Income Group (0 to 9)", "1", "Example: 5"),
            ("HandsetPrice", "Handset Price ($)", "1", "Example: 80"),
        ],
        "select_fields": [
            ("MadeCallToRetentionTeam", "Called Retention Team"),
            ("CreditRating", "Credit Rating"),
            ("PrizmCode", "Area Type"),
            ("MaritalStatus", "Marital Status"),
        ],
    },
]


NUMERIC_FIELD_NAMES = [
    field[0]
    for group in FIELD_GROUPS
    for field in group["numeric_fields"]
]

SELECT_FIELD_NAMES = [
    field[0]
    for group in FIELD_GROUPS
    for field in group["select_fields"]
]


def load_project_files():
    """Load the trained model and saved model information."""
    if not os.path.exists(MODEL_FILE) or not os.path.exists(INFO_FILE):
        return None, None

    model = joblib.load(MODEL_FILE)

    with open(INFO_FILE, "r", encoding="utf-8") as file:
        information = json.load(file)

    return model, information


def create_recommendations(values):
    """Give simple rule-based retention suggestions."""
    recommendations = []

    if values["OverageMinutes"] > 50:
        recommendations.append(
            ("More minutes", "Offer a plan with more included minutes to reduce overage charges.")
        )

    if values["DroppedCalls"] > 5:
        recommendations.append(
            ("Network check", "Check network quality and contact the customer about dropped calls.")
        )

    if values["CustomerCareCalls"] > 3:
        recommendations.append(
            ("Proactive support", "Assign a service agent for a friendly follow-up call.")
        )

    if values["CurrentEquipmentDays"] > 500:
        recommendations.append(
            ("Device upgrade", "Offer a handset upgrade, trade-in, or device discount.")
        )

    if values["MonthlyRevenue"] > 80:
        recommendations.append(
            ("Loyalty reward", "Offer a loyalty reward to recognise this high-value customer.")
        )

    if values["MadeCallToRetentionTeam"] == "Yes":
        recommendations.append(
            ("Fast response", "Follow up quickly because the customer already contacted retention.")
        )

    if len(recommendations) == 0:
        recommendations.append(
            ("Keep connected", "Continue normal service and monitor future customer changes.")
        )

    return recommendations


def read_form_values(category_options):
    """Read and validate the form. Return values, old form data, and an error message."""
    form_data = request.form.to_dict()
    values = {}

    for field_name in NUMERIC_FIELD_NAMES:
        entered_value = request.form.get(field_name, "").strip()

        if entered_value == "":
            return None, form_data, "Please complete every field before predicting."

        try:
            number = float(entered_value)
        except ValueError:
            return None, form_data, "Please enter valid numbers in all number fields."

        if number < 0:
            return None, form_data, "Number fields cannot contain negative values."

        if field_name == "IncomeGroup" and number > 9:
            return None, form_data, "Income Group must be between 0 and 9."

        values[field_name] = number

    for field_name in SELECT_FIELD_NAMES:
        selected_value = request.form.get(field_name, "").strip()
        allowed_values = category_options[field_name]

        if selected_value not in allowed_values:
            return None, form_data, "Please choose an option for every dropdown field."

        values[field_name] = selected_value

    return values, form_data, None


@app.route("/", methods=["GET", "POST"])
def home():
    model, information = load_project_files()

    if model is None:
        return "Please run python main.py first to train and save the model."

    result = None
    probability = None
    recommendations = []
    error_message = None
    form_data = {}

    if request.method == "POST":
        entered_values, form_data, error_message = read_form_values(
            information["category_options"]
        )

        if error_message is None:
            # Use typical training values only for model fields that are not
            # displayed on the website. All visible website fields are entered
            # by the user and start blank.
            model_values = information["defaults"].copy()
            model_values.update(entered_values)

            input_data = pd.DataFrame(
                [[model_values[column] for column in information["feature_columns"]]],
                columns=information["feature_columns"],
            )

            probability = float(model.predict_proba(input_data)[0][1])
            threshold = float(information["threshold"])

            if probability >= threshold:
                result = "Likely to Churn"
            else:
                result = "Likely to Stay"

            recommendations = create_recommendations(entered_values)

    total_fields = len(NUMERIC_FIELD_NAMES) + len(SELECT_FIELD_NAMES)

    return render_template(
        "index.html",
        model_name=information["best_model_name"],
        threshold=information["threshold"],
        category_options=information["category_options"],
        field_groups=FIELD_GROUPS,
        total_fields=total_fields,
        result=result,
        probability=probability,
        recommendations=recommendations,
        error_message=error_message,
        form_data=form_data,
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
