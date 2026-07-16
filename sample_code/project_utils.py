"""Shared helper functions for the Telco Customer Churn project."""

from __future__ import annotations

import numpy as np
import pandas as pd


SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def clean_basic_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply simple, transparent cleaning rules to the raw dataset."""
    cleaned = df.copy()

    # Remove extra spaces from text columns and convert blank strings to missing values.
    text_columns = cleaned.select_dtypes(include="object").columns
    for column in text_columns:
        cleaned[column] = cleaned[column].str.strip()
    cleaned = cleaned.replace(r"^\s*$", np.nan, regex=True)

    # Convert TotalCharges from text to a numeric column.
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")

    # The 11 blank TotalCharges records are new customers with zero tenure.
    # Therefore, zero is a logical value and avoids deleting valid new customers.
    new_customer_mask = cleaned["TotalCharges"].isna() & (cleaned["tenure"] == 0)
    cleaned.loc[new_customer_mask, "TotalCharges"] = 0.0

    # Remove duplicate rows and repeated customer IDs, if any appear in a future copy.
    cleaned = cleaned.drop_duplicates()
    cleaned = cleaned.drop_duplicates(subset="customerID", keep="first")

    # Remove records with an invalid target. The supplied dataset has none.
    cleaned = cleaned[cleaned["Churn"].isin(["Yes", "No"])].copy()

    return cleaned.reset_index(drop=True)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add two easy-to-explain features that are useful for churn analysis."""
    result = df.copy()

    # Count how many optional online services the customer subscribes to.
    available_service_columns = [c for c in SERVICE_COLUMNS if c in result.columns]
    result["NumOptionalServices"] = sum(
        (result[column] == "Yes").astype(int) for column in available_service_columns
    )

    # Group tenure into business-friendly customer lifecycle bands.
    tenure_labels = [
        "0-12 months",
        "13-24 months",
        "25-48 months",
        "49-60 months",
        "61-72 months",
    ]
    result["TenureGroup"] = pd.cut(
        result["tenure"],
        bins=[-1, 12, 24, 48, 60, np.inf],
        labels=tenure_labels,
    ).astype("object")

    return result


def prepare_single_customer(customer_data: dict) -> pd.DataFrame:
    """Convert website input into the same feature format used for training."""
    customer = pd.DataFrame([customer_data])
    customer = add_engineered_features(customer)
    return customer
