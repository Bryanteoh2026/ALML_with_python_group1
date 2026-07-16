"""Step 1: Clean and prepare the IBM Telco Customer Churn dataset."""

from pathlib import Path

import pandas as pd

from project_utils import add_engineered_features, clean_basic_data


BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = BASE_DIR / "data" / "telco_customer.csv"
CLEAN_DATA_PATH = BASE_DIR / "data" / "telco_customer_cleaned.csv"
CLEANING_REPORT_PATH = BASE_DIR / "outputs" / "data_cleaning_report.txt"


def main() -> None:
    df = pd.read_csv(RAW_DATA_PATH)

    original_rows, original_columns = df.shape
    original_duplicates = df.duplicated().sum()
    blank_total_charges = pd.to_numeric(df["TotalCharges"], errors="coerce").isna().sum()

    cleaned = clean_basic_data(df)
    cleaned = add_engineered_features(cleaned)

    cleaned.to_csv(CLEAN_DATA_PATH, index=False)

    report = f"""TELCO CUSTOMER CHURN - DATA CLEANING REPORT
================================================
Raw dataset shape: {original_rows} rows x {original_columns} columns
Cleaned dataset shape: {cleaned.shape[0]} rows x {cleaned.shape[1]} columns
Exact duplicate rows found: {original_duplicates}
Blank/non-numeric TotalCharges values found: {blank_total_charges}
Remaining missing values: {int(cleaned.isna().sum().sum())}

Cleaning decisions
------------------
1. Removed leading and trailing spaces from text columns.
2. Converted blank text to missing values.
3. Converted TotalCharges from text to numeric.
4. Set TotalCharges to 0 for new customers whose tenure is 0.
5. Removed duplicate rows and repeated customer IDs, if present.
6. Kept only valid Churn labels: Yes and No.
7. Added NumOptionalServices and TenureGroup as engineered features.

Important data-leakage note
---------------------------
Scaling, one-hot encoding and statistical imputation are NOT performed here.
They are fitted only on training data inside the modelling pipeline.
"""
    CLEANING_REPORT_PATH.write_text(report, encoding="utf-8")

    print(report)
    print(f"Cleaned CSV saved to: {CLEAN_DATA_PATH}")


if __name__ == "__main__":
    main()
