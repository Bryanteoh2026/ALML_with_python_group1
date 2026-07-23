# =========================================================
# DATA CLEANING FILE
# =========================================================

import pandas as pd
import numpy as np


def load_data(file_path):
    """Load the CSV dataset."""
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    return df


def check_basic_info(df):
    """Print simple information about the dataset."""
    print("\n--- BASIC DATA INFORMATION ---")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])
    print("\nTarget distribution:")
    print(df["Churn"].value_counts())


def check_missing_values(df):
    """Return a table showing missing values."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    result = pd.DataFrame({
        "MissingValues": missing,
        "MissingPercent": (missing / len(df) * 100).round(2)
    })

    print("\n--- MISSING VALUES ---")
    print(result if len(result) > 0 else "No missing values found.")
    return result


def check_duplicates(df):
    """Count exact duplicate rows."""
    duplicates = df.duplicated().sum()
    print("\nDuplicate rows:", duplicates)
    return duplicates


def check_outliers(df):
    """Count possible outliers using the IQR method."""
    numeric_columns = df.select_dtypes(include=np.number).columns
    outlier_count = {}

    for column in numeric_columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            outlier_count[column] = 0
            continue

        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr
        count = ((df[column] < lower_limit) | (df[column] > upper_limit)).sum()
        outlier_count[column] = int(count)

    result = pd.Series(outlier_count).sort_values(ascending=False)
    print("\n--- TOP POSSIBLE OUTLIERS ---")
    print(result.head(10))
    return result


def clean_data(df):
    """Perform simple and explainable cleaning steps."""
    print("\nCleaning dataset...")
    cleaned_df = df.copy()

    # Remove spaces from column names and text values.
    cleaned_df.columns = cleaned_df.columns.str.strip()
    text_columns = cleaned_df.select_dtypes(include="object").columns
    for column in text_columns:
        cleaned_df[column] = cleaned_df[column].apply(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    # Convert HandsetPrice from text to a number.
    # Values such as "Unknown" become missing values and are imputed later.
    if "HandsetPrice" in cleaned_df.columns:
        cleaned_df["HandsetPrice"] = pd.to_numeric(
            cleaned_df["HandsetPrice"].replace("Unknown", np.nan),
            errors="coerce"
        )

    # Remove exact duplicate rows.
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)

    # Remove rows only when the target is missing.
    cleaned_df = cleaned_df.dropna(subset=["Churn"])

    print("Cleaning completed.")
    print("Cleaned rows:", cleaned_df.shape[0])
    return cleaned_df


def save_cleaned_data(df, file_path):
    """Save cleaned data to a CSV file."""
    df.to_csv(file_path, index=False)
    print("Cleaned data saved to:", file_path)
