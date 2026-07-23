# =========================================================
# DATA PREPROCESSING FILE
# =========================================================

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# These features are understandable and useful for the website.
FEATURE_COLUMNS = [
    "MonthlyRevenue",
    "MonthlyMinutes",
    "TotalRecurringCharge",
    "OverageMinutes",
    "RoamingCalls",
    "PercChangeMinutes",
    "PercChangeRevenues",
    "DroppedCalls",
    "BlockedCalls",
    "UnansweredCalls",
    "CustomerCareCalls",
    "MonthsInService",
    "UniqueSubs",
    "ActiveSubs",
    "Handsets",
    "CurrentEquipmentDays",
    "AgeHH1",
    "RetentionCalls",
    "RetentionOffersAccepted",
    "IncomeGroup",
    "HandsetPrice",
    "HandsetRefurbished",
    "HandsetWebCapable",
    "Homeownership",
    "MadeCallToRetentionTeam",
    "CreditRating",
    "PrizmCode",
    "Occupation",
    "MaritalStatus"
]


def prepare_features_and_target(df):
    """Create X features and y target."""
    X = df[FEATURE_COLUMNS].copy()
    y = df["Churn"].map({"No": 0, "Yes": 1})
    return X, y


def split_data(X, y):
    """Split data into 70% training, 15% validation and 15% testing."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def get_column_groups(X):
    """Separate numeric columns and categorical columns."""
    numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=np.number).columns.tolist()
    return numeric_columns, categorical_columns


def build_preprocessor(X):
    """Create preprocessing steps for numbers and categories."""
    numeric_columns, categorical_columns = get_column_groups(X)

    numeric_steps = Pipeline(steps=[
        ("fill_missing", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])

    categorical_steps = Pipeline(steps=[
        ("fill_missing", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("number", numeric_steps, numeric_columns),
        ("category", categorical_steps, categorical_columns)
    ])

    return preprocessor
