# =========================================================
# DATA PREPROCESSING FILE
# =========================================================

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline


def split_features_target(df, target_column):
    pass


def split_train_validation_test(X, y, random_state):
    pass


def create_preprocessor(X_train, numeric_features):
    pass