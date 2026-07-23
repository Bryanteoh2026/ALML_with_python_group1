# =========================================================
# DATA CLEANING FILE
# =========================================================

import pandas as pd
import numpy as np

def load_data():
    df = pd.read_csv("telco_customer.csv")
    df.head()
    df.info()
    df.isnull().sum().sort_values(ascending=False)
    return df


def check_basic_info(df):
    pass


def clean_data(df):
    pass


def check_missing_values(df):
    pass


def check_duplicates(df):
    pass


def check_outliers(df):
    pass


def save_cleaned_data(df, file_path):
    pass