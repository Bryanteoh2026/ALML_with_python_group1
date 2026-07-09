# =========================================================
# MAIN FILE
# This file runs the whole machine learning workflow
# =========================================================
import pandas as pd

from cleaning import (
    load_data,
    check_basic_info,
    clean_data,
    check_missing_values,
    check_duplicates,
    check_outliers,
    save_cleaned_data
)

from preprocessing import (
    split_features_target,
    split_train_validation_test,
    create_preprocessor
)

from training import (
    create_models,
    train_models,
    tune_random_forest
)

from evaluation import (
    evaluate_models,
    evaluate_best_model,
    get_confusion_matrix
)

from visualization import (
    plot_target_distribution,
    plot_numeric_boxplots,
    plot_confusion_matrix,
    plot_feature_importance
)

def main():
    print("Loading dataset...")

    df = pd.read_csv("telco_customer.csv")

    print("Dataset loaded successfully.")
    print(df.head())
    print(df.shape)
    print(df.info())


if __name__ == "__main__":
    main()