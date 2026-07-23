# =========================================================
# MAIN PROJECT FILE
# =========================================================

import json
import os
import pandas as pd

from cleaning import check_basic_info
from cleaning import check_duplicates
from cleaning import check_missing_values
from cleaning import check_outliers
from cleaning import clean_data
from cleaning import load_data
from cleaning import save_cleaned_data
from evaluation import choose_best_model
from evaluation import evaluate_best_model
from evaluation import evaluate_models
from preprocessing import FEATURE_COLUMNS
from preprocessing import prepare_features_and_target
from preprocessing import split_data
from training import save_model
from training import train_models
from visualization import plot_churn_distribution
from visualization import plot_confusion_matrices
from visualization import plot_correlation_heatmap
from visualization import plot_feature_importance
from visualization import plot_missing_values
from visualization import plot_model_comparison
from visualization import plot_neural_network_loss
from visualization import plot_numeric_distributions
from visualization import plot_roc_curves


BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_FOLDER, "data", "cell2celltrain.csv")
CLEANED_FILE = os.path.join(BASE_FOLDER, "data", "cleaned_cell2cell.csv")
MODEL_FOLDER = os.path.join(BASE_FOLDER, "models")
OUTPUT_FOLDER = os.path.join(BASE_FOLDER, "outputs")


def create_website_information(X_train, best_model_name, threshold):
    """Save defaults and category choices needed by the website."""
    defaults = {}
    category_options = {}

    for column in FEATURE_COLUMNS:
        if pd.api.types.is_numeric_dtype(X_train[column]):
            defaults[column] = float(X_train[column].median())
        else:
            mode_value = X_train[column].mode(dropna=True)
            defaults[column] = str(mode_value.iloc[0]) if len(mode_value) > 0 else "Unknown"
            category_options[column] = sorted(
                X_train[column].dropna().astype(str).unique().tolist()
            )

    information = {
        "best_model_name": best_model_name,
        "threshold": threshold,
        "feature_columns": FEATURE_COLUMNS,
        "defaults": defaults,
        "category_options": category_options
    }

    info_file = os.path.join(MODEL_FOLDER, "model_information.json")
    with open(info_file, "w", encoding="utf-8") as file:
        json.dump(information, file, indent=4)


def main():
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 1. Load, inspect and clean the data.
    df = load_data(DATA_FILE)
    check_basic_info(df)
    check_missing_values(df)
    check_duplicates(df)
    check_outliers(df)

    cleaned_df = clean_data(df)
    save_cleaned_data(cleaned_df, CLEANED_FILE)

    # 2. Prepare train, validation and test data.
    X, y = prepare_features_and_target(cleaned_df)
    split_result = split_data(X, y)
    X_train, X_validation, X_test, y_train, y_validation, y_test = split_result

    print("\nTraining rows:", len(X_train))
    print("Validation rows:", len(X_validation))
    print("Testing rows:", len(X_test))

    # 3. Train and compare models.
    models, thresholds = train_models(
        X_train,
        y_train,
        X_validation,
        y_validation
    )

    validation_results = evaluate_models(
        models,
        thresholds,
        X_validation,
        y_validation
    )

    validation_results.to_csv(
        os.path.join(OUTPUT_FOLDER, "validation_model_results.csv"),
        index=False
    )

    print("\n--- VALIDATION MODEL COMPARISON ---")
    print(validation_results.round(4).to_string(index=False))

    best_model_name, best_model = choose_best_model(validation_results, models)
    best_threshold = thresholds[best_model_name]

    # 4. Final test evaluation.
    test_metrics, report = evaluate_best_model(
        best_model,
        best_threshold,
        X_test,
        y_test
    )

    test_results = pd.DataFrame([test_metrics])
    test_results.insert(0, "Model", best_model_name)
    test_results.insert(1, "Threshold", best_threshold)
    test_results.to_csv(
        os.path.join(OUTPUT_FOLDER, "best_model_test_results.csv"),
        index=False
    )

    with open(
        os.path.join(OUTPUT_FOLDER, "classification_report.txt"),
        "w",
        encoding="utf-8"
    ) as file:
        file.write("Best model: " + best_model_name + "\n")
        file.write("Threshold: " + str(best_threshold) + "\n\n")
        file.write(report)

    print("\nBest model:", best_model_name)
    print("Best threshold:", best_threshold)
    print("\n--- FINAL TEST METRICS ---")
    for metric_name, metric_value in test_metrics.items():
        print(metric_name + ":", round(metric_value, 4))
    print("\n--- CLASSIFICATION REPORT ---")
    print(report)

    # 5. Save model and website information.
    save_model(best_model, os.path.join(MODEL_FOLDER, "best_model.joblib"))
    create_website_information(X_train, best_model_name, best_threshold)

    # 6. Create visualisations.
    print("\nCreating visualisations...")
    plot_churn_distribution(cleaned_df, OUTPUT_FOLDER)
    plot_missing_values(df, OUTPUT_FOLDER)
    plot_numeric_distributions(cleaned_df, OUTPUT_FOLDER)
    plot_correlation_heatmap(cleaned_df, OUTPUT_FOLDER)
    plot_model_comparison(validation_results, OUTPUT_FOLDER)
    plot_confusion_matrices(models, thresholds, X_test, y_test, OUTPUT_FOLDER)
    plot_roc_curves(models, X_test, y_test, OUTPUT_FOLDER)
    plot_feature_importance(models["Random Forest"], OUTPUT_FOLDER)
    plot_neural_network_loss(models["Neural Network"], OUTPUT_FOLDER)

    print("All outputs saved in:", OUTPUT_FOLDER)
    print("\nNext step: run 'python app.py' to open the website.")


if __name__ == "__main__":
    main()
