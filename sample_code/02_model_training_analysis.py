"""Step 2: Train, validate, compare and test churn classification models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "telco_customer_cleaned.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"


sns.set_theme(style="whitegrid")


def calculate_metrics(y_true, y_pred, y_probability) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_probability),
    }


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def save_eda_plots(df: pd.DataFrame) -> None:
    churn_order = ["No", "Yes"]

    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=df, x="Churn", order=churn_order, color="tab:blue")
    ax.set_title("Customer Churn Class Distribution")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Number of Customers")
    for container in ax.containers:
        ax.bar_label(container)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "01_churn_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    churn_by_contract = pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100
    churn_by_contract = churn_by_contract.reindex(columns=churn_order)
    churn_by_contract.plot(kind="bar", stacked=True, ax=plt.gca())
    plt.title("Churn Percentage by Contract Type")
    plt.xlabel("Contract Type")
    plt.ylabel("Percentage of Customers")
    plt.xticks(rotation=0)
    plt.legend(title="Churn")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "02_churn_by_contract.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.histplot(data=df, x="tenure", hue="Churn", bins=24, multiple="layer", kde=True)
    plt.title("Tenure Distribution by Churn Status")
    plt.xlabel("Tenure (Months)")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "03_tenure_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", order=churn_order, color="tab:blue")
    plt.title("Monthly Charges by Churn Status")
    plt.xlabel("Churn")
    plt.ylabel("Monthly Charges")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "04_monthly_charges_boxplot.png", dpi=180)
    plt.close()

    numeric_for_correlation = df[
        ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "NumOptionalServices"]
    ].copy()
    numeric_for_correlation["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_for_correlation.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap for Numeric Features")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "05_correlation_heatmap.png", dpi=180)
    plt.close()


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    save_eda_plots(df)

    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    # 60% training, 20% validation, 20% testing.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        stratify=y_train_val,
        random_state=RANDOM_STATE,
    )

    split_summary = pd.DataFrame(
        {
            "Split": ["Training", "Validation", "Testing"],
            "Rows": [len(X_train), len(X_val), len(X_test)],
            "Churn_Rate": [y_train.mean(), y_val.mean(), y_test.mean()],
        }
    )
    split_summary.to_csv(OUTPUT_DIR / "data_split_summary.csv", index=False)

    preprocessor = make_preprocessor(X_train)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(32, 16),
            max_iter=220,
            early_stopping=True,
            validation_fraction=0.20,
            n_iter_no_change=15,
            random_state=RANDOM_STATE,
        ),
    }

    fitted_models = {}
    comparison_rows = []
    validation_predictions = {}
    validation_probabilities = {}

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        fitted_models[model_name] = pipeline

        train_pred = pipeline.predict(X_train)
        train_prob = pipeline.predict_proba(X_train)[:, 1]
        val_pred = pipeline.predict(X_val)
        val_prob = pipeline.predict_proba(X_val)[:, 1]

        train_metrics = calculate_metrics(y_train, train_pred, train_prob)
        val_metrics = calculate_metrics(y_val, val_pred, val_prob)

        comparison_rows.append(
            {
                "Model": model_name,
                **{f"Train_{key}": value for key, value in train_metrics.items()},
                **{f"Validation_{key}": value for key, value in val_metrics.items()},
                "F1_Gap": train_metrics["F1"] - val_metrics["F1"],
            }
        )
        validation_predictions[model_name] = val_pred
        validation_probabilities[model_name] = val_prob

    # Hyperparameter tuning on training data only.
    tuned_rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            (
                "model",
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )
    parameter_grid = {
        "model__n_estimators": [120, 200],
        "model__max_depth": [8, 12],
        "model__min_samples_leaf": [3],
        "model__max_features": ["sqrt"],
    }
    cross_validation = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    grid_search = GridSearchCV(
        estimator=tuned_rf_pipeline,
        param_grid=parameter_grid,
        scoring="f1",
        cv=cross_validation,
        n_jobs=1,
        refit=True,
    )
    grid_search.fit(X_train, y_train)
    tuned_rf = grid_search.best_estimator_
    fitted_models["Tuned Random Forest"] = tuned_rf

    tuned_train_pred = tuned_rf.predict(X_train)
    tuned_train_prob = tuned_rf.predict_proba(X_train)[:, 1]
    tuned_val_pred = tuned_rf.predict(X_val)
    tuned_val_prob = tuned_rf.predict_proba(X_val)[:, 1]
    tuned_train_metrics = calculate_metrics(y_train, tuned_train_pred, tuned_train_prob)
    tuned_val_metrics = calculate_metrics(y_val, tuned_val_pred, tuned_val_prob)

    comparison_rows.append(
        {
            "Model": "Tuned Random Forest",
            **{f"Train_{key}": value for key, value in tuned_train_metrics.items()},
            **{f"Validation_{key}": value for key, value in tuned_val_metrics.items()},
            "F1_Gap": tuned_train_metrics["F1"] - tuned_val_metrics["F1"],
        }
    )
    validation_predictions["Tuned Random Forest"] = tuned_val_pred
    validation_probabilities["Tuned Random Forest"] = tuned_val_prob

    comparison = pd.DataFrame(comparison_rows).sort_values(
        "Validation_F1", ascending=False
    )
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    tuning_results = {
        "best_parameters": grid_search.best_params_,
        "best_cross_validation_f1": grid_search.best_score_,
    }
    (OUTPUT_DIR / "random_forest_tuning_results.json").write_text(
        json.dumps(tuning_results, indent=2), encoding="utf-8"
    )

    # Model comparison chart.
    plot_data = comparison.melt(
        id_vars="Model",
        value_vars=["Validation_Accuracy", "Validation_Precision", "Validation_Recall", "Validation_F1"],
        var_name="Metric",
        value_name="Score",
    )
    plot_data["Metric"] = plot_data["Metric"].str.replace("Validation_", "", regex=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_data, x="Model", y="Score", hue="Metric")
    plt.title("Validation Performance Comparison")
    plt.ylim(0, 1)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "06_model_comparison.png", dpi=180)
    plt.close()

    # Training-vs-validation F1 for overfitting analysis.
    plt.figure(figsize=(11, 5))
    f1_plot = comparison.melt(
        id_vars="Model",
        value_vars=["Train_F1", "Validation_F1"],
        var_name="Dataset",
        value_name="F1 Score",
    )
    f1_plot["Dataset"] = f1_plot["Dataset"].str.replace("_F1", "", regex=False)
    sns.barplot(data=f1_plot, x="Model", y="F1 Score", hue="Dataset")
    plt.title("Training vs Validation F1 (Overfitting Check)")
    plt.ylim(0, 1)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "07_overfitting_check.png", dpi=180)
    plt.close()

    # Validation confusion matrices for every model.
    model_names = list(validation_predictions.keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for index, model_name in enumerate(model_names):
        matrix = confusion_matrix(y_val, validation_predictions[model_name])
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axes[index],
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
        )
        axes[index].set_title(model_name)
        axes[index].set_xlabel("Predicted")
        axes[index].set_ylabel("Actual")
    for empty_axis in axes[len(model_names):]:
        empty_axis.axis("off")
    fig.suptitle("Validation Confusion Matrices", fontsize=16)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "08_validation_confusion_matrices.png", dpi=180)
    plt.close()

    # ROC curves.
    plt.figure(figsize=(9, 7))
    for model_name, probability in validation_probabilities.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(y_val, probability)
        auc_score = roc_auc_score(y_val, probability)
        plt.plot(false_positive_rate, true_positive_rate, label=f"{model_name} (AUC={auc_score:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
    plt.title("Validation ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "09_roc_curves.png", dpi=180)
    plt.close()

    # Precision-recall curves.
    plt.figure(figsize=(9, 7))
    for model_name, probability in validation_probabilities.items():
        precision_values, recall_values, _ = precision_recall_curve(y_val, probability)
        plt.plot(recall_values, precision_values, label=model_name)
    plt.title("Validation Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "10_precision_recall_curves.png", dpi=180)
    plt.close()

    # Neural-network loss curve required by the rubric.
    mlp_model = fitted_models["Neural Network"].named_steps["model"]
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(mlp_model.loss_curve_) + 1), mlp_model.loss_curve_)
    plt.title("Neural Network Training Loss Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "11_neural_network_loss_curve.png", dpi=180)
    plt.close()

    if hasattr(mlp_model, "validation_scores_") and mlp_model.validation_scores_ is not None:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(mlp_model.validation_scores_) + 1), mlp_model.validation_scores_)
        plt.title("Neural Network Internal Validation Score")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "12_neural_network_validation_curve.png", dpi=180)
        plt.close()

    # Select the model using validation F1 only.
    best_model_name = comparison.iloc[0]["Model"]
    selected_pipeline = clone(fitted_models[best_model_name])

    # Refit the selected model using training + validation data.
    selected_pipeline.fit(X_train_val, y_train_val)

    # Final evaluation on the untouched test data.
    test_predictions = selected_pipeline.predict(X_test)
    test_probabilities = selected_pipeline.predict_proba(X_test)[:, 1]
    test_metrics = calculate_metrics(y_test, test_predictions, test_probabilities)

    test_result_table = pd.DataFrame(
        [{"Selected_Model": best_model_name, **test_metrics}]
    )
    test_result_table.to_csv(OUTPUT_DIR / "final_test_metrics.csv", index=False)

    report = classification_report(
        y_test,
        test_predictions,
        target_names=["No Churn", "Churn"],
        digits=4,
    )
    (OUTPUT_DIR / "final_classification_report.txt").write_text(
        f"Selected model: {best_model_name}\n\n{report}", encoding="utf-8"
    )

    final_matrix = confusion_matrix(y_test, test_predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        final_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
    )
    plt.title(f"Final Test Confusion Matrix - {best_model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "13_final_test_confusion_matrix.png", dpi=180)
    plt.close()

    # Model-agnostic feature importance using the untouched test set.
    importance = permutation_importance(
        selected_pipeline,
        X_test,
        y_test,
        scoring="f1",
        n_repeats=4,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    feature_importance = pd.DataFrame(
        {
            "Feature": X_test.columns,
            "Importance": importance.importances_mean,
            "Standard_Deviation": importance.importances_std,
        }
    ).sort_values("Importance", ascending=False)
    feature_importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    top_features = feature_importance.head(12).sort_values("Importance")
    plt.figure(figsize=(9, 6))
    sns.barplot(data=top_features, x="Importance", y="Feature", orient="h", color="tab:blue")
    plt.title(f"Top Feature Importance - {best_model_name}")
    plt.xlabel("Decrease in F1 When Feature Is Shuffled")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "14_feature_importance.png", dpi=180)
    plt.close()

    # Save deployment model and website choices.
    joblib.dump(selected_pipeline, MODEL_DIR / "best_churn_model.joblib")

    categorical_options = {
        column: sorted(df[column].dropna().astype(str).unique().tolist())
        for column in X.select_dtypes(exclude=np.number).columns
        if column != "TenureGroup"
    }
    model_metadata = {
        "selected_model": best_model_name,
        "test_metrics": test_metrics,
        "classification_threshold": 0.50,
        "categorical_options": categorical_options,
        "numeric_ranges": {
            "tenure": [int(df["tenure"].min()), int(df["tenure"].max())],
            "MonthlyCharges": [float(df["MonthlyCharges"].min()), float(df["MonthlyCharges"].max())],
            "TotalCharges": [float(df["TotalCharges"].min()), float(df["TotalCharges"].max())],
        },
    }
    (MODEL_DIR / "model_metadata.json").write_text(
        json.dumps(model_metadata, indent=2), encoding="utf-8"
    )

    summary = f"""MODEL TRAINING SUMMARY
======================
Selected model: {best_model_name}
Selection metric: Highest validation F1-score
Training rows: {len(X_train)}
Validation rows: {len(X_val)}
Testing rows: {len(X_test)}

Final test metrics
------------------
Accuracy:  {test_metrics['Accuracy']:.4f}
Precision: {test_metrics['Precision']:.4f}
Recall:    {test_metrics['Recall']:.4f}
F1-score:  {test_metrics['F1']:.4f}
ROC-AUC:   {test_metrics['ROC_AUC']:.4f}

Best Random Forest parameters
-----------------------------
{json.dumps(grid_search.best_params_, indent=2)}
"""
    (OUTPUT_DIR / "model_training_summary.txt").write_text(summary, encoding="utf-8")

    print(summary)
    print(f"Model saved to: {MODEL_DIR / 'best_churn_model.joblib'}")


if __name__ == "__main__":
    main()
