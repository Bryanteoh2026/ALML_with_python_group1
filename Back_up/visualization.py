# =========================================================
# DATA VISUALISATION FILE
# =========================================================

import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve

from evaluation import get_confusion_matrix


sns.set_theme(style="whitegrid")


def plot_churn_distribution(df, output_folder):
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="Churn", color="steelblue")
    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "01_churn_distribution.png"), dpi=200)
    plt.close()


def plot_missing_values(df, output_folder):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing.values, y=missing.index, color="steelblue")
    plt.title("Top Columns with Missing Values")
    plt.xlabel("Number of Missing Values")
    plt.ylabel("Column")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "02_missing_values.png"), dpi=200)
    plt.close()


def plot_numeric_distributions(df, output_folder):
    columns = [
        "MonthlyRevenue",
        "MonthlyMinutes",
        "OverageMinutes",
        "DroppedCalls",
        "MonthsInService",
        "CurrentEquipmentDays"
    ]

    melted = df[columns].melt(var_name="Feature", value_name="Value")

    grid = sns.FacetGrid(
        melted,
        col="Feature",
        col_wrap=3,
        sharex=False,
        sharey=False,
        height=3.2
    )
    grid.map_dataframe(sns.histplot, x="Value", bins=30, color="steelblue")
    grid.fig.suptitle("Selected Numeric Feature Distributions", y=1.03)
    grid.tight_layout()
    grid.savefig(os.path.join(output_folder, "03_numeric_distributions.png"), dpi=200)
    plt.close("all")


def plot_correlation_heatmap(df, output_folder):
    columns = [
        "MonthlyRevenue",
        "MonthlyMinutes",
        "TotalRecurringCharge",
        "OverageMinutes",
        "RoamingCalls",
        "DroppedCalls",
        "BlockedCalls",
        "CustomerCareCalls",
        "MonthsInService",
        "CurrentEquipmentDays",
        "RetentionCalls",
        "IncomeGroup"
    ]

    correlation = df[columns].corr()

    plt.figure(figsize=(11, 8))
    sns.heatmap(correlation, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "04_correlation_heatmap.png"), dpi=200)
    plt.close()


def plot_model_comparison(results_df, output_folder):
    score_columns = ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]
    plot_data = results_df.melt(
        id_vars="Model",
        value_vars=score_columns,
        var_name="Metric",
        value_name="Score"
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_data, x="Model", y="Score", hue="Metric")
    plt.title("Validation Performance Comparison")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "05_model_comparison.png"), dpi=200)
    plt.close()


def plot_confusion_matrices(models, thresholds, X_test, y_test, output_folder):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    for axis, (model_name, model) in zip(axes, models.items()):
        matrix = get_confusion_matrix(
            model,
            X_test,
            y_test,
            thresholds[model_name]
        )

        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axis
        )
        axis.set_title(model_name)
        axis.set_xlabel("Predicted Label")
        axis.set_ylabel("Actual Label")
        axis.set_xticklabels(["Stay", "Churn"])
        axis.set_yticklabels(["Stay", "Churn"], rotation=0)

    fig.suptitle("Confusion Matrices on Test Data", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "06_confusion_matrices.png"), dpi=200)
    plt.close()


def plot_roc_curves(models, X_test, y_test, output_folder):
    plt.figure(figsize=(8, 6))

    for model_name, model in models.items():
        probabilities = model.predict_proba(X_test)[:, 1]
        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_test,
            probabilities
        )
        area = auc(false_positive_rate, true_positive_rate)
        plt.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{model_name} (AUC={area:.3f})"
        )

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "07_roc_curves.png"), dpi=200)
    plt.close()


def plot_feature_importance(random_forest_pipeline, output_folder):
    preprocessor = random_forest_pipeline.named_steps["preprocessor"]
    model = random_forest_pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance = importance.sort_values(ascending=False).head(15)
    importance.index = (
        importance.index
        .str.replace("number__", "", regex=False)
        .str.replace("category__", "", regex=False)
    )

    plt.figure(figsize=(10, 7))
    sns.barplot(x=importance.values, y=importance.index, color="steelblue")
    plt.title("Top 15 Random Forest Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "08_feature_importance.png"), dpi=200)
    plt.close()


def plot_neural_network_loss(neural_network_pipeline, output_folder):
    model = neural_network_pipeline.named_steps["model"]
    loss_values = model.loss_curve_

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_values) + 1), loss_values, marker="o")
    plt.title("Neural Network Training Loss Curve")
    plt.xlabel("Training Iteration")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "09_neural_network_loss_curve.png"), dpi=200)
    plt.close()
