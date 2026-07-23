# =========================================================
# MODEL EVALUATION FILE
# =========================================================

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score


def make_predictions(model, X, threshold=0.50):
    """Convert churn probabilities into 0 or 1 predictions."""
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return predictions, probabilities


def evaluate_models(models, thresholds, X_validation, y_validation):
    """Compare models on validation data."""
    results = []

    for model_name, model in models.items():
        predictions, probabilities = make_predictions(
            model,
            X_validation,
            thresholds[model_name]
        )

        results.append({
            "Model": model_name,
            "Threshold": thresholds[model_name],
            "Accuracy": accuracy_score(y_validation, predictions),
            "Precision": precision_score(y_validation, predictions),
            "Recall": recall_score(y_validation, predictions),
            "F1_Score": f1_score(y_validation, predictions),
            "ROC_AUC": roc_auc_score(y_validation, probabilities)
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("F1_Score", ascending=False)
    return results_df


def choose_best_model(results_df, models):
    """Choose the model with the highest validation F1-score."""
    best_model_name = results_df.iloc[0]["Model"]
    best_model = models[best_model_name]
    return best_model_name, best_model


def evaluate_best_model(model, threshold, X_test, y_test):
    """Evaluate the final selected model on unseen test data."""
    predictions, probabilities = make_predictions(model, X_test, threshold)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1_Score": f1_score(y_test, predictions),
        "ROC_AUC": roc_auc_score(y_test, probabilities)
    }

    report = classification_report(
        y_test,
        predictions,
        target_names=["Stay", "Churn"]
    )

    return metrics, report


def get_confusion_matrix(model, X, y, threshold=0.50):
    """Return the confusion matrix for one model."""
    predictions, _ = make_predictions(model, X, threshold)
    return confusion_matrix(y, predictions)
