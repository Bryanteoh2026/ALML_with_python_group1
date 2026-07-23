# =========================================================
# MODEL TRAINING FILE
# =========================================================

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

from preprocessing import build_preprocessor


def get_models():
    """Create four simple classification models."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=25,
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(24,),
            max_iter=50,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42
        )
    }
    return models


def find_best_threshold(model, X_validation, y_validation):
    """Find a probability threshold that gives the best F1-score."""
    probabilities = model.predict_proba(X_validation)[:, 1]
    thresholds = np.arange(0.20, 0.71, 0.02)

    best_threshold = 0.50
    best_f1 = 0

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        score = f1_score(y_validation, predictions)

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return round(best_threshold, 2)


def train_models(X_train, y_train, X_validation, y_validation):
    """Train every model and find its validation threshold."""
    trained_models = {}
    thresholds = {}

    for model_name, model in get_models().items():
        print("Training:", model_name)

        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        trained_models[model_name] = pipeline
        thresholds[model_name] = find_best_threshold(
            pipeline,
            X_validation,
            y_validation
        )

    return trained_models, thresholds


def save_model(model, file_path):
    """Save the selected model."""
    joblib.dump(model, file_path)
    print("Best model saved to:", file_path)
