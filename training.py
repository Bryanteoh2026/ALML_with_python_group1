# =========================================================
# MODEL TRAINING FILE
# =========================================================

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV


def create_models(preprocessor, random_state):
    pass


def train_models(models, X_train, y_train):
    pass


def tune_random_forest(model, X_train, y_train):
    pass