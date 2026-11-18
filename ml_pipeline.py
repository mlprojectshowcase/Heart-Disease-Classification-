"""
ml_pipeline.py
--------------
All the machine-learning logic lives here, so that BOTH train_model.py (terminal)
and app.py (Streamlit website) use exactly the same code.

Steps:
  1. load_data()          -> read the CSV file
  2. build_preprocessor() -> scale numbers + convert text columns to numbers
  3. get_models()         -> the algorithms we compare
  4. train_and_evaluate() -> train, test, return accuracy / precision / recall / F1 / ROC-AUC
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent          # folder where this file lives
DATA_PATH = BASE_DIR / "data" / "heart_data.csv"
TARGET = "heart_disease"                            # 0 = no disease, 1 = disease
LABELS = ["No Disease", "Disease"]

NUMERIC = ["age", "resting_bp", "cholesterol", "max_heart_rate", "oldpeak_st_depression"]
CATEGORICAL = ["sex", "chest_pain_type", "fasting_blood_sugar_high", "resting_ecg",
               "exercise_induced_angina", "st_slope"]


def load_data(path=DATA_PATH):
    """Reads the CSV. If the file is missing (e.g. 'data' folder not uploaded to GitHub),
    the same dataset is regenerated automatically so the app never crashes."""
    if Path(path).exists():
        return pd.read_csv(path)
    from generate_dataset import make_dataframe
    return make_dataframe()


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=15),
    }


def train_and_evaluate(df):
    """Train all models on 80% of data, test on the remaining 20%."""
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    results, fitted = {}, {}
    for name, model in get_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        results[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),      # most important in medicine!
            "f1": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, proba),
            "cv_accuracy": cross_val_score(pipe, X_train, y_train, cv=5).mean(),
            "confusion": confusion_matrix(y_test, pred, labels=[0, 1]),
        }
        fitted[name] = pipe

    # Best model = highest F1 (balance of precision and recall)
    best = max(results, key=lambda n: results[n]["f1"])
    return fitted, results, best, (X_train, X_test, y_train, y_test)


def feature_importance(pipe):
    """Which inputs matter most? (tree-based models only)"""
    names = [n.split("__")[1] for n in pipe.named_steps["prep"].get_feature_names_out()]
    imp = pipe.named_steps["model"].feature_importances_
    return pd.Series(imp, index=names).sort_values(ascending=False)
