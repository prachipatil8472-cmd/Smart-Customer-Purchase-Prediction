
import os
import time
import joblib
import numpy as np
import pandas as pd

from model_utils import log_transform

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

DATA_PATH = "data/supervised_customer_data.csv"
FEATURES = ["Recency", "Frequency", "Monetary"]
TARGET = "Purchase"

print("Loading dataset...", flush=True)

df = pd.read_csv(DATA_PATH)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=[TARGET]).copy()

X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(df[TARGET], errors="raise").astype(int)

if y.nunique() != 2 or not set(y.unique()).issubset({0, 1}):
    raise ValueError("Purchase must contain only binary 0 and 1 labels.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

experiments = {
    "Random Forest": (
        RandomForestClassifier(random_state=42, n_jobs=1),
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [8, 15],
            "model__min_samples_leaf": [3, 5],
            "model__class_weight": ["balanced"]
        }
    ),
    "AdaBoost": (
        AdaBoostClassifier(random_state=42),
        {
            "model__n_estimators": [50, 100],
            "model__learning_rate": [0.1, 0.5]
        }
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=42),
        {
            "model__n_estimators": [100, 150],
            "model__learning_rate": [0.03, 0.1],
            "model__max_depth": [1, 2]
        }
    )
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []
best_models = {}
threshold_results = []

for name, (estimator, params) in experiments.items():
    print(f"\nTraining: {name}", flush=True)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(log_transform, validate=False)),
        ("model", estimator)
    ])

    search = GridSearchCV(
        pipeline,
        param_grid=params,
        scoring="accuracy",
        cv=cv,
        n_jobs=1,
        refit=True,
        verbose=1
    )

    start = time.time()
    search.fit(X_train, y_train)
    model = search.best_estimator_
    elapsed = time.time() - start

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    result = {
        "Model": name,
        "CV Accuracy": search.best_score_,
        "Test Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "Best Parameters": str(search.best_params_)
    }

    results.append(result)
    best_models[name] = model

    print(f"Finished in {elapsed:.1f} seconds.", flush=True)
    print("Best parameters:", search.best_params_)
    print(classification_report(y_test, predictions, zero_division=0))

    for threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        pred = (probabilities >= threshold).astype(int)
        threshold_results.append({
            "Model": name,
            "Threshold": threshold,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0)
        })

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

results_df = pd.DataFrame(results)
results_df.to_csv("data/improved_model_results.csv", index=False)
pd.DataFrame(threshold_results).to_csv(
    "data/threshold_comparison.csv", index=False
)

print("\nFINAL MODEL COMPARISON")
print(results_df[
    ["Model", "CV Accuracy", "Test Accuracy",
     "Precision", "Recall", "F1 Score", "ROC-AUC"]
].round(4).to_string(index=False))

winner = max(results, key=lambda row: row["CV Accuracy"])
best_name = winner["Model"]

joblib.dump(best_models[best_name], "models/best_tuned_model.pkl")

print(f"\nSelected by cross-validation: {best_name}")
print("Saved: models/best_tuned_model.pkl")
