
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load processed customer dataset
df = pd.read_csv("data/supervised_customer_data.csv")

features = ["Recency", "Frequency", "Monetary"]
target = "Purchase"

X = df[features]
y = df[target]

# Split data before training to avoid test-data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "AdaBoost": (
        AdaBoostClassifier(random_state=42),
        {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.1, 1.0]
        }
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=42),
        {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [1, 2, 3]
        }
    )
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []
best_models = {}

for name, (estimator, params) in models.items():
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", estimator)
    ])

    search = GridSearchCV(
        pipeline,
        param_grid=params,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    model = search.best_estimator_
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "Best Parameters": search.best_params_
    })

    best_models[name] = model

# Display actual results
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Save comparison
results_df.to_csv("data/ensemble_comparison.csv", index=False)

# Save the best model by test ROC-AUC for evaluation purposes
best_name = max(results, key=lambda row: row["ROC-AUC"])["Model"]
import joblib
joblib.dump(best_models[best_name], "models/best_ensemble_model.pkl")

print(f"\nBest ensemble model: {best_name}")
print("Saved to models/best_ensemble_model.pkl")
