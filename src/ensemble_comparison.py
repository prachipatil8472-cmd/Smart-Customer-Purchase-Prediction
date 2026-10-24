"""Compare AdaBoost and Gradient Boosting on the existing supervised RFM data."""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

FEATURES = ["Recency", "Frequency", "Monetary"]
TARGET = "Purchase"


def compare_ensembles(dataset_path, output_path):
    df = pd.read_csv(dataset_path)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    df = df[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce").dropna()
    X, y = df[FEATURES], df[TARGET].astype(int)
    if y.nunique() != 2:
        raise ValueError("Purchase target must contain both 0 and 1 classes.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    models = {
        "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, random_state=42),
    }
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        row = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, proba),
        }
        rows.append(row)
        print(f"\n{name}\n", classification_report(y_test, pred, zero_division=0))
    result = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base, "data", "supervised_customer_data.csv")
    output_path = os.path.join(base, "results", "ensemble_comparison.csv")
    print(compare_ensembles(data_path, output_path).to_string(index=False))
