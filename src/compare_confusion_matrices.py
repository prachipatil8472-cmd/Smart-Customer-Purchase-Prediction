
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "supervised_customer_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        "Dataset not found. Run python src/create_target.py first."
    )

df = pd.read_csv(DATA_FILE)

features = ["Recency", "Frequency", "Monetary"]
X = df[features]
y = df["Purchase"].astype(int)

if y.nunique() != 2:
    raise ValueError("Purchase must contain both classes 0 and 1.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

summary = []

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, (name, model) in zip(axes, models.items()):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Purchase", "Purchase"],
        yticklabels=["No Purchase", "Purchase"],
        ax=ax
    )

    ax.set_title(name)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probabilities)
    else:
        auc = float("nan")

    summary.append({
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
        "ROC-AUC": auc
    })

    print(f"\n{'=' * 50}")
    print(name)
    print("=" * 50)
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["No Purchase", "Purchase"],
        zero_division=0
    ))

plt.tight_layout()

matrix_path = os.path.join(
    OUTPUT_DIR, "all_model_confusion_matrices.png"
)
plt.savefig(matrix_path, dpi=300, bbox_inches="tight")
plt.close()

results = pd.DataFrame(summary)
results = results.sort_values("F1 Score", ascending=False)

results_path = os.path.join(
    OUTPUT_DIR, "all_model_evaluation.csv"
)
results.to_csv(results_path, index=False)

print("\nModel comparison:")
print(results.to_string(index=False))

print(f"\nConfusion matrices saved to: {matrix_path}")
print(f"Metrics saved to: {results_path}")
