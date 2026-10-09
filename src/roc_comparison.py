
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score

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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
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

plt.figure(figsize=(9, 7))
auc_results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, probabilities)
    auc = roc_auc_score(y_test, probabilities)

    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    auc_results.append({
        "Model": name,
        "ROC-AUC": auc
    })

plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
plt.title("ROC Curve Comparison of Supervised ML Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()

output_file = os.path.join(OUTPUT_DIR, "roc_comparison.png")
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.close()

auc_df = pd.DataFrame(auc_results).sort_values(
    "ROC-AUC", ascending=False
)
auc_file = os.path.join(OUTPUT_DIR, "roc_auc_comparison.csv")
auc_df.to_csv(auc_file, index=False)

print("\nROC-AUC comparison:")
print(auc_df.to_string(index=False))
print(f"\nROC graph saved to: {output_file}")
print(f"AUC results saved to: {auc_file}")
