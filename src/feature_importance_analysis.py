
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance

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
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}

for name, model in models.items():
    model.fit(X_train, y_train)

    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="f1",
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": result.importances_mean,
        "Std": result.importances_std
    }).sort_values("Importance", ascending=True)

    print(f"\n{name} - Permutation Importance")
    print(importance_df.sort_values(
        "Importance", ascending=False
    ).to_string(index=False))

    ax = importance_df.plot(
        x="Feature",
        y="Importance",
        kind="barh",
        xerr="Std",
        legend=False,
        figsize=(8, 5),
        capsize=4
    )

    ax.set_title(f"Feature Importance - {name}")
    ax.set_xlabel("Decrease in F1 Score When Feature Is Shuffled")
    ax.set_ylabel("Feature")
    ax.axvline(0, linewidth=1)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    plt.tight_layout()

    filename = name.lower().replace(" ", "_") + "_importance.png"
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    importance_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            name.lower().replace(" ", "_") + "_importance.csv"
        ),
        index=False
    )

    print(f"Graph saved: {output_path}")

print("\nFeature importance analysis completed.")
