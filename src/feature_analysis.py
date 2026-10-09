
import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "supervised_customer_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        "supervised_customer_data.csv not found. "
        "Run python src/create_target.py first."
    )

df = pd.read_csv(DATA_FILE)

features = ["Recency", "Frequency", "Monetary"]
missing = [col for col in features if col not in df.columns]

if missing:
    raise ValueError(f"Missing expected columns: {missing}")

for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=features)

# Print descriptive statistics
print("\nDescriptive Statistics:")
print(df[features].describe().round(2))

# Create one histogram per feature
for feature in features:
    plt.figure(figsize=(8, 5))
    plt.hist(df[feature], bins=40, edgecolor="black")
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Number of Customers")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()

    output_file = os.path.join(
        OUTPUT_DIR, f"{feature.lower()}_distribution.png"
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_file}")

# Compare feature distributions by target class
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, feature in zip(axes, features):
    groups = [
        df.loc[df["Purchase"] == label, feature].dropna()
        for label in [0, 1]
    ]

    ax.boxplot(groups, tick_labels=["No Purchase", "Purchase"])
    ax.set_title(f"{feature} by Purchase Class")
    ax.set_ylabel(feature)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

plt.tight_layout()

output_file = os.path.join(OUTPUT_DIR, "feature_comparison.png")
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {output_file}")
print("\nFeature analysis completed successfully.")
