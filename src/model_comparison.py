
import os
import pandas as pd
import matplotlib.pyplot as plt

# Get project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "model_results.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Check whether model results exist
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        "model_results.csv not found. Run this first: "
        "python src/train_model.py"
    )

# Load model evaluation results
df = pd.read_csv(DATA_FILE)

print("\nColumns found:", df.columns.tolist())
print("\nModel results:")
print(df.to_string(index=False))

# Identify the model-name column
def normalize(text):
    return str(text).strip().lower().replace("_", "").replace("-", "").replace(" ", "")

columns = {normalize(col): col for col in df.columns}

model_col = next(
    (columns[key] for key in ["model", "algorithm", "classifier"]
     if key in columns),
    None
)

if model_col is None:
    raise ValueError(
        "Could not identify the model column. "
        "Send me the column names printed above."
    )

# Identify available evaluation metrics
metric_aliases = {
    "Accuracy": ["accuracy"],
    "Precision": ["precision"],
    "Recall": ["recall"],
    "F1 Score": ["f1score", "f1"],
    "ROC-AUC": ["rocauc", "auc"]
}

metrics = {}

for label, aliases in metric_aliases.items():
    for alias in aliases:
        if alias in columns:
            metrics[label] = columns[alias]
            break

if not metrics:
    raise ValueError(
        "No recognized evaluation metric columns were found. "
        "Send me the printed column names."
    )

# Convert metrics to numeric values
for col in metrics.values():
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Plot comparison using actual results
plot_data = df.set_index(model_col)[list(metrics.values())]
plot_data.columns = list(metrics.keys())

ax = plot_data.plot(
    kind="bar",
    figsize=(12, 6),
    width=0.8
)

ax.set_title("Comparison of Supervised Machine Learning Algorithms")
ax.set_xlabel("Algorithm")
ax.set_ylabel("Evaluation Score")
ax.set_ylim(0, 1.05)
ax.legend(title="Evaluation Metrics")
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.xticks(rotation=15, ha="right")
plt.tight_layout()

# Save graph
output_file = os.path.join(RESULTS_DIR, "model_comparison.png")
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nGraph saved successfully at:\n{output_file}")
