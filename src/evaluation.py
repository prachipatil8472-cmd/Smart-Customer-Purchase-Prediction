import pandas as pd
import numpy as np
import os
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report
)


# =========================================
# PROJECT PATH
# =========================================

project_folder = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

model_path = os.path.join(
    project_folder,
    "models",
    "best_model.pkl"
)

test_path = os.path.join(
    project_folder,
    "data",
    "test_predictions.csv"
)


# =========================================
# LOAD MODEL AND TEST DATA
# =========================================

model = joblib.load(model_path)

test_data = pd.read_csv(test_path)

print("Model loaded successfully!")

print("\nTest data:")
print(test_data.head())


# =========================================
# FEATURES
# =========================================

X_test = test_data[
    [
        "Recency",
        "Frequency",
        "Monetary"
    ]
]

y_test = test_data["Actual"]


# =========================================
# PREDICTIONS
# =========================================

y_pred = model.predict(
    X_test
)

y_prob = model.predict_proba(
    X_test
)[:, 1]


# =========================================
# MODEL METRICS
# =========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)


# =========================================
# DISPLAY METRICS
# =========================================

print("\n=========================================")
print("MODEL EVALUATION")
print("=========================================")

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)


# =========================================
# CLASSIFICATION REPORT
# =========================================

print("\n=========================================")
print("CLASSIFICATION REPORT")
print("=========================================")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# =========================================
# CONFUSION MATRIX
# =========================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(
    figsize=(7, 5)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "Actual Label"
)

plt.title(
    "Confusion Matrix"
)

plt.tight_layout()

confusion_path = os.path.join(
    project_folder,
    "confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=300
)

plt.show()

plt.close()


# =========================================
# ROC CURVE
# =========================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

plt.figure(
    figsize=(7, 5)
)

plt.plot(
    fpr,
    tpr,
    label=f"ROC Curve (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve"
)

plt.legend()

plt.tight_layout()

roc_path = os.path.join(
    project_folder,
    "roc_curve.png"
)

plt.savefig(
    roc_path,
    dpi=300
)

plt.show()

plt.close()


# =========================================
# FEATURE IMPORTANCE
# =========================================

print("\n=========================================")
print("FEATURE IMPORTANCE")
print("=========================================")

feature_names = [
    "Recency",
    "Frequency",
    "Monetary"
]


# Direct feature importance
if hasattr(
    model,
    "feature_importances_"
):

    importances = model.feature_importances_


# Pipeline containing Random Forest
elif hasattr(
    model,
    "named_steps"
) and hasattr(
    model.named_steps.get("model"),
    "feature_importances_"
):

    importances = (
        model
        .named_steps["model"]
        .feature_importances_
    )


else:

    importances = None


if importances is not None:

    feature_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    feature_df = feature_df.sort_values(
        by="Importance",
        ascending=False
    )

    print(feature_df)

    plt.figure(
        figsize=(7, 5)
    )

    plt.bar(
        feature_df["Feature"],
        feature_df["Importance"]
    )

    plt.xlabel(
        "Feature"
    )

    plt.ylabel(
        "Importance"
    )

    plt.title(
        "Feature Importance"
    )

    plt.tight_layout()

    feature_path = os.path.join(
        project_folder,
        "feature_importance.png"
    )

    plt.savefig(
        feature_path,
        dpi=300
    )

    plt.show()

    plt.close()

else:

    print(
        "Feature importance is not available "
        "for the selected model."
    )


# =========================================
# SAVE EVALUATION RESULTS
# =========================================

evaluation_results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]
})

evaluation_path = os.path.join(
    project_folder,
    "data",
    "evaluation_results.csv"
)

evaluation_results.to_csv(
    evaluation_path,
    index=False
)


# =========================================
# FINAL MESSAGE
# =========================================

print("\n=========================================")
print("EVALUATION COMPLETED SUCCESSFULLY")
print("=========================================")

print("\nFiles created:")

print("confusion_matrix.png")
print("roc_curve.png")
print("feature_importance.png")
print("evaluation_results.csv")