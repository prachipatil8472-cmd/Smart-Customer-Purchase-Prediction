import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# =========================================
# PROJECT PATH
# =========================================

project_folder = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

data_path = os.path.join(
    project_folder,
    "data",
    "supervised_customer_data.csv"
)


# =========================================
# LOAD DATA
# =========================================

df = pd.read_csv(data_path)

print("Supervised dataset loaded successfully!")

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())


# =========================================
# CHECK TARGET
# =========================================

print("\nTarget distribution:")
print(df["Purchase"].value_counts())

print("\nTarget percentage:")
print(
    df["Purchase"].value_counts(normalize=True) * 100
)


# =========================================
# FEATURES AND TARGET
# =========================================

X = df[
    [
        "Recency",
        "Frequency",
        "Monetary"
    ]
]

y = df["Purchase"]


# =========================================
# TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:")
print(len(X_train))

print("\nTesting samples:")
print(len(X_test))


# =========================================
# DEFINE MODELS
# =========================================

models = {

    "Logistic Regression": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    )
}


# =========================================
# TRAIN MODELS
# =========================================

results = []

trained_models = {}

for name, model in models.items():

    print("\n=========================================")
    print(name)
    print("=========================================")

    # Train
    model.fit(
        X_train,
        y_train
    )

    # Predict
    y_pred = model.predict(
        X_test
    )

    # Probability
    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    # Metrics
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

    # Print metrics
    print(
        "Accuracy :",
        round(accuracy, 4)
    )

    print(
        "Precision:",
        round(precision, 4)
    )

    print(
        "Recall   :",
        round(recall, 4)
    )

    print(
        "F1 Score :",
        round(f1, 4)
    )

    print(
        "ROC-AUC  :",
        round(roc_auc, 4)
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    # Store results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

    trained_models[name] = model


# =========================================
# MODEL COMPARISON
# =========================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n=========================================")
print("MODEL COMPARISON")
print("=========================================")

print(
    results_df.to_string(
        index=False
    )
)


# =========================================
# SELECT BEST MODEL
# =========================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]

print("\n=========================================")
print("BEST MODEL")
print("=========================================")

print(
    best_model_name
)


# =========================================
# CREATE MODELS FOLDER
# =========================================

models_folder = os.path.join(
    project_folder,
    "models"
)

os.makedirs(
    models_folder,
    exist_ok=True
)


# =========================================
# SAVE BEST MODEL
# =========================================

model_path = os.path.join(
    models_folder,
    "best_model.pkl"
)

joblib.dump(
    best_model,
    model_path
)

print("\nBest model saved:")
print(model_path)


# =========================================
# SAVE MODEL RESULTS
# =========================================

results_path = os.path.join(
    project_folder,
    "data",
    "model_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print("\nModel results saved:")
print(results_path)


# =========================================
# SAVE FEATURE NAMES
# =========================================

feature_path = os.path.join(
    models_folder,
    "feature_names.pkl"
)

joblib.dump(
    list(X.columns),
    feature_path
)

print("\nFeature names saved:")
print(feature_path)


# =========================================
# SAVE TEST DATA FOR EVALUATION
# =========================================

test_data = X_test.copy()

test_data["Actual"] = y_test.values

test_data["Predicted"] = best_model.predict(
    X_test
)

test_data["Probability"] = best_model.predict_proba(
    X_test
)[:, 1]

test_path = os.path.join(
    project_folder,
    "data",
    "test_predictions.csv"
)

test_data.to_csv(
    test_path,
    index=False
)

print("\nTest predictions saved:")
print(test_path)


# =========================================
# FINAL MESSAGE
# =========================================

print("\n=========================================")
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=========================================")