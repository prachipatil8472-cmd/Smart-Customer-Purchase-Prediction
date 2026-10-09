
import os
import streamlit as st
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")


def show_advanced_analysis():
    st.title("Advanced Model Analysis")
    st.write(
        "Explore model performance, customer features, "
        "confusion matrices and feature importance."
    )

    # Model comparison
    st.header("1. Model Comparison")
    comparison_file = os.path.join(
        RESULTS_DIR, "all_model_evaluation.csv"
    )

    if os.path.exists(comparison_file):
        results = pd.read_csv(comparison_file)
        st.dataframe(results, hide_index=True, width="stretch")

        image_path = os.path.join(
            RESULTS_DIR, "model_comparison.png"
        )
        if os.path.exists(image_path):
            st.image(image_path, caption="Model Comparison Graph",
                     width="stretch")
        else:
            st.info("Run src/model_comparison.py to create this graph.")
    else:
        st.info(
            "Run src/compare_confusion_matrices.py first "
            "to generate model evaluation results."
        )

    # ROC comparison
    st.header("2. ROC Curve Comparison")
    roc_path = os.path.join(RESULTS_DIR, "roc_comparison.png")

    if os.path.exists(roc_path):
        st.image(roc_path, caption="ROC Curves for All Models",
                 width="stretch")
    else:
        st.info("Run src/roc_comparison.py to generate this graph.")

    # Confusion matrices
    st.header("3. Confusion Matrices")
    cm_path = os.path.join(
        RESULTS_DIR, "all_model_confusion_matrices.png"
    )

    if os.path.exists(cm_path):
        st.image(cm_path, caption="Confusion Matrices",
                 width="stretch")
    else:
        st.info(
            "Run src/compare_confusion_matrices.py "
            "to generate the confusion matrices."
        )

    # Feature importance
    st.header("4. Feature Importance")

    for model_name, filename in [
        ("Random Forest", "random_forest_importance.png"),
        ("Gradient Boosting", "gradient_boosting_importance.png")
    ]:
        path = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(path):
            st.subheader(model_name)
            st.image(path, width="stretch")

    # Feature distributions
    st.header("5. Customer Feature Analysis")

    for title, filename in [
        ("Recency Distribution", "recency_distribution.png"),
        ("Frequency Distribution", "frequency_distribution.png"),
        ("Monetary Distribution", "monetary_distribution.png"),
        ("Feature Comparison by Purchase Class",
         "feature_comparison.png")
    ]:
        path = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(path):
            st.subheader(title)
            st.image(path, width="stretch")

    # Show model metrics CSV if available
    st.header("6. Evaluation Metrics")
    metrics_path = os.path.join(
        RESULTS_DIR, "all_model_evaluation.csv"
    )

    if os.path.exists(metrics_path):
        metrics = pd.read_csv(metrics_path)
        st.dataframe(metrics, hide_index=True, width="stretch")
    else:
        st.info("Evaluation metrics are not available yet.")
