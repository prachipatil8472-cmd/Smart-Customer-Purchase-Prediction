import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt


# =========================================
# PAGE CONFIGURATION
# =========================================

st.set_page_config(
    page_title="Smart Customer Purchase Prediction",
    page_icon="🛍️",
    layout="wide"
)


# =========================================
# PROJECT PATH
# =========================================

project_folder = os.path.dirname(
    os.path.abspath(__file__)
)


# =========================================
# FILE PATHS
# =========================================

model_path = os.path.join(
    project_folder,
    "models",
    "best_model.pkl"
)

rfm_path = os.path.join(
    project_folder,
    "data",
    "supervised_customer_data.csv"
)

results_path = os.path.join(
    project_folder,
    "data",
    "model_results.csv"
)

evaluation_path = os.path.join(
    project_folder,
    "data",
    "evaluation_results.csv"
)

confusion_path = os.path.join(
    project_folder,
    "confusion_matrix.png"
)

roc_path = os.path.join(
    project_folder,
    "roc_curve.png"
)

feature_path = os.path.join(
    project_folder,
    "feature_importance.png"
)


# =========================================
# LOAD MODEL
# =========================================

model = joblib.load(
    model_path
)


# =========================================
# LOAD DATA
# =========================================

customer_data = pd.read_csv(
    rfm_path
)

results = pd.read_csv(
    results_path
)

evaluation = pd.read_csv(
    evaluation_path
)


# =========================================
# SIDEBAR
# =========================================

st.sidebar.title(
    "🛍️ Smart Customer Analytics"
)

st.sidebar.markdown(
    "### Navigation"
)

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "🎯 Customer Prediction",
        "📊 Dataset Analysis",
        "📈 Model Performance",
        "🔍 Model Evaluation"
    ]
)


# =========================================
# HOME
# =========================================

if page == "🏠 Home":

    st.title(
        "🛍️ Smart Customer Purchase Prediction"
    )

    st.subheader(
        "Machine Learning Based Customer Analytics System"
    )

    st.markdown(
        """
        This application uses supervised machine learning
        algorithms to analyze customer purchasing behavior
        and predict whether a customer is likely to make
        a purchase in the future.
        """
    )

    st.markdown("---")

    # Metrics

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Customers",
            len(customer_data)
        )

    with col2:

        st.metric(
            "Features",
            3
        )

    with col3:

        st.metric(
            "Algorithms",
            4
        )

    best_model = results.loc[
        results["F1 Score"].idxmax(),
        "Model"
    ]

    with col4:

        st.metric(
            "Best Model",
            best_model
        )

    st.markdown("---")

    st.subheader(
        "Project Workflow"
    )

    st.markdown(
        """
        **1️⃣ Real-World Dataset**

        Online Retail transaction data

        ↓

        **2️⃣ Data Preprocessing**

        Missing values, duplicates and invalid records

        ↓

        **3️⃣ Feature Engineering**

        Recency, Frequency and Monetary features

        ↓

        **4️⃣ Supervised Machine Learning**

        Logistic Regression, Decision Tree,
        Random Forest and Gradient Boosting

        ↓

        **5️⃣ Model Evaluation**

        Accuracy, Precision, Recall, F1-score and ROC-AUC

        ↓

        **6️⃣ Prediction**

        Predict future customer purchase behavior
        """
    )

    st.markdown("---")

    st.subheader(
        "Project Objective"
    )

    st.info(
        """
        The objective of this project is to develop a
        machine learning system capable of analyzing
        historical customer behavior and predicting
        whether a customer is likely to make a purchase
        in a future period.
        """
    )


# =========================================
# CUSTOMER PREDICTION
# =========================================

elif page == "🎯 Customer Prediction":

    st.title(
        "🎯 Customer Purchase Prediction"
    )

    st.write(
        """
        Enter the customer's historical behavior
        to predict whether they are likely to make
        a future purchase.
        """
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:

        recency = st.number_input(
            "Recency (Days)",
            min_value=0,
            max_value=1000,
            value=30,
            step=1
        )

    with col2:

        frequency = st.number_input(
            "Frequency (Number of Orders)",
            min_value=1,
            max_value=1000,
            value=10,
            step=1
        )

    with col3:

        monetary = st.number_input(
            "Monetary Value",
            min_value=0.0,
            max_value=1000000.0,
            value=1000.0,
            step=100.0
        )

    st.markdown("---")

    if st.button(
        "🔮 Predict Purchase",
        width="stretch"
    ):

        input_data = pd.DataFrame({
            "Recency": [recency],
            "Frequency": [frequency],
            "Monetary": [monetary]
        })

        prediction = model.predict(
            input_data
        )[0]

        probability = model.predict_proba(
            input_data
        )[0][1]

        st.markdown("---")

        if prediction == 1:

            st.success(
                "🟢 Customer is likely to make a purchase!"
            )

        else:

            st.warning(
                "🔴 Customer is unlikely to make a purchase."
            )

        st.metric(
            "Purchase Probability",
            f"{probability * 100:.2f}%"
        )

        st.markdown("---")

        st.subheader(
            "Input Customer Information"
        )

        st.dataframe(
            input_data,
            width="stretch"
        )


# =========================================
# DATASET ANALYSIS
# =========================================

elif page == "📊 Dataset Analysis":

    st.title(
        "📊 Customer Dataset Analysis"
    )

    st.write(
        "Customer-level dataset used for prediction."
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Customers",
            len(customer_data)
        )

    with col2:

        st.metric(
            "Average Recency",
            f"{customer_data['Recency'].mean():.2f} days"
        )

    with col3:

        st.metric(
            "Average Monetary Value",
            f"{customer_data['Monetary'].mean():.2f}"
        )

    st.markdown("---")

    st.subheader(
        "Customer Data"
    )

    st.dataframe(
        customer_data.head(100),
        width="stretch"
    )

    st.markdown("---")

    st.subheader(
        "Recency Distribution"
    )

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.hist(
        customer_data["Recency"],
        bins=30
    )

    ax.set_xlabel(
        "Recency (Days)"
    )

    ax.set_ylabel(
        "Number of Customers"
    )

    ax.set_title(
        "Customer Recency Distribution"
    )

    st.pyplot(fig)

    plt.close(fig)

    st.markdown("---")

    st.subheader(
        "Frequency Distribution"
    )

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.hist(
        customer_data["Frequency"],
        bins=30
    )

    ax.set_xlabel(
        "Frequency"
    )

    ax.set_ylabel(
        "Number of Customers"
    )

    ax.set_title(
        "Customer Purchase Frequency"
    )

    st.pyplot(fig)

    plt.close(fig)


# =========================================
# MODEL PERFORMANCE
# =========================================

elif page == "📈 Model Performance":

    st.title(
        "📈 Machine Learning Model Comparison"
    )

    st.write(
        """
        Four supervised machine learning algorithms
        were trained and compared.
        """
    )

    st.markdown("---")

    display_results = results.copy()

    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]:

        display_results[column] = display_results[
            column
        ].round(4)

    st.dataframe(
        display_results,
        width="stretch"
    )

    st.markdown("---")

    st.subheader(
        "F1 Score Comparison"
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.bar(
        results["Model"],
        results["F1 Score"]
    )

    ax.set_xlabel(
        "Algorithm"
    )

    ax.set_ylabel(
        "F1 Score"
    )

    ax.set_title(
        "Comparison of Machine Learning Models"
    )

    plt.xticks(
        rotation=20
    )

    st.pyplot(fig)

    plt.close(fig)

    st.markdown("---")

    best_row = results.loc[
        results["F1 Score"].idxmax()
    ]

    st.success(
        f"🏆 Best Model: {best_row['Model']}"
    )

    st.write(
        f"F1 Score: {best_row['F1 Score']:.4f}"
    )


# =========================================
# MODEL EVALUATION
# =========================================

elif page == "🔍 Model Evaluation":

    st.title(
        "🔍 Detailed Model Evaluation"
    )

    st.markdown("---")

    st.subheader(
        "Evaluation Metrics"
    )

    evaluation_display = evaluation.copy()

    evaluation_display["Score"] = (
        evaluation_display["Score"]
        .round(4)
    )

    st.dataframe(
        evaluation_display,
        width="stretch"
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Confusion Matrix"
        )

        if os.path.exists(
            confusion_path
        ):

            st.image(
                confusion_path,
                width="stretch"
            )

    with col2:

        st.subheader(
            "ROC Curve"
        )

        if os.path.exists(
            roc_path
        ):

            st.image(
                roc_path,
                width="stretch"
            )

    st.markdown("---")

    st.subheader(
        "Feature Importance"
    )

    if os.path.exists(
        feature_path
    ):

        st.image(
            feature_path,
            width="stretch"
        )

    st.markdown("---")

    st.info(
        """
        The evaluation metrics help determine how well
        the selected machine learning model performs on
        unseen test data.
        """
    )


# =========================================
# FOOTER
# =========================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "Smart Customer Purchase Prediction"
)

st.sidebar.caption(
    "Machine Learning Mini Project"
)