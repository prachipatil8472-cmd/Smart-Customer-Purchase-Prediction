import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Smart Customer Purchase Prediction",
    page_icon="🛍️",
    layout="wide"
)

# =====================================================
# CUSTOM DESIGN
# =====================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #08111f, #10243a);
    color: #edf5ff;
}
[data-testid="stSidebar"] {
    background: #101b2d;
    border-right: 1px solid #2dd4bf;
}
.block-container {
    max-width: 1450px;
    padding-top: 2rem;
}
h1, h2, h3 {
    color: #f4f8ff !important;
}
p, label {
    color: #d6e2f1;
}
div[data-testid="stMetric"] {
    background: #14243a;
    border: 1px solid #2dd4bf55;
    border-radius: 15px;
    padding: 18px;
}
div[data-testid="stMetricLabel"] {
    color: #9aacc3;
}
div[data-testid="stMetricValue"] {
    color: #5eead4;
}
.stButton > button, [data-testid="stFormSubmitButton"] button {
    background: #14b8a6;
    color: #061522;
    font-weight: bold;
    border: none;
    border-radius: 10px;
    min-height: 42px;
}
.stButton > button:hover {
    background: #5eead4;
    color: #061522;
}
.hero {
    background: linear-gradient(120deg, #123b4b, #14243a);
    padding: 30px;
    border-radius: 20px;
    border: 1px solid #2dd4bf;
    margin-bottom: 22px;
}
.hero h1 {
    font-size: 38px;
    margin-bottom: 10px;
}
.hero p {
    color: #c1d1e5;
    font-size: 17px;
}
hr {
    border-color: #34465f;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# PROJECT PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SRC_DIR = os.path.join(BASE_DIR, "src")

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
CUSTOMER_DATA_PATH = os.path.join(
    DATA_DIR, "supervised_customer_data.csv"
)
MODEL_RESULTS_PATH = os.path.join(DATA_DIR, "model_results.csv")
EVALUATION_PATH = os.path.join(DATA_DIR, "evaluation_results.csv")

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def load_csv(path, name):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as error:
        st.error(f"Unable to load {name}: {error}")
        return None


def show_image(path, caption=None):
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.info(f"Graph not found: {os.path.relpath(path, BASE_DIR)}")


def numeric_values(series):
    return pd.to_numeric(series, errors="coerce").dropna()


def show_histogram(data, column, title, xlabel):
    if column not in data.columns:
        st.info(f"{column} is not available in the dataset.")
        return

    values = numeric_values(data[column])

    if values.empty:
        st.info(f"No numeric values available for {column}.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#14243a")
    ax.set_facecolor("#14243a")

    ax.hist(values, bins=30, color="#2dd4bf", edgecolor="#08111f")
    ax.set_title(title, color="white")
    ax.set_xlabel(xlabel, color="white")
    ax.set_ylabel("Customers", color="white")
    ax.tick_params(colors="white")
    ax.grid(axis="y", alpha=0.2)

    for spine in ax.spines.values():
        spine.set_color("#34465f")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def show_results_table(data):
    if data is None or data.empty:
        st.info("Results are not available.")
        return

    display_data = data.copy()

    for column in [
        "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"
    ]:
        if column in display_data.columns:
            display_data[column] = pd.to_numeric(
                display_data[column], errors="coerce"
            ).round(4)

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# LOAD DATA AND MODEL
# =====================================================
customer_data = load_csv(CUSTOMER_DATA_PATH, "customer dataset")
results = load_csv(MODEL_RESULTS_PATH, "model results")
evaluation = load_csv(EVALUATION_PATH, "evaluation results")

model = None

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as error:
        st.error(f"Unable to load trained model: {error}")

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.markdown("# 🛍️ Smart Analytics")
st.sidebar.caption("CUSTOMER PURCHASE PREDICTION")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Home",
        "🎯 Customer Prediction",
        "📊 Dataset Analysis",
        "📈 Model Performance",
        "🔍 Model Evaluation",
        "🧪 Advanced Analysis",
        "🧠 New Algorithms"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Machine Learning Mini Project")
st.sidebar.caption("Python • Pandas • Scikit-learn")

# =====================================================
# HOME PAGE
# =====================================================
if page == "🏠 Home":

    st.markdown("""
    <div class="hero">
        <h1>🛍️ Smart Customer Purchase Prediction</h1>
        <p>
        Machine learning dashboard to analyse customer behaviour,
        compare algorithms and predict future purchasing activity.
        </p>
    </div>
    """, unsafe_allow_html=True)

    best_model = "Not available"

    if (
        results is not None
        and not results.empty
        and {"Model", "F1 Score"}.issubset(results.columns)
    ):
        temp = results.copy()
        temp["F1 Score"] = pd.to_numeric(
            temp["F1 Score"], errors="coerce"
        )
        temp = temp.dropna(subset=["F1 Score"])

        if not temp.empty:
            best_model = temp.loc[
                temp["F1 Score"].idxmax(), "Model"
            ]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Customers",
        f"{len(customer_data):,}"
        if customer_data is not None else "N/A"
    )
    c2.metric("RFM Features", "3")
    c3.metric("ML Algorithms", "4")
    c4.metric("Best Model by F1", str(best_model))

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.subheader("📌 Project Objective")
        st.write(
            "Analyse historical customer purchasing behaviour and "
            "predict whether a customer may make a future purchase."
        )

        st.subheader("📊 RFM Features")
        st.markdown("""
        - **Recency:** How recently a customer purchased.
        - **Frequency:** Number of customer orders.
        - **Monetary:** Total recorded customer spending.
        """)

    with right:
        st.subheader("⚙️ Project Workflow")
        st.markdown("""
        1. Collect retail transaction data.
        2. Clean missing and invalid records.
        3. Calculate RFM features.
        4. Prepare the purchase target.
        5. Train machine learning algorithms.
        6. Evaluate model performance.
        7. Predict future customer behaviour.
        """)

    st.markdown("---")
    st.subheader("📈 Customer Insights")

    if customer_data is not None and not customer_data.empty:
        features = [
            x for x in ["Recency", "Frequency", "Monetary"]
            if x in customer_data.columns
        ]

        columns = st.columns(max(1, len(features)))

        for index, feature in enumerate(features):
            values = numeric_values(customer_data[feature])
            average = values.mean() if not values.empty else 0

            columns[index].metric(
                f"Average {feature}",
                f"{average:,.2f}"
            )
    else:
        st.info("Customer dataset is not available yet.")

# =====================================================
# CUSTOMER PREDICTION
# =====================================================
elif page == "🎯 Customer Prediction":

    st.title("🎯 Customer Purchase Prediction")
    st.write(
        "Enter customer RFM information to generate a prediction."
    )
    st.markdown("---")

    with st.form("prediction_form"):

        c1, c2, c3 = st.columns(3)

        with c1:
            recency = st.number_input(
                "Recency (Days)", 0, 1000, 30
            )

        with c2:
            frequency = st.number_input(
                "Frequency (Orders)", 1, 1000, 10
            )

        with c3:
            monetary = st.number_input(
                "Monetary Value", 0.0, 1000000.0, 1000.0,
                step=100.0
            )

        submitted = st.form_submit_button(
            "🔮 Predict Purchase",
            use_container_width=True
        )

    if submitted:

        if model is None:
            st.error(
                "Trained model not found. Check models/best_model.pkl."
            )

        else:
            input_data = pd.DataFrame({
                "Recency": [recency],
                "Frequency": [frequency],
                "Monetary": [monetary]
            })

            try:
                prediction = model.predict(input_data)[0]

                st.markdown("---")
                left, right = st.columns(2)

                with left:
                    if int(prediction) == 1:
                        st.success(
                            "🟢 Model predicts a future purchase."
                        )
                    else:
                        st.warning(
                            "🔴 Model predicts no future purchase."
                        )

                with right:
                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(input_data)[0]
                        classes = list(model.classes_)

                        probability = (
                            float(probabilities[classes.index(1)])
                            if 1 in classes else 0.0
                        )

                        st.metric(
                            "Estimated Purchase Probability",
                            f"{probability * 100:.2f}%"
                        )

                st.subheader("Customer Input")
                st.dataframe(
                    input_data,
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    "Predictions are model estimates, not guarantees."
                )

            except Exception as error:
                st.error(f"Prediction failed: {error}")

# =====================================================
# DATASET ANALYSIS
# =====================================================
elif page == "📊 Dataset Analysis":

    st.title("📊 Customer Dataset Analysis")
    st.write("Explore customer-level data and RFM distributions.")

    if customer_data is None:
        st.warning(
            "Dataset not found. Check data/supervised_customer_data.csv."
        )

    else:
        c1, c2, c3 = st.columns(3)

        c1.metric("Total Customers", f"{len(customer_data):,}")

        if "Recency" in customer_data.columns:
            c2.metric(
                "Average Recency",
                f"{numeric_values(customer_data['Recency']).mean():.2f} days"
            )

        if "Monetary" in customer_data.columns:
            c3.metric(
                "Average Monetary",
                f"{numeric_values(customer_data['Monetary']).mean():,.2f}"
            )

        st.markdown("---")
        st.subheader("Customer Data Preview")
        st.dataframe(
            customer_data.head(100),
            use_container_width=True
        )

        left, right = st.columns(2)

        with left:
            show_histogram(
                customer_data, "Recency",
                "Recency Distribution", "Days"
            )
            show_histogram(
                customer_data, "Monetary",
                "Monetary Distribution", "Monetary Value"
            )

        with right:
            show_histogram(
                customer_data, "Frequency",
                "Purchase Frequency", "Number of Orders"
            )

            st.info("""
            **How to interpret RFM**

            - Lower recency means a more recent purchase.
            - Higher frequency means more orders.
            - Monetary represents recorded spending.
            """)

# =====================================================
# MODEL PERFORMANCE
# =====================================================
elif page == "📈 Model Performance":

    st.title("📈 Machine Learning Model Performance")
    st.write("Compare the results of the supervised algorithms.")

    if results is None:
        st.info("File data/model_results.csv is not available.")

    else:
        show_results_table(results)

        metrics = [
            x for x in
            ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
            if x in results.columns
        ]

        if "Model" in results.columns and metrics:

            selected_metric = st.selectbox(
                "Select comparison metric", metrics
            )

            plot_data = results[["Model", selected_metric]].copy()
            plot_data[selected_metric] = pd.to_numeric(
                plot_data[selected_metric], errors="coerce"
            )
            plot_data = plot_data.dropna()

            if not plot_data.empty:

                fig, ax = plt.subplots(figsize=(10, 5))
                fig.patch.set_facecolor("#14243a")
                ax.set_facecolor("#14243a")

                bars = ax.bar(
                    plot_data["Model"].astype(str),
                    plot_data[selected_metric],
                    color="#2dd4bf"
                )

                ax.set_ylim(0, 1)
                ax.set_ylabel(selected_metric, color="white")
                ax.set_title(
                    f"{selected_metric} Comparison",
                    color="white"
                )
                ax.tick_params(axis="x", rotation=20, colors="white")
                ax.tick_params(axis="y", colors="white")
                ax.grid(axis="y", alpha=0.2)

                for bar, value in zip(
                    bars, plot_data[selected_metric]
                ):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.015,
                        f"{value:.3f}",
                        ha="center",
                        color="white"
                    )

                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                best = plot_data.loc[
                    plot_data[selected_metric].idxmax()
                ]

                st.success(
                    f"Highest {selected_metric}: {best['Model']} "
                    f"({best[selected_metric]:.4f})"
                )

    st.markdown("---")
    st.subheader("Additional Model Comparison")
    show_image(
        os.path.join(RESULTS_DIR, "model_comparison.png")
    )

# =====================================================
# MODEL EVALUATION
# =====================================================
elif page == "🔍 Model Evaluation":

    st.title("🔍 Detailed Model Evaluation")

    if evaluation is not None:
        st.subheader("Evaluation Metrics")
        show_results_table(evaluation)

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.subheader("Confusion Matrix")
        show_image(
            os.path.join(BASE_DIR, "confusion_matrix.png")
        )

    with right:
        st.subheader("ROC Curve")
        show_image(
            os.path.join(BASE_DIR, "roc_curve.png")
        )

    st.markdown("---")
    st.subheader("Feature Importance")
    show_image(
        os.path.join(BASE_DIR, "feature_importance.png")
    )

    st.info("""
    **Evaluation Metrics**

    - Accuracy: Overall correctness.
    - Precision: Correctness of predicted purchases.
    - Recall: Actual purchases identified.
    - F1 Score: Balance between precision and recall.
    - ROC-AUC: Ability to distinguish between classes.
    """)

# =====================================================
# ADVANCED ANALYSIS
# =====================================================
elif page == "🧪 Advanced Analysis":

    st.title("🧪 Advanced Model Analysis")

    st.subheader("1. Model Comparison")

    comparison_csv = os.path.join(
        RESULTS_DIR, "all_model_evaluation.csv"
    )
    comparison_data = load_csv(
        comparison_csv, "all-model evaluation"
    )

    if comparison_data is not None:
        show_results_table(comparison_data)
    else:
        st.info("Additional comparison CSV is not available.")

    show_image(
        os.path.join(RESULTS_DIR, "model_comparison.png"),
        "Model Comparison"
    )

    st.markdown("---")
    st.subheader("2. ROC Curve Comparison")
    show_image(
        os.path.join(RESULTS_DIR, "roc_comparison.png")
    )

    st.markdown("---")
    st.subheader("3. Confusion Matrices")
    show_image(
        os.path.join(
            RESULTS_DIR, "all_model_confusion_matrices.png"
        )
    )

    st.markdown("---")
    st.subheader("4. Feature Importance")

    left, right = st.columns(2)

    with left:
        st.write("Random Forest")
        show_image(
            os.path.join(
                RESULTS_DIR, "random_forest_importance.png"
            )
        )

    with right:
        st.write("Gradient Boosting")
        show_image(
            os.path.join(
                RESULTS_DIR, "gradient_boosting_importance.png"
            )
        )

    st.markdown("---")
    st.subheader("5. Feature Distributions")

    for title, filename in [
        ("Recency Distribution", "recency_distribution.png"),
        ("Frequency Distribution", "frequency_distribution.png"),
        ("Monetary Distribution", "monetary_distribution.png"),
        ("Feature Comparison", "feature_comparison.png")
    ]:
        st.write(title)
        show_image(os.path.join(RESULTS_DIR, filename))

    st.markdown("---")
    st.subheader("6. ROC-AUC Results")

    auc_data = load_csv(
        os.path.join(RESULTS_DIR, "roc_auc_comparison.csv"),
        "ROC-AUC comparison"
    )

    if auc_data is not None:
        st.dataframe(auc_data, use_container_width=True)
    else:
        st.info("ROC-AUC comparison CSV is not available.")

# =====================================================
# NEW ALGORITHMS
# =====================================================
elif page == "🧠 New Algorithms":

    st.title("🧠 Additional Machine Learning Algorithms")
    st.write(
        "Check the availability of the extra algorithm scripts "
        "in your src folder."
    )

    st.markdown("---")

    algorithms = {
        "Customer Clustering": "customer_clustering.py",
        "AdaBoost and Gradient Boosting": "ensemble_comparison.py",
        "HMM Demonstration": "hmm_algorithms.py"
    }

    for name, filename in algorithms.items():

        script_path = os.path.join(SRC_DIR, filename)

        with st.expander(name):

            if os.path.exists(script_path):
                st.success(f"Script found: src/{filename}")
            else:
                st.warning(f"Script not found: src/{filename}")

    st.info("""
    **Algorithms covered**

    - K-Means, K-Medoids and DBSCAN: customer clustering.
    - AdaBoost and Gradient Boosting: ensemble learning.
    - HMM: Hidden Markov Model educational demonstration.

    These modules must exist in your src folder to be available.
    The HMM example should not be treated as a validated purchase
    prediction unless it has been appropriately trained and evaluated.
    """)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown(
    "<center>Smart Customer Purchase Prediction | "
    "Machine Learning Mini Project</center>",
    unsafe_allow_html=True
)