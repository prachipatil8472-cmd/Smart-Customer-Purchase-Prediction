import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Smart Customer Purchase Prediction",
    page_icon="🛍️",
    layout="wide"
)

# =========================================================
# PROJECT PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
CUSTOMER_DATA_PATH = os.path.join(DATA_DIR, "supervised_customer_data.csv")
MODEL_RESULTS_PATH = os.path.join(DATA_DIR, "model_results.csv")
EVALUATION_PATH = os.path.join(DATA_DIR, "evaluation_results.csv")

# Existing evaluation images
CONFUSION_PATH = os.path.join(BASE_DIR, "confusion_matrix.png")
ROC_PATH = os.path.join(BASE_DIR, "roc_curve.png")
FEATURE_PATH = os.path.join(BASE_DIR, "feature_importance.png")

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def show_image_if_exists(path, caption=None):
    """Display an image if the requested file exists."""
    if os.path.exists(path):
        st.image(path, caption=caption, width="stretch")
        return True
    st.info(f"Graph not found yet: {os.path.relpath(path, BASE_DIR)}")
    return False


def load_csv(path, description):
    """Load a CSV file or return None with a helpful message."""
    if not os.path.exists(path):
        st.warning(
            f"{description} was not found: "
            f"`{os.path.relpath(path, BASE_DIR)}`"
        )
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.error(f"Could not read {description}: {exc}")
        return None


def display_results_table(data):
    """Display available model metrics in a readable table."""
    if data is None or data.empty:
        st.info("Model results are not available.")
        return

    display_data = data.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]:
        if col in display_data.columns:
            display_data[col] = pd.to_numeric(
                display_data[col], errors="coerce"
            ).round(4)

    st.dataframe(display_data, hide_index=True, width="stretch")


# =========================================================
# LOAD PROJECT DATA
# =========================================================
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as exc:
        st.error(f"Could not load the trained model: {exc}")
else:
    st.warning(
        "The trained model file is missing. Run `python src/train_model.py` "
        "after preparing the dataset to generate it."
    )

customer_data = load_csv(CUSTOMER_DATA_PATH, "Customer dataset")
results = load_csv(MODEL_RESULTS_PATH, "Model results")
evaluation = load_csv(EVALUATION_PATH, "Evaluation results")

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
st.sidebar.title("🛍️ Smart Customer Analytics")
st.sidebar.markdown("### Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "🎯 Customer Prediction",
        "📊 Dataset Analysis",
        "📈 Model Performance",
        "🔍 Model Evaluation",
        "🧪 Advanced Analysis",
        "🧩 New Algorithms"
    ]
)

# =========================================================
# HOME PAGE
# =========================================================
if page == "🏠 Home":
    st.title("🛍️ Smart Customer Purchase Prediction")
    st.subheader("Machine Learning Based Customer Analytics System")

    st.markdown(
        """
        This application uses supervised machine learning algorithms
        to analyze customer purchasing behavior and predict whether a
        customer is likely to make a purchase in a future period.
        """
    )

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Customers",
            f"{len(customer_data):,}" if customer_data is not None else "N/A"
        )

    with col2:
        st.metric("Input Features", "3")

    with col3:
        st.metric("Algorithms", "4")

    best_model_name = "Not available"
    if (
        results is not None
        and not results.empty
        and "F1 Score" in results.columns
        and "Model" in results.columns
    ):
        valid_results = results.copy()
        valid_results["F1 Score"] = pd.to_numeric(
            valid_results["F1 Score"], errors="coerce"
        )
        valid_results = valid_results.dropna(subset=["F1 Score"])
        if not valid_results.empty:
            best_model_name = valid_results.loc[
                valid_results["F1 Score"].idxmax(), "Model"
            ]

    with col4:
        st.metric("Best Model by F1", best_model_name)

    st.markdown("---")
    st.subheader("Project Workflow")
    st.markdown(
        """
        **1. Real-World Dataset:** UCI Online Retail transaction data

        ↓

        **2. Data Preprocessing:** Remove duplicates, missing customer IDs,
        invalid quantities and invalid prices

        ↓

        **3. Feature Engineering:** Recency, Frequency and Monetary features

        ↓

        **4. Target Creation:** Label future purchases in the final 30-day period

        ↓

        **5. Supervised Machine Learning:** Logistic Regression, Decision Tree,
        Random Forest and Gradient Boosting

        ↓

        **6. Model Evaluation:** Accuracy, Precision, Recall, F1 Score and ROC-AUC

        ↓

        **7. Prediction:** Predict future customer purchase behavior
        """
    )

    st.markdown("---")
    st.subheader("Project Objective")
    st.info(
        "The objective is to analyze historical customer behavior and predict "
        "whether a customer is likely to make a purchase in a future period."
    )

# =========================================================
# CUSTOMER PREDICTION PAGE
# =========================================================
elif page == "🎯 Customer Prediction":
    st.title("🎯 Customer Purchase Prediction")
    st.write(
        "Enter historical customer behavior to estimate the likelihood "
        "of a future purchase."
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

    if st.button("🔮 Predict Purchase", width="stretch"):
        if model is None:
            st.error(
                "The trained model is unavailable. Run "
                "`python src/train_model.py` first."
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
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(input_data)[0]
                    classes = list(model.classes_)
                    if 1 in classes:
                        probability = float(probabilities[classes.index(1)])
                    else:
                        probability = 0.0
                    st.metric("Estimated Purchase Probability",
                              f"{probability * 100:.2f}%")

                if int(prediction) == 1:
                    st.success("🟢 The model predicts a future purchase.")
                else:
                    st.warning("🔴 The model predicts no purchase in the future period.")

                st.subheader("Input Customer Information")
                st.dataframe(input_data, hide_index=True, width="stretch")

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

# =========================================================
# DATASET ANALYSIS PAGE
# =========================================================
elif page == "📊 Dataset Analysis":
    st.title("📊 Customer Dataset Analysis")
    st.write("Customer-level data used by the supervised prediction model.")

    if customer_data is None:
        st.info(
            "Prepare the dataset by running `python src/create_target.py`."
        )
    else:
        required = {"Recency", "Frequency", "Monetary"}
        if required.issubset(customer_data.columns):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Customers", f"{len(customer_data):,}")
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
        st.subheader("Customer Data Preview")
        st.dataframe(customer_data.head(100), width="stretch")

        for feature, title, xlabel in [
            ("Recency", "Customer Recency Distribution", "Recency (Days)"),
            ("Frequency", "Customer Purchase Frequency", "Number of Orders"),
            ("Monetary", "Customer Monetary Value Distribution", "Monetary Value")
        ]:
            if feature in customer_data.columns:
                st.subheader(title)
                fig, ax = plt.subplots(figsize=(9, 4))
                ax.hist(
                    pd.to_numeric(customer_data[feature], errors="coerce").dropna(),
                    bins=30,
                    edgecolor="black"
                )
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Number of Customers")
                ax.set_title(title)
                ax.grid(axis="y", linestyle="--", alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)

# =========================================================
# MODEL PERFORMANCE PAGE
# =========================================================
elif page == "📈 Model Performance":
    st.title("📈 Machine Learning Model Comparison")
    st.write("Compare the four supervised learning algorithms.")

    if results is not None:
        display_results_table(results)

        if not results.empty and {"Model", "F1 Score"}.issubset(results.columns):
            st.markdown("---")
            st.subheader("F1 Score Comparison")
            plot_data = results.copy()
            plot_data["F1 Score"] = pd.to_numeric(
                plot_data["F1 Score"], errors="coerce"
            )
            plot_data = plot_data.dropna(subset=["F1 Score"])

            if not plot_data.empty:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(plot_data["Model"], plot_data["F1 Score"])
                ax.set_xlabel("Algorithm")
                ax.set_ylabel("F1 Score")
                ax.set_ylim(0, 1)
                ax.set_title("F1 Score Comparison of ML Models")
                ax.tick_params(axis="x", rotation=20)
                ax.grid(axis="y", linestyle="--", alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                best_row = plot_data.loc[plot_data["F1 Score"].idxmax()]
                st.success(f"Best model by F1 Score: {best_row['Model']}")
                st.write(f"F1 Score: {best_row['F1 Score']:.4f}")

    st.markdown("---")
    st.subheader("Additional Model Comparison Graph")
    show_image_if_exists(
        os.path.join(RESULTS_DIR, "model_comparison.png"),
        "Comparison of accuracy, precision, recall, F1 Score and ROC-AUC"
    )

# =========================================================
# MODEL EVALUATION PAGE
# =========================================================
elif page == "🔍 Model Evaluation":
    st.title("🔍 Detailed Model Evaluation")

    if evaluation is not None:
        st.subheader("Evaluation Metrics")
        st.dataframe(evaluation, hide_index=True, width="stretch")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        show_image_if_exists(CONFUSION_PATH, "Confusion Matrix")

    with col2:
        st.subheader("ROC Curve")
        show_image_if_exists(ROC_PATH, "ROC Curve")

    st.markdown("---")
    st.subheader("Feature Importance")
    show_image_if_exists(FEATURE_PATH, "Feature Importance")

    st.info(
        "Accuracy measures overall correctness. Precision measures how many "
        "predicted purchases were actual purchases. Recall measures how many "
        "actual purchases were identified. F1 Score balances precision and "
        "recall, while ROC-AUC measures ranking/discrimination ability."
    )

# =========================================================
# ADVANCED ANALYSIS PAGE
# =========================================================
elif page == "🧪 Advanced Analysis":
    st.title("🧪 Advanced Model Analysis")
    st.write(
        "Explore experimental results, ROC curves, confusion matrices, "
        "feature importance and customer feature distributions."
    )

    st.header("1. Model Comparison")
    comparison_csv = os.path.join(RESULTS_DIR, "all_model_evaluation.csv")
    comparison_graph = os.path.join(RESULTS_DIR, "model_comparison.png")

    if os.path.exists(comparison_csv):
        comparison_df = pd.read_csv(comparison_csv)
        display_results_table(comparison_df)
    else:
        st.info(
            "Run `python src/compare_confusion_matrices.py` to generate "
            "the separate evaluation results."
        )

    show_image_if_exists(comparison_graph, "Model Comparison Graph")

    st.markdown("---")
    st.header("2. ROC Curve Comparison")
    show_image_if_exists(
        os.path.join(RESULTS_DIR, "roc_comparison.png"),
        "ROC curves for four algorithms"
    )

    st.markdown("---")
    st.header("3. Confusion Matrices for All Models")
    show_image_if_exists(
        os.path.join(RESULTS_DIR, "all_model_confusion_matrices.png"),
        "Confusion matrices for all four algorithms"
    )

    st.markdown("---")
    st.header("4. Feature Importance")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Random Forest")
        show_image_if_exists(
            os.path.join(RESULTS_DIR, "random_forest_importance.png")
        )
    with col2:
        st.subheader("Gradient Boosting")
        show_image_if_exists(
            os.path.join(RESULTS_DIR, "gradient_boosting_importance.png")
        )

    st.markdown("---")
    st.header("5. Customer Feature Distributions")

    for title, filename in [
        ("Recency Distribution", "recency_distribution.png"),
        ("Frequency Distribution", "frequency_distribution.png"),
        ("Monetary Distribution", "monetary_distribution.png"),
        ("Feature Comparison by Purchase Class", "feature_comparison.png")
    ]:
        st.subheader(title)
        show_image_if_exists(os.path.join(RESULTS_DIR, filename))

    st.markdown("---")
    st.header("6. ROC-AUC Results")
    auc_csv = os.path.join(RESULTS_DIR, "roc_auc_comparison.csv")
    if os.path.exists(auc_csv):
        auc_df = pd.read_csv(auc_csv)
        st.dataframe(auc_df, hide_index=True, width="stretch")
    else:
        st.info("Run `python src/roc_comparison.py` to generate ROC-AUC results.")


# =========================================================
# NEW ALGORITHMS PAGE
# =========================================================
elif page == "🧩 New Algorithms":
    st.title("🧩 Additional Algorithms")
    st.caption("Added to the existing Smart Customer Purchase Prediction project.")

    tab_cluster, tab_ensemble, tab_hmm = st.tabs([
        "Customer Clustering", "Boosting Comparison", "HMM Forward & Viterbi"
    ])

    with tab_cluster:
        st.subheader("Customer segmentation using RFM")
        st.write(
            "K-Means and K-Medoids form a requested number of groups. "
            "DBSCAN groups dense regions and labels outliers as noise. "
            "Features are scaled before clustering; the plot uses PCA only for display."
        )
        rfm_path = os.path.join(DATA_DIR, "customer_rfm.csv")
        if not os.path.exists(rfm_path):
            st.warning("Missing data/customer_rfm.csv. Please restore the existing RFM dataset.")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                k_value = st.slider("K for K-Means / K-Medoids", 2, 8, 4)
            with col2:
                eps_value = st.slider("DBSCAN eps", 0.1, 3.0, 0.8, 0.1)
            with col3:
                min_samples_value = st.slider("DBSCAN min_samples", 3, 20, 8)
            if st.button("Run clustering algorithms", key="run_customer_clustering"):
                try:
                    from src.customer_clustering import run_clustering
                    summary, plot_path, labels_path = run_clustering(
                        rfm_path, RESULTS_DIR, n_clusters=k_value,
                        dbscan_eps=eps_value, min_samples=min_samples_value
                    )
                    st.session_state["cluster_summary"] = summary
                    st.session_state["cluster_plot"] = plot_path
                    st.session_state["cluster_labels"] = labels_path
                except Exception as exc:
                    st.error(f"Clustering failed: {exc}")
            if "cluster_summary" in st.session_state:
                st.dataframe(st.session_state["cluster_summary"], hide_index=True, width="stretch")
                show_image_if_exists(st.session_state["cluster_plot"], "Customer cluster comparison (PCA projection)")
                st.download_button(
                    "Download customer cluster labels CSV",
                    data=open(st.session_state["cluster_labels"], "rb").read(),
                    file_name="customer_cluster_labels.csv",
                    mime="text/csv"
                )
        st.info(
            "K-Medoids is implemented with a PAM-style swap search. To keep it responsive, "
            "the clustering module samples up to 1,500 customers by default."
        )

    with tab_ensemble:
        st.subheader("AdaBoost vs Gradient Boosting")
        st.write("Both models use the project's existing supervised customer dataset and the same 80:20 stratified split.")
        supervised_path = os.path.join(DATA_DIR, "supervised_customer_data.csv")
        if st.button("Train and compare boosting models", key="run_ensemble_comparison"):
            try:
                from src.ensemble_comparison import compare_ensembles
                comparison = compare_ensembles(
                    supervised_path,
                    os.path.join(RESULTS_DIR, "ensemble_comparison.csv")
                )
                st.session_state["ensemble_comparison"] = comparison
            except Exception as exc:
                st.error(f"Ensemble comparison failed: {exc}")
        if "ensemble_comparison" in st.session_state:
            comp = st.session_state["ensemble_comparison"]
            display_results_table(comp)
            metric = st.selectbox(
                "Metric to visualize",
                ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
                key="ensemble_metric"
            )
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(comp["Model"], comp[metric])
            ax.set_ylim(0, 1)
            ax.set_ylabel(metric)
            ax.set_title(f"AdaBoost vs Gradient Boosting — {metric}")
            ax.grid(axis="y", alpha=0.25)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Compare the measured metrics; neither algorithm is guaranteed to perform best.")

    with tab_hmm:
        st.subheader("Hidden Markov Model: Forward and Viterbi")
        st.write(
            "This is an educational discrete-HMM demonstration. The sample probabilities below "
            "are illustrative, not learned from the retail dataset. Forward calculates the "
            "observation-sequence probability; Viterbi finds the most likely hidden-state path."
        )
        symbols = {"A": 0, "B": 1}
        seq_text = st.text_input("Observation sequence (use A and B, e.g. ABAB)", "ABAB").upper().replace(" ", "")
        start = [0.6, 0.4]
        transition = [[0.7, 0.3], [0.4, 0.6]]
        emission = [[0.5, 0.5], [0.1, 0.9]]
        st.code(
            "Start probabilities: [0.6, 0.4]\n"
            "Transition matrix: [[0.7, 0.3], [0.4, 0.6]]\n"
            "Emission matrix: [[0.5, 0.5], [0.1, 0.9]]"
        )
        if st.button("Run HMM algorithms", key="run_hmm"):
            if not seq_text or any(char not in symbols for char in seq_text):
                st.error("Enter a sequence containing only A and B, such as ABAB.")
            else:
                try:
                    from src.hmm_algorithms import forward_probability, viterbi_decode
                    obs = [symbols[char] for char in seq_text]
                    likelihood = forward_probability(start, transition, emission, obs)
                    path, path_probability = viterbi_decode(start, transition, emission, obs)
                    c1, c2 = st.columns(2)
                    c1.metric("Forward sequence probability", f"{likelihood:.8f}")
                    c2.metric("Viterbi path probability", f"{path_probability:.8f}")
                    st.write("Most likely hidden-state path:", " → ".join(map(str, path)))
                    st.caption("Hidden states are numbered 0 and 1. This example does not represent a trained customer prediction model.")
                except Exception as exc:
                    st.error(f"HMM calculation failed: {exc}")

# =========================================================
# SIDEBAR FOOTER
# =========================================================
st.sidebar.markdown("---")
st.sidebar.caption("Smart Customer Purchase Prediction")
st.sidebar.caption("Machine Learning Mini Project")
