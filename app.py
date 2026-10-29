
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    pairwise_distances
)

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="Smart Customer Purchase Prediction",
    page_icon="📊",
    layout="wide"
)

# ----------------------------------------------------------
# CUSTOM THEME
# ----------------------------------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: #e5e7eb;
}
[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #263449;
}
h1, h2, h3 {
    color: #f8fafc !important;
}
div[data-testid="stMetric"] {
    background-color: #151f32;
    border: 1px solid #293850;
    padding: 16px;
    border-radius: 12px;
}
div[data-testid="stMetricValue"] {
    color: #2dd4bf;
}
.stButton > button {
    background-color: #0d9488;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: 600;
}
.stButton > button:hover {
    background-color: #14b8a6;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------
DATA_PATH = "data/supervised_customer_data.csv"
FEATURES = ["Recency", "Frequency", "Monetary"]
TARGET = "Purchase"

MODEL_PATHS = [
    "models/best_tuned_model.pkl",
    "models/best_ensemble_model.pkl",
    "models/best_model.pkl"
]


# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)

    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=[TARGET]).copy()
    data[TARGET] = data[TARGET].astype(int)

    if not set(data[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Purchase must contain only 0 and 1.")

    return data


@st.cache_resource
def load_trained_model():
    errors = []

    for path in MODEL_PATHS:
        if os.path.exists(path):
            try:
                return joblib.load(path), path, None
            except Exception as exc:
                errors.append(f"{path}: {exc}")

    return None, None, (
        "No compatible model could be loaded. "
        + " | ".join(errors)
        if errors else
        "No trained model file was found."
    )


try:
    df = load_data()
    data_error = None
except Exception as exc:
    df = None
    data_error = str(exc)

model, model_path, model_error = load_trained_model()


# ----------------------------------------------------------
# HELPERS
# ----------------------------------------------------------
def make_chart():
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")
    ax.tick_params(colors="#cbd5e1")

    for spine in ax.spines.values():
        spine.set_color("#334155")

    ax.xaxis.label.set_color("#cbd5e1")
    ax.yaxis.label.set_color("#cbd5e1")
    ax.title.set_color("#f8fafc")

    return fig, ax


def evaluate_model(classifier, X_test, y_test):
    predictions = classifier.predict(X_test)

    result = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        )
    }

    try:
        probabilities = classifier.predict_proba(X_test)[:, 1]
        result["ROC-AUC"] = roc_auc_score(y_test, probabilities)
    except Exception:
        probabilities = None
        result["ROC-AUC"] = np.nan

    return result, predictions, probabilities


def get_comparison_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=50,
            learning_rate=0.5,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=1,
            random_state=42
        )
    }


# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------
with st.sidebar:
    st.title("📊 SmartPredict")
    st.caption("Customer Purchase Intelligence")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Home",
            "🔮 Customer Prediction",
            "📁 Dataset Analysis",
            "📈 Model Performance",
            "🧪 Model Evaluation",
            "🧠 Feature Selection",
            "🤖 Ensemble Algorithms",
            "🔬 Advanced Analysis"
        ]
    )

    st.divider()

    if df is not None:
        st.metric("Customer Records", f"{len(df):,}")

    if model is not None:
        st.success("Saved model loaded")
        st.caption(model_path)
    else:
        st.warning("Saved model unavailable")


# ----------------------------------------------------------
# VALIDATE DATASET
# ----------------------------------------------------------
if data_error:
    st.error(f"Dataset error: {data_error}")

if df is None:
    st.error(
        "Place supervised_customer_data.csv inside the data folder."
    )
    st.stop()


# ==========================================================
# PAGE 1: HOME
# ==========================================================
if page == "🏠 Home":
    st.title("Smart Customer Purchase Prediction")
    st.subheader("AI-powered customer behavior analysis and prediction")

    st.write(
        "Analyze customer behavior using Recency, Frequency and "
        "Monetary (RFM) features. Compare machine learning models, "
        "predict customer purchases and explore customer segments."
    )

    st.divider()

    total = len(df)
    buyers = int((df[TARGET] == 1).sum())
    nonbuyers = int((df[TARGET] == 0).sum())
    rate = buyers / total * 100 if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customer Records", f"{total:,}")
    c2.metric("Purchasers", f"{buyers:,}")
    c3.metric("Non-Purchasers", f"{nonbuyers:,}")
    c4.metric("Purchase Rate", f"{rate:.2f}%")

    left, right = st.columns(2)

    with left:
        st.subheader("Purchase Distribution")
        counts = df[TARGET].value_counts().sort_index()

        fig, ax = make_chart()
        ax.bar(
            [
                "No Purchase" if x == 0 else "Purchase"
                for x in counts.index
            ],
            counts.values
        )
        ax.set_ylabel("Customers")
        ax.set_title("Customer Purchase Distribution")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("RFM Summary")
        st.dataframe(
            df[FEATURES].describe().round(2),
            use_container_width=True
        )

    st.info(
        "Model predictions are estimates and depend on the "
        "training data and definition of the Purchase target."
    )


# ==========================================================
# PAGE 2: CUSTOMER PREDICTION
# ==========================================================
elif page == "🔮 Customer Prediction":
    st.title("🔮 Customer Purchase Prediction")

    st.write("Enter a customer's RFM values.")

    if model is None:
        st.error("The saved model could not be loaded.")
        st.code("python improve_model.py")
        if model_error:
            st.caption(model_error)
        st.stop()

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            recency = st.number_input(
                "Recency (days)", 0.0, 10000.0, 30.0
            )

        with c2:
            frequency = st.number_input(
                "Frequency (orders)", 0.0, 100000.0, 3.0
            )

        with c3:
            monetary = st.number_input(
                "Monetary (spending)", 0.0, 100000000.0, 1000.0
            )

        submitted = st.form_submit_button(
            "Predict Purchase",
            use_container_width=True
        )

    if submitted:
        customer = pd.DataFrame([{
            "Recency": recency,
            "Frequency": frequency,
            "Monetary": monetary
        }])

        try:
            prediction = int(model.predict(customer)[0])

            probability = None
            if hasattr(model, "predict_proba"):
                probability = float(
                    model.predict_proba(customer)[0][1]
                )

            st.divider()
            left, right = st.columns(2)

            with left:
                if prediction == 1:
                    st.success("Likely to Purchase")
                else:
                    st.warning("Unlikely to Purchase")

            with right:
                if probability is not None:
                    st.metric(
                        "Estimated Purchase Probability",
                        f"{probability * 100:.2f}%"
                    )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# ==========================================================
# PAGE 3: DATASET ANALYSIS
# ==========================================================
elif page == "📁 Dataset Analysis":
    st.title("📁 Dataset Analysis")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Duplicate Rows", f"{df.duplicated().sum():,}")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(100), use_container_width=True)

    st.subheader("Missing Values and Data Types")
    details = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isna().sum().values
    })
    st.dataframe(details, use_container_width=True)

    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe().T, use_container_width=True)

    feature = st.selectbox("Choose a feature", FEATURES)

    fig, ax = make_chart()
    ax.hist(df[feature].dropna(), bins=30)
    ax.set_title(f"Distribution of {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Count")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Feature Correlation")
    corr = df[FEATURES + [TARGET]].corr()

    fig, ax = make_chart()
    image = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.download_button(
        "Download Dataset CSV",
        df.to_csv(index=False).encode("utf-8"),
        "customer_dataset.csv",
        "text/csv"
    )


# ==========================================================
# PAGE 4: MODEL PERFORMANCE — ALL SIX MODELS
# ==========================================================
elif page == "📈 Model Performance":
    st.title("📈 Complete Model Comparison")

    st.write(
        "Compare Logistic Regression, Decision Tree, Random Forest, "
        "AdaBoost, Gradient Boosting and Voting Classifier."
    )

    st.warning(
        "Clicking the button trains new comparison models. "
        "The results may differ from your earlier tuned-model results."
    )

    if st.button("Train and Compare All Six Models",
                 use_container_width=True):

        X = df[FEATURES]
        y = df[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        # Fit imputer only on training data
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        models = get_comparison_models()
        rows = []

        progress = st.progress(0)
        status = st.empty()

        for i, (name, classifier) in enumerate(models.items()):
            status.write(f"Training {name}...")

            classifier.fit(X_train_imp, y_train)
            metrics, _, _ = evaluate_model(
                classifier, X_test_imp, y_test
            )

            rows.append({"Model": name, **metrics})
            progress.progress((i + 1) / 6)

        status.write("Training Voting Classifier...")

        voting = VotingClassifier(
            estimators=[
                ("lr", LogisticRegression(
                    max_iter=2000, random_state=42
                )),
                ("rf", RandomForestClassifier(
                    n_estimators=100, random_state=42
                )),
                ("gb", GradientBoostingClassifier(
                    n_estimators=100, random_state=42
                ))
            ],
            voting="soft"
        )

        voting.fit(X_train_imp, y_train)
        voting_metrics, _, _ = evaluate_model(
            voting, X_test_imp, y_test
        )
        rows.append({
            "Model": "Voting Classifier",
            **voting_metrics
        })

        progress.progress(1.0)
        status.success("Comparison completed.")

        results = pd.DataFrame(rows).sort_values(
            "Accuracy", ascending=False
        )

        os.makedirs("data", exist_ok=True)
        results.to_csv(
            "data/all_model_comparison.csv", index=False
        )

        st.subheader("Actual Test Results")
        st.dataframe(
            results.round(4),
            use_container_width=True
        )

        fig, ax = make_chart()
        ax.bar(results["Model"], results["Accuracy"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Test Accuracy")
        ax.set_title("Comparison of All Six Models")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.download_button(
            "Download Model Comparison",
            results.to_csv(index=False).encode("utf-8"),
            "all_model_comparison.csv",
            "text/csv"
        )

    # Display previously saved results when available
    saved_path = "data/all_model_comparison.csv"

    if os.path.exists(saved_path):
        st.subheader("Last Saved Comparison")
        saved = pd.read_csv(saved_path)
        st.dataframe(saved.round(4), use_container_width=True)


# ==========================================================
# PAGE 5: MODEL EVALUATION
# ==========================================================
elif page == "🧪 Model Evaluation":
    st.title("🧪 Model Evaluation")

    if model is None:
        st.error("No saved model is available.")
        if model_error:
            st.caption(model_error)
        st.stop()

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    try:
        predictions = model.predict(X_test)

        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(y_test, predictions, zero_division=0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{accuracy * 100:.2f}%")
        c2.metric("Precision", f"{precision * 100:.2f}%")
        c3.metric("Recall", f"{recall * 100:.2f}%")
        c4.metric("F1 Score", f"{f1 * 100:.2f}%")

        st.subheader("Classification Report")
        report = classification_report(
            y_test, predictions,
            output_dict=True, zero_division=0
        )
        st.dataframe(
            pd.DataFrame(report).T,
            use_container_width=True
        )

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, predictions, labels=[0, 1])

        fig, ax = make_chart()
        image = ax.imshow(cm)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Predicted 0", "Predicted 1"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Actual 0", "Actual 1"])

        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center", color="white"
                )

        fig.colorbar(image, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

        if probabilities is not None:
            auc = roc_auc_score(y_test, probabilities)
            fpr, tpr, _ = roc_curve(y_test, probabilities)

            st.metric("ROC-AUC", f"{auc:.4f}")

            fig, ax = make_chart()
            ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
            ax.plot([0, 1], [0, 1], "--")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

    except Exception as exc:
        st.error(f"Evaluation failed: {exc}")


# ==========================================================
# PAGE 6: FEATURE SELECTION
# ==========================================================
elif page == "🧠 Feature Selection":
    st.title("🧠 Feature Selection")

    X = df[FEATURES]
    y = df[TARGET]

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    selector = SelectKBest(score_func=f_classif, k="all")
    selector.fit(X_imp, y)
    anova = np.nan_to_num(selector.scores_, nan=0.0)

    mi = mutual_info_classif(
        X_imp, y, random_state=42
    )

    rf = RandomForestClassifier(
        n_estimators=200, random_state=42
    )
    rf.fit(X_imp, y)

    results = pd.DataFrame({
        "Feature": FEATURES,
        "ANOVA F Score": anova,
        "Mutual Information": mi,
        "Random Forest Importance": rf.feature_importances_
    })

    st.dataframe(results, use_container_width=True)

    metric = st.selectbox(
        "Ranking method",
        [
            "ANOVA F Score",
            "Mutual Information",
            "Random Forest Importance"
        ]
    )

    ordered = results.sort_values(metric)

    fig, ax = make_chart()
    ax.barh(ordered["Feature"], ordered[metric])
    ax.set_xlabel(metric)
    ax.set_title("Feature Relevance")
    st.pyplot(fig)
    plt.close(fig)


# ==========================================================
# PAGE 7: ENSEMBLE ALGORITHMS
# ==========================================================
elif page == "🤖 Ensemble Algorithms":
    st.title("🤖 Ensemble Algorithms")

    st.write(
        "Train and compare AdaBoost, Gradient Boosting and "
        "Voting Classifier on the same train/test split."
    )

    if st.button("Train Ensemble Models",
                 use_container_width=True):

        X = df[FEATURES]
        y = df[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        models = {
            "AdaBoost": AdaBoostClassifier(
                n_estimators=50,
                learning_rate=0.5,
                random_state=42
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=1,
                random_state=42
            ),
            "Voting Classifier": VotingClassifier(
                estimators=[
                    ("lr", LogisticRegression(
                        max_iter=2000, random_state=42
                    )),
                    ("rf", RandomForestClassifier(
                        n_estimators=100, random_state=42
                    )),
                    ("gb", GradientBoostingClassifier(
                        n_estimators=100, random_state=42
                    ))
                ],
                voting="soft"
            )
        }

        rows = []
        progress = st.progress(0)

        for i, (name, classifier) in enumerate(models.items()):
            classifier.fit(X_train_imp, y_train)
            metrics, _, _ = evaluate_model(
                classifier, X_test_imp, y_test
            )
            rows.append({"Model": name, **metrics})
            progress.progress((i + 1) / len(models))

        results = pd.DataFrame(rows)
        st.dataframe(results.round(4), use_container_width=True)

        fig, ax = make_chart()
        ax.bar(results["Model"], results["Accuracy"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title("Ensemble Comparison")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.download_button(
            "Download Ensemble Results",
            results.to_csv(index=False).encode("utf-8"),
            "ensemble_results.csv",
            "text/csv"
        )


# ==========================================================
# PAGE 8: ADVANCED ANALYSIS
# ==========================================================
elif page == "🔬 Advanced Analysis":
    st.title("🔬 Advanced Customer Analysis")

    analysis = st.selectbox(
        "Choose analysis",
        [
            "RFM Scatter Analysis",
            "DBSCAN Clustering",
            "K-Medoids-style Clustering"
        ]
    )

    if analysis == "RFM Scatter Analysis":
        x_col = st.selectbox("X-axis", FEATURES, index=0)
        y_col = st.selectbox("Y-axis", FEATURES, index=1)

        fig, ax = make_chart()

        for label, name in [(0, "No Purchase"), (1, "Purchase")]:
            subset = df[df[TARGET] == label]
            ax.scatter(
                subset[x_col], subset[y_col],
                alpha=0.6, label=name
            )

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    elif analysis == "DBSCAN Clustering":
        eps = st.slider("Epsilon", 0.1, 3.0, 0.7, 0.1)
        min_samples = st.slider("Minimum samples", 2, 20, 5)

        if st.button("Run DBSCAN"):
            X_imp = SimpleImputer(
                strategy="median"
            ).fit_transform(df[FEATURES])

            X_scaled = StandardScaler().fit_transform(X_imp)

            labels = DBSCAN(
                eps=eps, min_samples=min_samples
            ).fit_predict(X_scaled)

            output = df.copy()
            output["Cluster"] = labels

            st.subheader("Cluster Counts")
            st.dataframe(
                output["Cluster"].value_counts().sort_index()
                .rename("Customers")
            )

            st.metric(
                "Clusters excluding noise",
                len(set(labels) - {-1})
            )
            st.metric(
                "Noise records",
                int((labels == -1).sum())
            )

            fig, ax = make_chart()
            scatter = ax.scatter(
                X_scaled[:, 0], X_scaled[:, 1],
                c=labels, alpha=0.7
            )
            ax.set_xlabel("Scaled Recency")
            ax.set_ylabel("Scaled Frequency")
            fig.colorbar(scatter, ax=ax)
            st.pyplot(fig)
            plt.close(fig)

            st.download_button(
                "Download DBSCAN Results",
                output.to_csv(index=False).encode("utf-8"),
                "dbscan_clusters.csv",
                "text/csv"
            )

    elif analysis == "K-Medoids-style Clustering":
        st.caption(
            "Educational PAM-style clustering on a sample of customers."
        )

        k = st.slider("Number of clusters", 2, 8, 3)
        sample_size = min(600, len(df))

        if st.button("Run K-Medoids-style Clustering"):
            if sample_size <= k:
                st.error("Not enough customers for this value of k.")
                st.stop()

            sample = df[FEATURES].sample(
                n=sample_size, random_state=42
            ).reset_index(drop=True)

            X_imp = SimpleImputer(
                strategy="median"
            ).fit_transform(sample[FEATURES])

            X_scaled = StandardScaler().fit_transform(X_imp)
            distances = pairwise_distances(X_scaled)

            rng = np.random.default_rng(42)
            medoids = rng.choice(
                sample_size, size=k, replace=False
            )

            for _ in range(20):
                labels = np.argmin(
                    distances[:, medoids], axis=1
                )
                new_medoids = medoids.copy()

                for cluster_id in range(k):
                    members = np.where(labels == cluster_id)[0]

                    if len(members) == 0:
                        continue

                    costs = distances[
                        np.ix_(members, members)
                    ].sum(axis=1)

                    new_medoids[cluster_id] = members[
                        np.argmin(costs)
                    ]

                if np.array_equal(new_medoids, medoids):
                    break

                medoids = new_medoids

            labels = np.argmin(distances[:, medoids], axis=1)
            output = sample.copy()
            output["Cluster"] = labels

            st.subheader("Cluster Summary")
            st.dataframe(
                output.groupby("Cluster")[FEATURES].mean().round(2),
                use_container_width=True
            )

            fig, ax = make_chart()
            scatter = ax.scatter(
                X_scaled[:, 0], X_scaled[:, 1],
                c=labels, alpha=0.7
            )
            ax.scatter(
                X_scaled[medoids, 0],
                X_scaled[medoids, 1],
                marker="X", s=160, label="Medoids"
            )
            ax.set_xlabel("Scaled Recency")
            ax.set_ylabel("Scaled Frequency")
            ax.legend()
            fig.colorbar(scatter, ax=ax)
            st.pyplot(fig)
            plt.close(fig)

            st.download_button(
                "Download K-Medoids Results",
                output.to_csv(index=False).encode("utf-8"),
                "kmedoids_clusters.csv",
                "text/csv"
            )


# ----------------------------------------------------------
# FOOTER
# ----------------------------------------------------------
st.divider()
st.caption(
    "Smart Customer Purchase Prediction | Machine Learning Project"
)
