
# Smart Customer Purchase Prediction Using Machine Learning

## 1. Project Overview

This project uses supervised machine learning algorithms to predict whether a customer will make a purchase during a future period based on historical shopping behavior.

The project demonstrates data cleaning, feature engineering, target creation, model training, evaluation, visualization, and an interactive Streamlit application.

## 2. Problem Statement

Retail businesses need to understand customer purchasing behavior to improve customer engagement and plan marketing activities. This project analyzes historical retail transactions to predict future customer purchases.

## 3. Objectives

- Analyze real-world retail transaction data.
- Clean missing, duplicate, and invalid records.
- Perform feature engineering using RFM analysis.
- Define a future-purchase target variable.
- Train and compare four supervised learning algorithms.
- Evaluate models using multiple classification metrics.
- Visualize model performance and feature importance.
- Build an interactive Streamlit application.

## 4. Dataset

**Dataset:** Online Retail Dataset  
**Source:** UCI Machine Learning Repository  
**Link:** https://archive.ics.uci.edu/dataset/352/online+retail

The original dataset contains 541,909 transaction records and eight columns:

- InvoiceNo
- StockCode
- Description
- Quantity
- InvoiceDate
- UnitPrice
- CustomerID
- Country

The dataset represents historical transactions from a UK-based online retailer.

## 5. Data Preprocessing

The following operations were performed:

1. Removed duplicate transaction records.
2. Removed records with missing CustomerID.
3. Removed records with Quantity less than or equal to zero.
4. Removed records with UnitPrice less than or equal to zero.
5. Converted InvoiceDate to datetime format.
6. Created a TotalAmount feature.

The TotalAmount feature is calculated as:

`TotalAmount = Quantity × UnitPrice`

After these cleaning steps, 392,692 transaction records remained.

## 6. Feature Engineering: RFM Analysis

Three customer-level features were generated:

- **Recency:** Number of days since the customer's most recent purchase.
- **Frequency:** Number of unique invoices/orders made by the customer.
- **Monetary:** Total amount spent by the customer.

These features summarize historical customer behavior and are used as model inputs.

## 7. Target Variable

The target variable is `Purchase`.

- `1`: The customer made a purchase during the final 30-day prediction period.
- `0`: The customer did not make a purchase during that period.

Customer features are calculated from transactions before the future period to avoid using future transactions as input features.

## 8. Algorithms Implemented

### 8.1 Logistic Regression

A classification algorithm that estimates the probability of a customer making a future purchase.

### 8.2 Decision Tree

A tree-based classifier that applies feature-based decision rules to predict the target class.

### 8.3 Random Forest

An ensemble learning algorithm that combines predictions from multiple decision trees.

### 8.4 Gradient Boosting

An ensemble algorithm that builds models sequentially, with later models attempting to correct errors made by earlier models.

## 9. Experimental Setup

- Train-test split: 80:20
- Stratified splitting to preserve target-class proportions
- Input features: Recency, Frequency, Monetary
- Target: Purchase
- Evaluation metrics: Accuracy, Precision, Recall, F1 Score, ROC-AUC

## 10. Model Evaluation

The models are evaluated using:

- **Accuracy:** Proportion of all predictions that are correct.
- **Precision:** Proportion of predicted purchases that are actual purchases.
- **Recall:** Proportion of actual purchases correctly identified.
- **F1 Score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures how well the model ranks the two classes.

The project also generates confusion matrices and ROC curves for visual evaluation.

## 11. Feature Importance Analysis

Permutation importance was calculated for Random Forest and Gradient Boosting using F1 Score.

In the current experiment, Frequency had the highest permutation importance for both models, followed by Recency and Monetary.

This indicates that purchase frequency contributed the most to predictive performance in these experiments.

## 12. Application Features

The Streamlit application provides:

- Home page and project workflow
- Customer purchase prediction
- Dataset analysis
- Model performance comparison
- Detailed model evaluation
- Advanced analysis with generated graphs

## 13. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- Git and GitHub

## 14. Project Workflow

1. Load the UCI Online Retail dataset.
2. Clean transaction data.
3. Generate TotalAmount.
4. Create customer-level RFM features.
5. Define the future-purchase target.
6. Split the data into training and testing sets.
7. Train four supervised classification algorithms.
8. Compare evaluation metrics.
9. Analyze feature importance.
10. Visualize results in the Streamlit application.

## 15. How to Run the Project

### Install dependencies

```bash
pip install -r requirements.txt
```

### Download the dataset

Download the Online Retail dataset from the UCI repository and place `Online Retail.xlsx` inside the `data` folder.

### Run preprocessing and target creation

```bash
python src/data_analysis.py
python src/create_target.py
```

### Train and evaluate the model

```bash
python src/train_model.py
python src/evaluation.py
```

### Generate analysis graphs

```bash
python src/model_comparison.py
python src/feature_analysis.py
python src/compare_confusion_matrices.py
python src/roc_comparison.py
python src/feature_importance_analysis.py
```

### Launch the application

```bash
streamlit run app.py
```

## 16. Limitations

- The model uses only three customer-level features.
- Predictions depend on the historical transaction period and target definition.
- Model performance may vary with different data splits and parameter settings.
- Purchase behavior may change over time.
- Further validation and model tuning may improve performance.

## 17. Future Scope

- Add more customer and product features.
- Experiment with class balancing and hyperparameter tuning.
- Evaluate performance on a later, independent time period.
- Explore customer segmentation.
- Deploy the application for wider access.

## 18. Conclusion

This project demonstrates a complete supervised machine learning workflow for predicting future customer purchasing behavior from real-world retail data. It combines preprocessing, RFM feature engineering, four classification algorithms, model evaluation, visual analysis, and a Streamlit application.

## Author

Prachi Patil


## 15. Additional Algorithms (Integrated)

A new **🧩 New Algorithms** page has been added to the existing Streamlit application.

### Customer clustering
- **K-Means:** groups customers by distance to cluster centroids.
- **K-Medoids (PAM-style):** uses actual customer observations as cluster representatives.
- **DBSCAN:** identifies density-based clusters and noise/outlier points.
- The module uses `data/customer_rfm.csv`, standardizes the RFM inputs, and saves a PCA visualization and customer labels under `results/`.
- The default implementation analyses up to 1,500 sampled customers for responsiveness. Adjust this only if your computer can handle the extra computation.

### Boosting comparison
- **AdaBoost** and **Gradient Boosting** are trained on `data/supervised_customer_data.csv`.
- Both use Recency, Frequency, and Monetary as inputs and Purchase as the target.
- Results include Accuracy, Precision, Recall, F1 Score, and ROC-AUC, saved to `results/ensemble_comparison.csv`.
- Compare results on the same split; the highest score is not guaranteed in advance.

### HMM Forward and Viterbi
- `src/hmm_algorithms.py` implements Forward sequence likelihood and Viterbi most-likely hidden-state decoding.
- The app uses a small illustrative discrete HMM with symbols A/B. Its probabilities are demonstration values, not values learned from the Online Retail dataset, so the output should not be presented as a real customer-purchase prediction.

### Run the updated application
```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the **🧩 New Algorithms** page from the sidebar. Use the buttons in each tab to run the algorithms.
