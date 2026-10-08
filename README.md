# 🛍️ Smart Customer Purchase Prediction Using Machine Learning

## 📌 Project Overview

Smart Customer Purchase Prediction is a supervised machine learning project that analyzes historical customer transaction data and predicts whether a customer is likely to make a purchase in a future period.

The project uses real-world e-commerce transaction data and demonstrates the complete machine learning workflow, including data preprocessing, feature engineering, model building, model evaluation, and deployment through a Streamlit web application.

---

## 🎯 Objectives

- Analyze real-world customer transaction data.
- Perform data preprocessing and cleaning.
- Handle missing values and duplicate records.
- Perform feature engineering using RFM analysis.
- Predict future customer purchasing behavior.
- Compare multiple supervised machine learning algorithms.
- Evaluate model performance using different metrics.
- Build an interactive web application using Streamlit.

---

## 📊 Dataset

### Online Retail Dataset

The project uses the Online Retail dataset containing transaction records from a UK-based online retail store.

### Dataset Features

- InvoiceNo
- StockCode
- Description
- Quantity
- InvoiceDate
- UnitPrice
- CustomerID
- Country

The dataset is available from the UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/352/online+retail

---

## 🧹 Data Preprocessing

The following preprocessing techniques were applied:

- Removed duplicate records.
- Removed records with missing CustomerID.
- Removed invalid quantity values.
- Removed invalid unit price values.
- Converted InvoiceDate into datetime format.
- Created TotalAmount feature.

### Total Amount

TotalAmount was calculated as:

Quantity × UnitPrice

---

## 🔧 Feature Engineering

RFM analysis was used to create customer-level features.

### Recency

Number of days since the customer's most recent purchase.

### Frequency

Number of unique orders made by the customer.

### Monetary

Total amount spent by the customer.

These features were used as input variables for the machine learning models.

---

## 🤖 Machine Learning Algorithms

The following supervised learning algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting

Random Forest and Gradient Boosting are ensemble learning techniques.

---

## 🎯 Target Variable

The target variable represents whether a customer made a purchase during the defined future prediction period.

- `1` → Customer made a future purchase
- `0` → Customer did not make a future purchase

Historical customer behavior was used to predict future purchasing behavior.

---

## 📈 Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- ROC Curve

The model with the best F1 Score was selected as the final model.

---

## 🖥️ Application

A Streamlit web application was developed to provide an interactive interface.

### Application Features

- Customer purchase prediction
- Dataset analysis
- Model comparison
- Model evaluation
- Confusion matrix
- ROC curve
- Feature analysis

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- Excel
- Git
- GitHub

---

## 📁 Project Structure

```text
Smart-Customer-Purchase-Prediction/
│
├── data/
├── models/
├── src/
│   ├── data_analysis.py
│   ├── create_target.py
│   ├── train_model.py
│   └── evaluation.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore