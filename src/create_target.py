import pandas as pd
import os


# =========================================
# PROJECT PATH
# =========================================

project_folder = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

cleaned_path = os.path.join(
    project_folder,
    "data",
    "cleaned_retail.csv"
)


# =========================================
# LOAD CLEANED DATA
# =========================================

df = pd.read_csv(
    cleaned_path,
    parse_dates=["InvoiceDate"]
)

print("Cleaned dataset loaded successfully!")

print("Dataset shape:", df.shape)


# =========================================
# DEFINE DATE
# =========================================

max_date = df["InvoiceDate"].max()

print("\nLatest transaction date:")
print(max_date)


# =========================================
# CREATE PAST AND FUTURE PERIODS
# =========================================

# Use the last 30 days as the future prediction period
future_start = max_date - pd.Timedelta(days=30)

# Data before future period
past_data = df[
    df["InvoiceDate"] < future_start
].copy()

# Data during future period
future_data = df[
    df["InvoiceDate"] >= future_start
].copy()


print("\nPast data:")
print(past_data.shape)

print("\nFuture data:")
print(future_data.shape)


# =========================================
# CREATE CUSTOMER FEATURES FROM PAST DATA
# =========================================

snapshot_date = past_data["InvoiceDate"].max() + pd.Timedelta(days=1)

customer_features = past_data.groupby(
    "CustomerID"
).agg({

    "InvoiceDate":
        lambda x: (snapshot_date - x.max()).days,

    "InvoiceNo":
        "nunique",

    "TotalAmount":
        "sum"
})


# =========================================
# RENAME FEATURES
# =========================================

customer_features.rename(
    columns={
        "InvoiceDate": "Recency",
        "InvoiceNo": "Frequency",
        "TotalAmount": "Monetary"
    },
    inplace=True
)


# =========================================
# CREATE FUTURE PURCHASE TARGET
# =========================================

future_customers = set(
    future_data["CustomerID"].unique()
)


customer_features["Purchase"] = (
    customer_features.index.isin(
        future_customers
    )
).astype(int)


# =========================================
# DISPLAY TARGET
# =========================================

print("\nTarget distribution:")

print(
    customer_features["Purchase"].value_counts()
)


print("\nTarget percentage:")

print(
    customer_features["Purchase"]
    .value_counts(normalize=True)
    * 100
)


# =========================================
# DISPLAY DATA
# =========================================

print("\nCustomer features:")

print(
    customer_features.head(10)
)


# =========================================
# SAVE DATASET
# =========================================

output_path = os.path.join(
    project_folder,
    "data",
    "supervised_customer_data.csv"
)


customer_features.to_csv(
    output_path
)


print("\n=========================================")
print("SUPERVISED DATASET CREATED")
print("=========================================")

print("\nSaved at:")

print(output_path)

print("\nDataset shape:")

print(customer_features.shape)