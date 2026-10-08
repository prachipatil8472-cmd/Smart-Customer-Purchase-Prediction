import pandas as pd
import os

# =========================================
# STEP 1: FIND PROJECT FOLDER
# =========================================

project_folder = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Dataset path
file_path = os.path.join(
    project_folder,
    "data",
    "Online Retail.xlsx"
)

print("Dataset path:")
print(file_path)


# =========================================
# STEP 2: LOAD DATASET
# =========================================

df = pd.read_excel(file_path)

print("\nDataset loaded successfully!")


# =========================================
# STEP 3: BASIC DATA INFORMATION
# =========================================

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# =========================================
# STEP 4: CHECK DUPLICATES
# =========================================

print("\nNumber of duplicate rows:")
print(df.duplicated().sum())


# =========================================
# STEP 5: REMOVE DUPLICATES
# =========================================

df = df.drop_duplicates()

print("\nShape after removing duplicates:")
print(df.shape)


# =========================================
# STEP 6: REMOVE MISSING CUSTOMER IDs
# =========================================

df = df.dropna(subset=["CustomerID"])

print("\nShape after removing missing CustomerID:")
print(df.shape)


# =========================================
# STEP 7: REMOVE INVALID QUANTITIES
# =========================================

df = df[df["Quantity"] > 0]

print("\nShape after removing invalid Quantity:")
print(df.shape)


# =========================================
# STEP 8: REMOVE INVALID PRICES
# =========================================

df = df[df["UnitPrice"] > 0]

print("\nShape after removing invalid UnitPrice:")
print(df.shape)


# =========================================
# STEP 9: CONVERT DATE
# =========================================

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])


# =========================================
# STEP 10: CREATE TOTAL AMOUNT
# =========================================

df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

print("\nTotalAmount feature created.")


# =========================================
# STEP 11: FINAL CLEANED DATA INFORMATION
# =========================================

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nFinal Columns:")
print(df.columns.tolist())

print("\nFinal Missing Values:")
print(df.isnull().sum())

print("\nCleaned Dataset:")
print(df.head())


# =========================================
# STEP 12: RFM FEATURE ENGINEERING
# =========================================

# Reference date
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

# Create RFM features
rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
    "InvoiceNo": "nunique",
    "TotalAmount": "sum"
})


# =========================================
# STEP 13: RENAME RFM COLUMNS
# =========================================

rfm.rename(columns={
    "InvoiceDate": "Recency",
    "InvoiceNo": "Frequency",
    "TotalAmount": "Monetary"
}, inplace=True)


# =========================================
# STEP 14: DISPLAY RFM DATA
# =========================================

print("\n=========================================")
print("RFM CUSTOMER DATA")
print("=========================================")

print("\nFirst 10 customers:")
print(rfm.head(10))

print("\nRFM Shape:")
print(rfm.shape)

print("\nRFM Information:")
print(rfm.info())

print("\nRFM Statistics:")
print(rfm.describe())


# =========================================
# STEP 15: CHECK RFM MISSING VALUES
# =========================================

print("\nRFM Missing Values:")
print(rfm.isnull().sum())


# =========================================
# STEP 16: SAVE CLEANED DATASET
# =========================================

cleaned_path = os.path.join(
    project_folder,
    "data",
    "cleaned_retail.csv"
)

df.to_csv(
    cleaned_path,
    index=False
)

print("\nCleaned dataset saved successfully!")
print(cleaned_path)


# =========================================
# STEP 17: SAVE RFM DATASET
# =========================================

rfm_path = os.path.join(
    project_folder,
    "data",
    "customer_rfm.csv"
)

rfm.to_csv(
    rfm_path,
    index=True
)

print("\nRFM dataset saved successfully!")
print(rfm_path)


# =========================================
# STEP 18: FINAL MESSAGE
# =========================================

print("\n=========================================")
print("DATA PREPROCESSING COMPLETED")
print("RFM FEATURE ENGINEERING COMPLETED")
print("=========================================")