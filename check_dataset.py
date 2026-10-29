
import pandas as pd

df = pd.read_csv("data/supervised_customer_data.csv")

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nPurchase class distribution:")
print(df["Purchase"].value_counts(dropna=False))

print("\nPurchase class percentages:")
print(
    (df["Purchase"].value_counts(normalize=True, dropna=False) * 100)
    .round(2)
)

print("\nFeature statistics:")
print(df[["Recency", "Frequency", "Monetary"]].describe())

print("\nDuplicate rows:", df.duplicated().sum())
