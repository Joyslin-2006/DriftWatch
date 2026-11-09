
import pandas as pd

# Load the dataset
df = pd.read_csv("data/creditcard.csv")

# Basic dataset information
print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
print(df.isnull().sum().sum())

print("\n--- CLASS DISTRIBUTION ---")
print(df["Class"].value_counts())

print("\n--- CLASS PERCENTAGES ---")
print(df["Class"].value_counts(normalize=True) * 100)

print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())
