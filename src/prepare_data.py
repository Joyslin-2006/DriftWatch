
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load original dataset
df = pd.read_csv("data/creditcard.csv")

print("Original dataset:", df.shape)

# 2. Count exact duplicate rows
duplicate_count = df.duplicated().sum()
print("Duplicate rows:", duplicate_count)

# 3. Remove exact duplicates for modelling
df = df.drop_duplicates().copy()

print("Dataset after removing duplicates:", df.shape)

# 4. Separate features and target
X = df.drop(columns=["Class"])
y = df["Class"]

# 5. First split: 70% training, 30% remaining
X_train, X_remaining, y_train, y_remaining = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# 6. Second split: 15% reference, 15% test
X_reference, X_test, y_reference, y_test = train_test_split(
    X_remaining,
    y_remaining,
    test_size=0.50,
    random_state=42,
    stratify=y_remaining
)

# 7. Save the datasets
X_train.assign(Class=y_train).to_csv(
    "data/train.csv", index=False
)

X_reference.assign(Class=y_reference).to_csv(
    "data/reference.csv", index=False
)

X_test.assign(Class=y_test).to_csv(
    "data/test.csv", index=False
)

# 8. Display results
print("\n--- FINAL DATA SPLITS ---")

for name, X_part, y_part in [
    ("Training", X_train, y_train),
    ("Reference", X_reference, y_reference),
    ("Test", X_test, y_test)
]:
    print(f"\n{name}: {len(X_part)} rows")
    print(y_part.value_counts())
    print("Fraud percentage:", round(y_part.mean() * 100, 4), "%")

print("\nAll datasets saved successfully!")
