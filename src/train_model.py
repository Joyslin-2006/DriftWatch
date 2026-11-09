
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)

# 1. Load training and test datasets
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# 2. Separate features and target
X_train = train_df.drop(columns=["Class"])
y_train = train_df["Class"]

X_test = test_df.drop(columns=["Class"])
y_test = test_df["Class"]

print("Training rows:", len(X_train))
print("Test rows:", len(X_test))

# 3. Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

# 4. Train model
print("\nTraining model...")
model.fit(X_train, y_train)

# 5. Generate predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 6. Evaluate model
print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred, zero_division=0))

print("\n--- CONFUSION MATRIX ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- ROC-AUC ---")
print(roc_auc_score(y_test, y_prob))

print("\n--- PR-AUC (Average Precision) ---")
print(average_precision_score(y_test, y_prob))

# 7. Save trained model
joblib.dump(model, "models/fraud_model.pkl")

print("\nModel saved successfully!")
