import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib

# Load dataset
data = pd.read_csv('data/heart.csv')

# Encode categorical features
data_encoded = pd.get_dummies(data, drop_first=True)

# Features and target
X = data_encoded.drop('HeartDisease', axis=1)
y = data_encoded['HeartDisease']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("\nModel Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, 'models/heart_model.pkl')
print("Model saved to models/heart_model.pkl")

# ---------- NEW: PREDICTION ON NEW PERSON DATA ----------

# Sample patient input (same order as encoded features)
sample_input = {
    'Age': 54,
    'RestingBP': 150,
    'Cholesterol': 195,
    'FastingBS': 0,
    'MaxHR': 122,
    'Oldpeak': 0.0,
    
    # One-hot encoded features (based on drop_first=True)
    'Sex_M': 1,
    'ChestPainType_ATA': 0,
    'ChestPainType_ASY': 0,
    'ChestPainType_NAP': 1,
    'RestingECG_Normal': 1,
    'RestingECG_ST': 0,
    'ExerciseAngina_Y': 0,
    'ST_Slope_Flat': 0,
    'ST_Slope_Up': 1
}

# Ensure all required columns are present
input_df = pd.DataFrame([sample_input])
missing_cols = [col for col in X.columns if col not in input_df.columns]
for col in missing_cols:
    input_df[col] = 0  # Fill missing columns with 0 (safe default)

# Align column order with training data
input_df = input_df[X.columns]

# Predict
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1] * 100

# Output
if prediction == 1:
    print(f"\n🔴 The model predicts the person is at **risk of heart disease** ({probability:.2f}% confidence).")
else:
    print(f"\n🟢 The model predicts the person is **not likely to have heart disease** ({100 - probability:.2f}% confidence).")
