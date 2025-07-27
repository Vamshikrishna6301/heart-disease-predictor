# 🫀 Heart Disease Prediction Model

This project uses **Logistic Regression** to predict whether a person is likely to develop **heart disease** based on clinical parameters like age, cholesterol, blood pressure, chest pain type, and more. It applies core **machine learning techniques** and data preprocessing to enable early disease risk assessment — a crucial application of AI in healthcare.

---

## 📊 Dataset

- Source: [Kaggle – Heart Disease Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- Total Records: 918
- Features include:
  - Age, Resting Blood Pressure, Cholesterol
  - Chest Pain Type
  - Maximum Heart Rate Achieved (MaxHR)
  - Fasting Blood Sugar (FastingBS)
  - Exercise-Induced Angina
  - ST Segment Slope, etc.

---

## 🧠 Technologies Used

- Python 3.x
- Pandas, NumPy
- scikit-learn
- Matplotlib, Seaborn
- Joblib (for model saving)

---

## ⚙️ How to Run

### 1. 🔧 Clone the Repository

```bash
git clone https://github.com/<your-username>/heart-disease-predictor.git
cd heart-disease-predictor


Sample Output
Model Accuracy: 94.57%

Classification Report:
              precision    recall  f1-score   support
           0       0.93      0.91      0.92        93
           1       0.95      0.96      0.95        92

Confusion Matrix:
[[85  8]
 [ 4 88]]

🟢 The model predicts the person is not likely to have heart disease (87.53% confidence).
