# ==========================================================
# PREDIKSI CHURN PELANGGAN TELEKOMUNIKASI
# Logistic Regression vs XGBoost
# ==========================================================

import warnings
warnings.filterwarnings('ignore')

# =======================
# IMPORT LIBRARY
# =======================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# seaborn sering berat di Windows + Py 3.11
try:
    import seaborn as sns
except:
    sns = None

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, f1_score, precision_score, recall_score,
    roc_curve, auc
)

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# =======================
# LOAD DATASET
# =======================
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

print("===== DATA INFO =====")
print(df.info())

# =======================
# DATA CLEANING
# =======================
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

df.drop(columns=['customerID'], inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# =======================
# EDA SINGKAT
# =======================
if sns:
    sns.countplot(x='Churn', data=df)
    plt.title("Distribusi Churn Pelanggan")
    plt.show()

# =======================
# FEATURE & TARGET
# =======================
X = df.drop('Churn', axis=1)
y = df['Churn']

categorical_cols = X.select_dtypes(include='object').columns
numerical_cols = X.select_dtypes(exclude='object').columns

# =======================
# PREPROCESSING
# =======================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ]
)

# =======================
# SPLIT DATA
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

# =======================
# HANDLE IMBALANCE (SMOTE)
# =======================
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_prep, y_train)

print("\nBefore SMOTE:", np.bincount(y_train))
print("After SMOTE :", np.bincount(y_train_res))

# =======================
# MODEL 1: LOGISTIC REGRESSION
# =======================
lr = LogisticRegression(max_iter=1000, n_jobs=1)
lr.fit(X_train_res, y_train_res)

y_pred_lr = lr.predict(X_test_prep)
y_prob_lr = lr.predict_proba(X_test_prep)[:, 1]

# =======================
# MODEL 2: XGBOOST
# =======================
xgb = XGBClassifier(
    learning_rate=0.1,
    max_depth=5,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    n_jobs=1
)

xgb.fit(X_train_res, y_train_res)
y_pred_xgb = xgb.predict(X_test_prep)
y_prob_xgb = xgb.predict_proba(X_test_prep)[:, 1]

# =======================
# EVALUATION FUNCTION
# =======================
def evaluate_model(name, y_true, y_pred, y_prob):
    print(f"\n===== {name} =====")
    print(classification_report(y_true, y_pred))
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1-Score :", f1_score(y_true, y_pred))
    print("ROC-AUC  :", roc_auc_score(y_true, y_prob))

evaluate_model("Logistic Regression", y_test, y_pred_lr, y_prob_lr)
evaluate_model("XGBoost", y_test, y_pred_xgb, y_prob_xgb)

# =======================
# CONFUSION MATRIX (XGBOOST)
# =======================
cm = confusion_matrix(y_test, y_pred_xgb)

plt.figure(figsize=(6, 4))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix - XGBoost')
plt.colorbar()
plt.xticks([0, 1], ['Tidak Churn', 'Churn'])
plt.yticks([0, 1], ['Tidak Churn', 'Churn'])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.show()

# =======================
# ROC CURVE (XGBOOST)
# =======================
fpr, tpr, _ = roc_curve(y_test, y_prob_xgb)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - XGBoost')
plt.legend()
plt.show()

# =======================
# COMPARISON RESULT
# =======================
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'XGBoost'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_xgb)
    ],
    'F1-Score': [
        f1_score(y_test, y_pred_lr),
        f1_score(y_test, y_pred_xgb)
    ],
    'ROC-AUC': [
        roc_auc_score(y_test, y_prob_lr),
        roc_auc_score(y_test, y_prob_xgb)
    ]
})

print("\n===== MODEL COMPARISON =====")
print(results)
