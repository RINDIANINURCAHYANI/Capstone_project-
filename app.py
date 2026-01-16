from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

app = Flask(__name__)

# ===============================
# LOAD & PREPROCESS DATA
# ===============================
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

df.drop("customerID", axis=1, inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

X = df.drop("Churn", axis=1)
y = df["Churn"]

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

# ===============================
# PREPROCESSOR (FIX UTAMA)
# ===============================
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

model = XGBClassifier(
    learning_rate=0.1,
    max_depth=5,
    eval_metric="logloss",
    use_label_encoder=False
)

model.fit(X_train, y_train)

# ===============================
# ROUTE
# ===============================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None

    if request.method == "POST":
        input_dict = {}

        # ----- numeric -----
        for col in num_cols:
            input_dict[col] = float(request.form[col])

        # ----- categorical (NORMALISASI DI SINI) -----
        for col in cat_cols:
            val = request.form[col].strip().lower()

            if col == "gender":
                if val in ["f", "female"]:
                    val = "Female"
                elif val in ["m", "male"]:
                    val = "Male"

            input_dict[col] = val.capitalize()

        input_df = pd.DataFrame([input_dict])

        input_processed = preprocessor.transform(input_df)

        pred = model.predict(input_processed)[0]
        prob = model.predict_proba(input_processed)[0][1]

        prediction = "CHURN" if pred == 1 else "TIDAK CHURN"
        probability = round(prob * 100, 2)

        # ===============================
        # PLOT
        # ===============================
        os.makedirs("static/plots", exist_ok=True)

        cm = confusion_matrix(y_test, model.predict(X_test))
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")
        plt.title("Confusion Matrix")
        plt.savefig("static/plots/cm.png")
        plt.close()

        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.savefig("static/plots/roc.png")
        plt.close()

        plt.figure()
        sns.histplot(df["MonthlyCharges"], kde=True)
        plt.title("Distribusi Monthly Charges")
        plt.savefig("static/plots/monthly.png")
        plt.close()

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        num_cols=num_cols,
        cat_cols=cat_cols
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
