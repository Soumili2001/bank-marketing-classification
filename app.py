import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Bank Marketing Classification",
    page_icon="🏦",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("🏦 Bank Marketing Classification")

st.write(
    "Interactive evaluation of six machine learning "
    "classification models for predicting whether a "
    "bank customer subscribes to a term deposit."
)

# =========================================================
# MODEL PATHS
# =========================================================

MODEL_PATHS = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Random Forest": "model/random_forest.joblib",
    "SVM": "model/svm.joblib"
}

# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():
    loaded_models = {}

    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            loaded_models[name] = joblib.load(path)

    return loaded_models


models = load_models()

if len(models) == 0:
    st.error(
        "No trained models were found. "
        "Please make sure the model folder contains "
        "the six .joblib files."
    )
    st.stop()

# =========================================================
# MODEL SELECTION
# =========================================================

st.sidebar.header("Model Selection")

selected_model_name = st.sidebar.selectbox(
    "Choose a classification model:",
    list(models.keys())
)

selected_model = models[selected_model_name]

# =========================================================
# FILE UPLOAD
# =========================================================

st.header("📂 Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)

# IMPORTANT: Stop here until a file is uploaded
if uploaded_file is None:
    st.info(
        "Please upload test_data.csv to evaluate "
        "the selected model."
    )
    st.stop()

# =========================================================
# READ DATA
# =========================================================

try:
    data = pd.read_csv(uploaded_file)

except Exception as e:
    st.error(f"Unable to read the CSV file: {e}")
    st.stop()

st.success("Test dataset uploaded successfully.")

# =========================================================
# DATASET PREVIEW
# =========================================================

st.subheader("📋 Dataset Preview")

st.dataframe(
    data.head(),
    width="stretch"
)

st.write(
    f"**Number of test records:** {len(data):,}"
)

# =========================================================
# VALIDATE TARGET
# =========================================================

if "y" not in data.columns:
    st.error(
        "The uploaded CSV must contain a 'y' target column."
    )
    st.stop()

# =========================================================
# SEPARATE FEATURES AND TARGET
# =========================================================

X_test_app = data.drop(
    "y",
    axis=1
)

y_test_app = data["y"].map({
    "no": 0,
    "yes": 1
})

if y_test_app.isnull().any():
    st.error(
        "The target column 'y' must contain only "
        "'yes' and 'no' values."
    )
    st.stop()

# =========================================================
# PREDICTION
# =========================================================

try:
    y_pred_app = selected_model.predict(
        X_test_app
    )

except Exception as e:
    st.error(
        f"Prediction failed: {e}"
    )
    st.stop()

# =========================================================
# AUC
# =========================================================

try:

    if hasattr(selected_model, "predict_proba"):

        y_score_app = (
            selected_model
            .predict_proba(X_test_app)[:, 1]
        )

    else:

        y_score_app = (
            selected_model
            .decision_function(X_test_app)
        )

    auc = roc_auc_score(
        y_test_app,
        y_score_app
    )

except Exception:
    auc = np.nan

# =========================================================
# METRICS
# =========================================================

accuracy = accuracy_score(
    y_test_app,
    y_pred_app
)

precision = precision_score(
    y_test_app,
    y_pred_app,
    zero_division=0
)

recall = recall_score(
    y_test_app,
    y_pred_app,
    zero_division=0
)

f1 = f1_score(
    y_test_app,
    y_pred_app,
    zero_division=0
)

mcc = matthews_corrcoef(
    y_test_app,
    y_pred_app
)

# =========================================================
# RESULTS
# =========================================================

st.header(
    f"📊 Results — {selected_model_name}"
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Accuracy",
        f"{accuracy:.4f}"
    )

with col2:
    st.metric(
        "AUC",
        f"{auc:.4f}"
    )

with col3:
    st.metric(
        "Precision",
        f"{precision:.4f}"
    )

col4, col5, col6 = st.columns(3)

with col4:
    st.metric(
        "Recall",
        f"{recall:.4f}"
    )

with col5:
    st.metric(
        "F1 Score",
        f"{f1:.4f}"
    )

with col6:
    st.metric(
        "MCC",
        f"{mcc:.4f}"
    )

# =========================================================
# CONFUSION MATRIX
# =========================================================

st.subheader("📊 Confusion Matrix")

cm = confusion_matrix(
    y_test_app,
    y_pred_app
)

fig, ax = plt.subplots(
    figsize=(5, 4)
)

im = ax.imshow(cm)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(["No", "Yes"])
ax.set_yticklabels(["No", "Yes"])

ax.set_title(
    f"Confusion Matrix - {selected_model_name}"
)

for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

fig.colorbar(im, ax=ax)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

st.subheader("📋 Classification Report")

report = classification_report(
    y_test_app,
    y_pred_app,
    target_names=["No", "Yes"],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report
).transpose()

st.dataframe(
    report_df.round(4),
    width="stretch"
)

# =========================================================
# PREDICTION SUMMARY
# =========================================================

st.subheader("📈 Prediction Summary")

prediction_counts = (
    pd.Series(y_pred_app)
    .map({
        0: "No",
        1: "Yes"
    })
    .value_counts()
)

st.bar_chart(
    prediction_counts
)

# =========================================================
# ADD PREDICTIONS TO DATA
# =========================================================

results_with_predictions = data.copy()

results_with_predictions["Predicted"] = (
    pd.Series(y_pred_app)
    .map({
        0: "no",
        1: "yes"
    })
    .values
)

# =========================================================
# ACTUAL YES CUSTOMERS
# =========================================================

st.subheader("✅ Customers Who Actually Subscribed")

actual_yes_customers = data[
    data["y"] == "yes"
].copy()

st.write(
    f"**{len(actual_yes_customers):,} customers actually "
    f"subscribed to the term deposit.**"
)

st.dataframe(
    actual_yes_customers,
    width="stretch",
    height=500
)

actual_yes_csv = actual_yes_customers.to_csv(
    index=False
)

st.download_button(
    label="📥 Download Actual YES Customers",
    data=actual_yes_csv,
    file_name="actual_yes_customers.csv",
    mime="text/csv"
)

# =========================================================
# PREDICTED YES CUSTOMERS
# =========================================================

st.subheader("🤖 Customers Predicted as YES")

predicted_yes_customers = results_with_predictions[
    results_with_predictions["Predicted"] == "yes"
].copy()

st.write(
    f"**{len(predicted_yes_customers):,} customers are "
    f"predicted as YES by {selected_model_name}.**"
)

st.dataframe(
    predicted_yes_customers,
    width="stretch",
    height=500
)

predicted_yes_csv = predicted_yes_customers.to_csv(
    index=False
)

st.download_button(
    label="📥 Download Predicted YES Customers",
    data=predicted_yes_csv,
    file_name=(
        f"{selected_model_name.replace(' ', '_').lower()}"
        "_predicted_yes.csv"
    ),
    mime="text/csv"
)

# =========================================================
# ACTUAL VS PREDICTED
# =========================================================

st.subheader("🔍 Actual vs Predicted Results")

comparison = results_with_predictions.copy()

st.dataframe(
    comparison,
    width="stretch",
    height=500
)

# =========================================================
# TRUE POSITIVES
# =========================================================

true_positives = comparison[
    (comparison["y"] == "yes") &
    (comparison["Predicted"] == "yes")
].copy()

st.write(
    f"✅ **True Positives:** {len(true_positives):,}"
)

# =========================================================
# FALSE NEGATIVES
# =========================================================

false_negatives = comparison[
    (comparison["y"] == "yes") &
    (comparison["Predicted"] == "no")
].copy()

st.write(
    f"❌ **False Negatives:** {len(false_negatives):,}"
)

# =========================================================
# FALSE POSITIVES
# =========================================================

false_positives = comparison[
    (comparison["y"] == "no") &
    (comparison["Predicted"] == "yes")
].copy()

st.write(
    f"⚠️ **False Positives:** {len(false_positives):,}"
)

# =========================================================
# ALL PREDICTIONS DOWNLOAD
# =========================================================

st.subheader("📥 Download All Predictions")

all_predictions_csv = (
    results_with_predictions.to_csv(
        index=False
    )
)

st.download_button(
    label="Download All Test Predictions",
    data=all_predictions_csv,
    file_name=(
        f"{selected_model_name.replace(' ', '_').lower()}"
        "_all_predictions.csv"
    ),
    mime="text/csv"
)

# =========================================================
# FINAL STATUS
# =========================================================

st.success(
    f"{selected_model_name} evaluation completed successfully."
)
