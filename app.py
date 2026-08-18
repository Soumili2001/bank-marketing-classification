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


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Bank Marketing Classification",
    page_icon="🏦",
    layout="wide"
)


# ---------------------------------------------------------
# Title
# ---------------------------------------------------------

st.title("🏦 Bank Marketing Classification")
st.write(
    "Interactive comparison of six machine learning "
    "classification models for predicting whether a "
    "customer subscribes to a term deposit."
)


# ---------------------------------------------------------
# Model paths
# ---------------------------------------------------------

MODEL_PATHS = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Random Forest": "model/random_forest.joblib",
    "SVM": "model/svm.joblib"
}


# ---------------------------------------------------------
# Load models
# ---------------------------------------------------------

@st.cache_resource
def load_models():

    loaded_models = {}

    for name, path in MODEL_PATHS.items():

        if os.path.exists(path):
            loaded_models[name] = joblib.load(path)

    return loaded_models


models = load_models()


# ---------------------------------------------------------
# Check models
# ---------------------------------------------------------

if len(models) == 0:

    st.error(
        "No trained models were found. "
        "Please make sure the model folder contains "
        "the six .joblib files."
    )

    st.stop()


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.header("Model Selection")

selected_model_name = st.sidebar.selectbox(
    "Choose a classification model:",
    list(models.keys())
)

selected_model = models[selected_model_name]


# ---------------------------------------------------------
# Dataset upload
# ---------------------------------------------------------

st.header("📂 Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)


if uploaded_file is None:

    st.info(
        "Please upload the test_data.csv file "
        "to evaluate the selected model."
    )

    st.stop()


# ---------------------------------------------------------
# Read uploaded data
# ---------------------------------------------------------

try:

    data = pd.read_csv(uploaded_file)

except Exception as e:

    st.error(f"Unable to read the CSV file: {e}")

    st.stop()


st.success("Test dataset uploaded successfully.")

st.subheader("Dataset Preview")

st.dataframe(
    data.head(),
    use_container_width=True
)


# ---------------------------------------------------------
# Validate target column
# ---------------------------------------------------------

if "y" not in data.columns:

    st.error(
        "The uploaded CSV must contain a 'y' target column."
    )

    st.stop()


# ---------------------------------------------------------
# Separate features and target
# ---------------------------------------------------------

X_test_app = data.drop("y", axis=1)

y_test_app = data["y"].map({
    "no": 0,
    "yes": 1
})


# Check target conversion

if y_test_app.isnull().any():

    st.error(
        "The target column 'y' must contain only "
        "'yes' and 'no' values."
    )

    st.stop()


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

try:

    y_pred_app = selected_model.predict(X_test_app)

except Exception as e:

    st.error(
        f"Prediction failed: {e}"
    )

    st.stop()


# ---------------------------------------------------------
# AUC score
# ---------------------------------------------------------

try:

    if hasattr(selected_model, "predict_proba"):

        y_score_app = selected_model.predict_proba(
            X_test_app
        )[:, 1]

    else:

        y_score_app = selected_model.decision_function(
            X_test_app
        )

    auc = roc_auc_score(
        y_test_app,
        y_score_app
    )

except Exception:

    auc = np.nan


# ---------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Display selected model
# ---------------------------------------------------------

st.header(
    f"📊 Results — {selected_model_name}"
)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------

st.subheader("Confusion Matrix")

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

st.pyplot(fig)


# ---------------------------------------------------------
# Classification Report
# ---------------------------------------------------------

st.subheader("Classification Report")

report = classification_report(
    y_test_app,
    y_pred_app,
    target_names=["No", "Yes"],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True
)


# ---------------------------------------------------------
# Prediction summary
# ---------------------------------------------------------

st.subheader("Prediction Summary")

prediction_counts = pd.Series(
    y_pred_app
).map({
    0: "No",
    1: "Yes"
}).value_counts()

st.bar_chart(prediction_counts)


st.success(
    f"{selected_model_name} evaluation completed successfully."
)