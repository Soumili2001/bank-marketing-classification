# Bank Marketing Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to develop and compare multiple machine learning classification models for predicting whether a customer will subscribe to a term deposit as a result of a bank marketing campaign.

The project implements six classification algorithms on the same Bank Marketing dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier
6. Support Vector Machine (SVM)

The models are evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

An interactive Streamlit web application is also developed to allow users to upload test data, select a machine learning model, and view its evaluation results, confusion matrix, and classification report.

---

## 2. Dataset Description

### Dataset Name

Bank Marketing Dataset

### Source

UCI Machine Learning Repository — Bank Marketing Dataset

### Objective

The dataset contains information related to direct marketing campaigns conducted by a Portuguese banking institution. The classification task is to predict whether a customer subscribed to a term deposit.

The target variable is:

* `yes` — Customer subscribed to a term deposit
* `no` — Customer did not subscribe to a term deposit

### Dataset Size

The dataset contains:

* **45,211 instances**
* **16 input features**
* **1 target variable**
* **Binary classification problem**
* **0 duplicate rows**

### Features

The 16 input features are:

* `age`
* `job`
* `marital`
* `education`
* `default`
* `balance`
* `housing`
* `loan`
* `contact`
* `day`
* `month`
* `duration`
* `campaign`
* `pdays`
* `previous`
* `poutcome`

The target variable is `y`.

### Class Distribution

The target distribution is:

| Class |  Count | Percentage |
| ----- | -----: | ---------: |
| No    | 39,922 |     88.30% |
| Yes   |  5,289 |     11.70% |

The dataset is therefore imbalanced, with significantly more `no` observations than `yes` observations. Because of this imbalance, multiple evaluation metrics such as Precision, Recall, F1 Score, AUC, and MCC were considered in addition to Accuracy.

### Preprocessing

The numerical features were standardized using `StandardScaler`.

The categorical features were converted into numerical representations using `OneHotEncoder`.

A Scikit-learn `ColumnTransformer` was used to apply the appropriate preprocessing to numerical and categorical variables.

The dataset was divided into:

* 80% training data
* 20% testing data

The split was performed using stratification to preserve the class distribution.

---

## 3. GitHub Repository Link

**GitHub Repository:**

https://github.com/Soumili2001/bank-marketing-classification

---

## 4. Models Comparison

Six classification models were implemented using the same dataset and the same train-test split.

The following table presents the evaluation results obtained on the test dataset.

| ML Model                     |   Accuracy |        AUC |  Precision | Recall |   F1 Score |        MCC |
| ---------------------------- | ---------: | ---------: | ---------: | -----: | ---------: | ---------: |
| Logistic Regression          |     0.9015 |     0.9055 |     0.6462 | 0.3488 |     0.4530 |     0.4275 |
| Decision Tree                |     0.8746 |     0.7015 |     0.4649 | 0.4754 |     0.4701 |     0.3990 |
| KNN                          |     0.8962 |     0.8277 |     0.5990 | 0.3403 |     0.4346 |     0.4001 |
| Naive Bayes                  |     0.8548 |     0.8101 |     0.4059 | 0.5198 |     0.4559 |     0.3774 |
| **Random Forest (Ensemble)** | **0.9045** | **0.9263** | **0.6506** | 0.3960 | **0.4924** | **0.4597** |
| SVM                          |     0.8932 |     0.9039 |     0.6503 | 0.1881 |     0.2918 |     0.3105 |

---

## 5. Observations on Model Performance

| ML Model                 | Observation About Model Performance                                                                                                                                                                                                                                     |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression      | Logistic Regression performed strongly, achieving 90.15% accuracy and an AUC of 0.9055. Its precision was 0.6462, but its recall was relatively low at 0.3488. It provided a strong baseline for the classification problem.                                            |
| Decision Tree            | Decision Tree achieved 87.46% accuracy and an AUC of 0.7015. Its recall of 0.4754 was higher than Logistic Regression and KNN, but its overall AUC and precision were lower.                                                                                            |
| KNN                      | KNN achieved 89.62% accuracy and an AUC of 0.8277. Its precision was 0.5990, while recall was 0.3403. Its F1 score of 0.4346 was lower than Logistic Regression and Random Forest.                                                                                      |
| Naive Bayes              | Naive Bayes had the lowest accuracy among the six models at 85.48%. However, it achieved the highest recall of 0.5198, meaning it identified a larger proportion of the positive `yes` cases. Its lower precision resulted in an F1 score of 0.4559.                    |
| Random Forest (Ensemble) | Random Forest produced the strongest overall performance. It achieved the highest accuracy (0.9045), highest AUC (0.9263), highest F1 score (0.4924), and highest MCC (0.4597). It provided a good balance between precision and recall compared with the other models. |
| SVM                      | SVM achieved 89.32% accuracy and an AUC of 0.9039. Its precision was high at 0.6503, but its recall was only 0.1881, resulting in the lowest F1 score among the six models.                                                                                             |

---

## Overall Winner

### Random Forest

Random Forest is the overall best-performing model for this dataset based on the combination of evaluation metrics.

It achieved:

* **Accuracy:** 0.9045
* **AUC:** 0.9263
* **Precision:** 0.6506
* **Recall:** 0.3960
* **F1 Score:** 0.4924
* **MCC:** 0.4597

Random Forest achieved the highest Accuracy, AUC, F1 Score, and MCC among the six implemented models.

Although Naive Bayes achieved the highest Recall (0.5198), its precision was considerably lower. Therefore, Random Forest provided a better overall balance between precision and recall.

---

## 6. Streamlit Application

An interactive Streamlit web application was developed to demonstrate the trained classification models

Streamlit App Link: https://bank-marketing-classification-h8axwnvb8ksoi8pipambtx.streamlit.app/

The application provides the following functionality:

### Dataset Upload

Users can upload the `test_data.csv` file through the Streamlit interface.

### Model Selection

Users can select one of the six trained classification models:

* Logistic Regression
* Decision Tree
* KNN
* Naive Bayes
* Random Forest
* SVM

### Evaluation Metrics

The application displays:

* Accuracy
* AUC
* Precision
* Recall
* F1 Score
* MCC

### Confusion Matrix

A confusion matrix is displayed for the selected model.

### Classification Report

The application also displays the classification report containing class-level precision, recall, F1 score, and support.

---

## 7. Project Structure

```text
Bank_Classification/
│
├── app.py
├── requirements.txt
├── README.md
├── bank-full.csv
├── test_data.csv
├── model_training.ipynb
│
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── random_forest.joblib
    └── svm.joblib
```

---

## 8. Requirements

The project uses the following Python libraries:

```text
streamlit
pandas
numpy
scikit-learn
matplotlib
joblib
```

All required dependencies are listed in `requirements.txt`.

---

## 9. Conclusion

This project demonstrates an end-to-end machine learning classification workflow, starting from dataset selection and preprocessing, followed by implementation and evaluation of six classification algorithms.

Among the evaluated models, Random Forest provided the best overall performance on the Bank Marketing test dataset. The project also demonstrates how trained machine learning models can be integrated into an interactive Streamlit application and prepared for deployment through Streamlit Community Cloud.
