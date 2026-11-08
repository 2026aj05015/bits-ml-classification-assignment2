# Classification Model Lab — Assignment 2 learning reference

> **Academic integrity:** Study, run, modify and document your own work. The assignment explicitly prohibits direct AI-generated copy-paste submissions.

## a. Problem statement
Compare five supervised binary classifiers for predicting malignant (1) versus benign (0) breast tumor labels from image-derived measurements. This is a statistical ML demonstration, not a clinical tool.

## b. Dataset description
- Source: UCI Breast Cancer Wisconsin (Diagnostic), distributed via `sklearn.datasets.load_breast_cancer`.
- 569 observations, 30 numeric features, two classes.
- Stratified 80% training / 20% testing, `random_state=42`.
- Dataset target is remapped so **1 = malignant** and **0 = benign**.
- No test examples are used for model fitting. Standardization is inside training pipelines for Logistic Regression, KNN and SVM.

## c. GitHub repository link
**Replace with your actual public GitHub repository URL:** `https://github.com/YOUR_USERNAME/YOUR_REPO`

## d. Models used and metrics
Run `python train_models.py` in the BITS Virtual Lab. Copy the *verified* values from `model_metrics.csv` into the table below.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Support Vector Machine | | | | | | |

Metrics use malignant as the positive class; AUC uses predicted probabilities, not predicted labels.

## Model observations — complete after running your own experiment
| Model | Observation about model performance |
|---|---|
| Logistic Regression | Compare its precision/recall and AUC with SVM. |
| Decision Tree | Note interpretability and potential overfitting. |
| KNN | Discuss sensitivity to feature scaling and neighbors. |
| Naive Bayes | Discuss the Gaussian conditional-independence assumption. |
| Support Vector Machine | Compare its nonlinear separation with logistic regression. |
| Overall winner | Select using observed F1, MCC and AUC, and justify the trade-off. |

## Run locally or on BITS Virtual Lab
```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```
Upload `test_data.csv` using the web app. The app loads the five saved `.joblib` models and **does not retrain** on uploaded data.

## Deploy
1. Run training and confirm the `model/` directory contains five `.joblib` files and `feature_columns.json`.
2. Push `app.py`, `train_models.py`, `requirements.txt`, `README.md`, `test_data.csv`, `model_metrics.csv`, and `model/` to your own GitHub repository.
3. Visit https://streamlit.io/cloud, choose New App, select your repo/main branch and `app.py`, then deploy.
4. Test CSV upload, model dropdown, metrics and confusion matrix on the live URL.

## Streamlit link
**Replace with your deployed URL:** `https://YOUR-APP.streamlit.app`

## BITS Virtual Lab proof
Take one genuine screenshot of the training script or running app inside BITS Virtual Lab; do not fabricate this evidence.

## Submission PDF order
1. Actual GitHub repository link.
2. Actual working Streamlit app link.
3. Genuine BITS Virtual Lab screenshot.
4. Your completed README content, including your verified comparison and personal observations.
