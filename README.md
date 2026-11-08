# Machine Learning - Assignment 2

**Programme:** BITS Pilani WILP - M.Tech (AIML)  
**Course:** Machine Learning  
**Dataset:** Breast Cancer Wisconsin (Diagnostic)  
**GitHub:** https://github.com/2026aj05015/bits-ml-classification-assignment2  
**Live Streamlit app:** https://bits-ml-classification-assignment2.streamlit.app/

## a. Problem statement
Compare five supervised classification algorithms for distinguishing malignant from benign breast tumor samples using numerical cell-nucleus features. Evaluate all models on the same held-out test set and present their results through a deployed Streamlit application. This is an educational ML demonstration, not a clinical diagnostic tool.

## b. Dataset description
- **Source:** UCI Machine Learning Repository, Breast Cancer Wisconsin (Diagnostic): https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- **Loading method:** `sklearn.datasets.load_breast_cancer`.
- **Dataset size:** 569 instances, 30 numerical features, two classes (benign and malignant).
- **Experiment encoding:** 0 = benign; 1 = malignant (remapped from the source encoding).
- **Split:** Stratified 80% training and 20% testing; `random_state=42`; 114 held-out test records.
- **Preprocessing:** Standardization is applied within training pipelines for Logistic Regression, KNN, and SVM. Test data are not used to fit models.

## c. GitHub repository link
https://github.com/2026aj05015/bits-ml-classification-assignment2

## d. Models used and evaluation metrics
All five classifiers were evaluated on the same 114-record test set. Metrics use malignant as the positive class; AUC is calculated from model scores/probabilities.

| ML model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9035 | 0.9471 | 0.9429 | 0.7857 | 0.8571 | 0.7926 |
| KNN | 0.9561 | 0.9825 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

## Model performance observations

| Model | Observation |
|---|---|
| Logistic Regression | Highest AUC (0.9960), with 96.49% accuracy and 92.86% malignant recall. Its strong ranking ability is useful when choosing different classification thresholds. |
| Decision Tree | Lowest F1 (0.8571), MCC (0.7926) and malignant recall (0.7857). The single tree is interpretable, but it misses 9 of 42 malignant cases in this test set. |
| KNN | Competitive accuracy (95.61%) and F1 (0.9383), with malignant recall of 90.48%. Distance-based classification benefits from standardized features. |
| Naive Bayes | Perfect malignant precision (1.0000) in this test split, but recall is 83.33%, meaning 7 of 42 malignant cases are missed. Its independence assumption may limit performance. |
| Support Vector Machine | Best accuracy (97.37%), F1 (0.9630), and MCC (0.9442), with precision 1.0000 and malignant recall 92.86%. It is the overall winner by these balanced performance metrics. |
| **Overall winner** | **Support Vector Machine** leads in accuracy, F1 and MCC on the held-out test split. Logistic Regression leads in AUC. Because false negatives are important in this context, the 92.86% malignant recall of both SVM and Logistic Regression should also be considered. |

## Streamlit application and evaluation
- Single-page interface showing the dataset summary, model comparison table and interactive comparison chart.
- User selects one of five saved classifiers and uploads a labeled test CSV containing the required 30 features and the target column.
- The application loads saved `.joblib` models, calculates Accuracy, AUC, Precision, Recall, F1 and MCC, and displays a confusion matrix and classification report.
- Uploaded test data are evaluated **without retraining**.

## Repository contents
- `app.py` - Streamlit interface and inference/evaluation.
- `train_models.py` - training and test split workflow.
- `requirements.txt` - Python dependencies.
- `README.md` - project description and results.
- `test_data.csv` - labeled held-out test data.
- `model_metrics.csv` - model comparison metrics.
- `model/` - five `.joblib` classifiers and `feature_columns.json`.

## Reproduce the experiment
```bash
python -m pip install -r requirements.txt
python train_models.py
python -m streamlit run app.py
```

## Deployment and submission
- **GitHub:** https://github.com/2026aj05015/bits-ml-classification-assignment2
- **Streamlit Community Cloud:** https://bits-ml-classification-assignment2.streamlit.app/
- **BITS Virtual Lab:** The deployed Streamlit application was accessed and evaluated within the BITS Virtual Lab environment on 08-Nov-2026. Screenshots show the lab desktop/browser, dataset overview, five-model comparison, and uploaded-test evaluation for multiple classifiers. The screenshots document web-app evaluation in the lab; they do not independently show local execution of `train_models.py` inside the lab.
- **Final submission:** One PDF containing the GitHub link, live Streamlit link, BITS Virtual Lab screenshots, and complete README content, in that order.

## Limitations
The reported scores come from one held-out split of a small public dataset. Results may vary with random seed, tuning and validation strategy; they do not establish clinical suitability.
