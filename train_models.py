"""Run in BITS Virtual Lab. Saves fitted pipelines and held-out test data."""
# Import data processing, model training, persistence, and scoring libraries.
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

# Use a project-relative directory for generated models and datasets.
ROOT = Path(__file__).resolve().parent
(ROOT / 'model').mkdir(exist_ok=True)
# Load the public Breast Cancer Wisconsin (Diagnostic) dataset with named columns.
data = load_breast_cancer(as_frame=True)
# Separate the 30 numerical predictors from the binary target.
X = data.data
# Positive class = malignant (1), negative class = benign (0)
# Remap labels: malignant is the positive class (1), benign is negative (0).
y = (data.target == 0).astype(int)
# Create a reproducible, stratified 80/20 split that preserves class proportions.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
# Configure the five classifiers. Scaling is included inside pipelines where needed.
# Fitting the scaler inside a pipeline prevents test-set information leaking into training.
models = {
    'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=42)),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=42),
    'KNN': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, random_state=42)),
}
# Collect comparable test-set scores for each classifier.
results = []
# Train and evaluate each model using the same training/test partitions.
for name, estimator in models.items():
# Fit the estimator or preprocessing-plus-estimator pipeline on training data only.
    estimator.fit(X_train, y_train)
# Generate held-out test predictions and positive-class probabilities.
    predictions = estimator.predict(X_test)
    scores = estimator.predict_proba(X_test)[:, list(estimator.classes_).index(1)]
# Compute accuracy, AUC, precision, recall, F1, and Matthews correlation coefficient.
    results.append({'Model': name, 'Accuracy': accuracy_score(y_test, predictions),
        'AUC': roc_auc_score(y_test, scores), 'Precision': precision_score(y_test, predictions, zero_division=0),
        'Recall': recall_score(y_test, predictions, zero_division=0),
        'F1': f1_score(y_test, predictions, zero_division=0), 'MCC': matthews_corrcoef(y_test, predictions)})
# Save the entire fitted model/pipeline for reuse in the Streamlit app.
    joblib.dump(estimator, ROOT / 'model' / (name.lower().replace(' ', '_') + '.joblib'))

# Export untouched held-out feature rows together with their true labels.
test = X_test.copy()
test['target'] = y_test.to_numpy()
test.to_csv(ROOT / 'test_data.csv', index=False)
# Save the model comparison metrics for the dashboard benchmark table.
pd.DataFrame(results).to_csv(ROOT / 'model_metrics.csv', index=False)
# Save the expected input feature names and order for future CSV validation.
(ROOT / 'model' / 'feature_columns.json').write_text(json.dumps(list(X.columns), indent=2))
# Print a readable summary after training finishes.
print(pd.DataFrame(results).round(4).to_string(index=False))
print('Saved test_data.csv and five fitted models; target: 1=malignant, 0=benign')
