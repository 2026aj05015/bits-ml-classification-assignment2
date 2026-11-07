"""Run in BITS Virtual Lab. Saves fitted pipelines and held-out test data."""
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

ROOT = Path(__file__).resolve().parent
(ROOT / 'model').mkdir(exist_ok=True)
data = load_breast_cancer(as_frame=True)
X = data.data
# Positive class = malignant (1), negative class = benign (0)
y = (data.target == 0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
models = {
    'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=42)),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=42),
    'KNN': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, random_state=42)),
}
results = []
for name, estimator in models.items():
    estimator.fit(X_train, y_train)
    predictions = estimator.predict(X_test)
    scores = estimator.predict_proba(X_test)[:, list(estimator.classes_).index(1)]
    results.append({'Model': name, 'Accuracy': accuracy_score(y_test, predictions),
        'AUC': roc_auc_score(y_test, scores), 'Precision': precision_score(y_test, predictions, zero_division=0),
        'Recall': recall_score(y_test, predictions, zero_division=0),
        'F1': f1_score(y_test, predictions, zero_division=0), 'MCC': matthews_corrcoef(y_test, predictions)})
    joblib.dump(estimator, ROOT / 'model' / (name.lower().replace(' ', '_') + '.joblib'))

test = X_test.copy()
test['target'] = y_test.to_numpy()
test.to_csv(ROOT / 'test_data.csv', index=False)
pd.DataFrame(results).to_csv(ROOT / 'model_metrics.csv', index=False)
(ROOT / 'model' / 'feature_columns.json').write_text(json.dumps(list(X.columns), indent=2))
print(pd.DataFrame(results).round(4).to_string(index=False))
print('Saved test_data.csv and five fitted models; target: 1=malignant, 0=benign')
