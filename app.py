"""Interactive evaluation of PRETRAINED models on held-out labeled test CSV."""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay, classification_report)

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title='Classification Model Lab', page_icon='🧪', layout='wide')
st.title('Classification Model Lab')
st.caption('Wisconsin Diagnostic Breast Cancer | 1 = malignant, 0 = benign | Educational demonstration only')
st.info('Upload a labeled test CSV with all 30 original feature columns and a target column. The saved model is loaded, never retrained.')
model_names = ['Logistic Regression', 'Decision Tree', 'KNN', 'Naive Bayes', 'Support Vector Machine']
selected = st.selectbox('Choose a trained classifier', model_names)
upload = st.file_uploader('Upload test CSV', type=['csv'])
with st.expander('Expected data format'):
    st.write('Use the test_data.csv produced by train_models.py. target is 1 for malignant and 0 for benign.')
    if (BASE / 'test_data.csv').exists():
        st.download_button('Download example held-out test data', (BASE / 'test_data.csv').read_bytes(), 'test_data.csv', 'text/csv')
if upload is not None:
    try:
        df = pd.read_csv(upload)
        feature_names = json.loads((BASE / 'model' / 'feature_columns.json').read_text())
        missing = [c for c in feature_names + ['target'] if c not in df.columns]
        if missing:
            st.error(f'Missing required columns: {missing}')
            st.stop()
        X = df.loc[:, feature_names].apply(pd.to_numeric, errors='raise')
        y = pd.to_numeric(df['target'], errors='raise')
        if X.isna().any().any() or y.isna().any():
            st.error('Missing values are not supported by these saved models.')
            st.stop()
        if not set(y.unique()).issubset({0, 1}) or len(y.unique()) != 2:
            st.error('The test CSV must contain both target classes, 0 and 1, for ROC AUC.')
            st.stop()
        model = joblib.load(BASE / 'model' / (selected.lower().replace(' ', '_') + '.joblib'))
        pred = model.predict(X)
        score = model.predict_proba(X)[:, list(model.classes_).index(1)]
        values = {
            'Accuracy': accuracy_score(y, pred), 'AUC': roc_auc_score(y, score),
            'Precision': precision_score(y, pred, zero_division=0),
            'Recall': recall_score(y, pred, zero_division=0),
            'F1': f1_score(y, pred, zero_division=0),
            'MCC': matthews_corrcoef(y, pred)
        }
        st.subheader(f'{selected} — Test Results')
        for col, (label, val) in zip(st.columns(6), values.items()):
            col.metric(label, f'{val:.4f}')
        left, right = st.columns(2)
        with left:
            st.subheader('Confusion matrix')
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(confusion_matrix(y, pred, labels=[0, 1]), display_labels=['Benign', 'Malignant']).plot(ax=ax, colorbar=False)
            st.pyplot(fig)
            plt.close(fig)
        with right:
            st.subheader('Classification report')
            st.code(classification_report(y, pred, target_names=['Benign', 'Malignant'], zero_division=0))
        st.dataframe(pd.DataFrame([values]).round(4), use_container_width=True)
    except Exception as exc:
        st.error(f'Could not evaluate uploaded CSV: {exc}')
else:
    st.warning('Upload test_data.csv to display evaluation results.')
st.caption('Models are pre-trained; do not use for clinical diagnosis.')
