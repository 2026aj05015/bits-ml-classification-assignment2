"""BITS WILP ML Assignment 2: evaluation of five saved classifiers (no retraining)."""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix,
                             ConfusionMatrixDisplay, classification_report)

BASE = Path(__file__).resolve().parent
MODEL_NAMES = ['Logistic Regression', 'Decision Tree', 'KNN', 'Naive Bayes', 'Support Vector Machine']
METRICS = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
st.set_page_config(page_title='ML Assignment 2 | Classification Benchmark', page_icon='📊', layout='wide')
st.markdown('''<style>
.block-container {padding-top:1.5rem; max-width:1250px;}
.hero {background:linear-gradient(110deg,#122e43,#176f75); color:white; padding:27px 32px; border-radius:18px; margin-bottom:18px;}
.hero h1 {font-size:2rem; margin:0 0 8px; color:white;}
.hero p {color:#e0f5f3; margin:0; font-size:1rem;}
.smalltag {display:inline-block; background:#ffffff24; border:1px solid #ffffff40; border-radius:20px; padding:5px 12px; margin-bottom:12px; font-size:.78rem;}
.info-card {border:1px solid #d9e6e6; border-radius:12px; padding:16px; min-height:110px;}
.info-card b {font-size:1.5rem; color:#16777b;}
.info-card span {display:block; color:#65727c; margin-top:5px;}
</style>''', unsafe_allow_html=True)
st.markdown('''<div class="hero"><div class="smalltag">BITS Pilani WILP · M.Tech (AIML) · Machine Learning</div>
<h1>Assignment 2 — Classification Model Benchmark</h1>
<p>Five supervised learning algorithms · Public UCI dataset · Six evaluation metrics · Interactive test-set evaluation</p></div>''', unsafe_allow_html=True)

st.subheader('Dataset used for this assignment')
st.markdown('**Breast Cancer Wisconsin (Diagnostic)** — UCI Machine Learning Repository. '
            '[Original dataset and citation](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic). '
            'Loaded for training through `sklearn.datasets.load_breast_cancer`.')
a,b,c,d = st.columns(4)
for col, val, label in [(a,'569','Total observations'),(b,'30','Numerical features'),(c,'2','Classes: benign / malignant'),(d,'80 / 20','Train / test split')]:
    with col:
        st.markdown(f'<div class="info-card"><b>{val}</b><span>{label}</span></div>', unsafe_allow_html=True)
st.caption('Prediction target: 0 = benign; 1 = malignant. The original scikit-learn target encoding is remapped for this experiment. Stratified hold-out split, random_state=42; 114 test samples.')

with st.expander('Research problem and methodology'):
    st.write('**Problem:** Compare five supervised classifiers for distinguishing malignant and benign samples using 30 numerical cell-nucleus features.')
    st.write('**Models:** Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, and RBF Support Vector Machine.')
    st.write('**Protocol:** Train on 80% of the public dataset, hold out 20% for testing, and calculate Accuracy, ROC AUC, Precision, Recall, F1, and Matthews Correlation Coefficient. Preprocessing for relevant models is saved within fitted pipelines.')
    st.info('The uploaded test CSV is evaluated against previously saved fitted models. No fitting or retraining occurs in this web app.')

comparison_tab, evaluate_tab = st.tabs(['📊 Model comparison', '🧪 Evaluate uploaded test CSV'])
with comparison_tab:
    metric_file = BASE / 'model_metrics.csv'
    if metric_file.exists():
        comparison = pd.read_csv(metric_file)
        st.subheader('Five-model performance comparison')
        st.dataframe(comparison.style.format({m:'{:.4f}' for m in METRICS}), use_container_width=True, hide_index=True)
        st.caption('These are the recorded results for the saved held-out test split; not a new experiment on each page load.')
        chosen_metric = st.selectbox('Compare models by metric', METRICS, index=4)
        st.bar_chart(comparison.set_index('Model')[[chosen_metric]])
        best = comparison.loc[comparison['F1'].idxmax()]
        st.success(f"Highest F1 in this experiment: {best['Model']} (F1 = {best['F1']:.4f}). Compare AUC and other metrics before deciding which model is preferable.")
        st.markdown('**Observations to review and explain in your own README:**')
        st.write('Logistic Regression achieves the highest AUC; SVM leads on accuracy, F1, and MCC in this split. Decision Tree has lower recall, KNN performs competitively, and Gaussian Naive Bayes has high precision but lower recall. Discuss why these trade-offs matter.')
    else:
        st.warning('model_metrics.csv not found. Run train_models.py and commit the generated file.')

with evaluate_tab:
    selected = st.selectbox('Choose a trained classifier', MODEL_NAMES)
    upload = st.file_uploader('Upload labeled test CSV', type=['csv'], help='Use test_data.csv from the project; include all 30 features and the target column.')
    with st.expander('Expected data format / download test CSV'):
        st.write('30 feature columns with their original names plus a binary `target` column (0=benign, 1=malignant).')
        if (BASE / 'test_data.csv').exists():
            st.download_button('Download example held-out test CSV', (BASE / 'test_data.csv').read_bytes(), 'test_data.csv', 'text/csv')
    if upload is None:
        st.info('Upload test_data.csv above to see model metrics, confusion matrix, and classification report.')
    else:
        try:
            df = pd.read_csv(upload)
            feature_names = json.loads((BASE / 'model' / 'feature_columns.json').read_text())
            missing = [name for name in feature_names + ['target'] if name not in df.columns]
            if missing:
                st.error(f'Missing required columns: {missing}')
                st.stop()
            X = df[feature_names].apply(pd.to_numeric, errors='raise')
            y = pd.to_numeric(df['target'], errors='raise')
            if X.isna().any().any() or y.isna().any():
                st.error('Missing feature or target values are not supported.')
                st.stop()
            if not set(y.unique()).issubset({0, 1}) or len(y.unique()) != 2:
                st.error('For ROC AUC, upload a labeled test file containing both classes 0 and 1.')
                st.stop()
            model_path = BASE / 'model' / (selected.lower().replace(' ', '_') + '.joblib')
            model = joblib.load(model_path)
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, list(model.classes_).index(1)]
            results = {'Accuracy':accuracy_score(y,predictions), 'AUC':roc_auc_score(y,probabilities),
                       'Precision':precision_score(y,predictions,zero_division=0),
                       'Recall':recall_score(y,predictions,zero_division=0),
                       'F1':f1_score(y,predictions,zero_division=0),
                       'MCC':matthews_corrcoef(y,predictions)}
            st.subheader(f'{selected} — test results')
            st.caption(f'Evaluated {len(df)} uploaded samples using saved model: {model_path.name}')
            cols = st.columns(6)
            for col,(label,value) in zip(cols,results.items()):
                col.metric(label,f'{value:.4f}')
            left,right = st.columns(2)
            with left:
                st.subheader('Confusion matrix')
                fig,ax = plt.subplots(figsize=(5,4))
                ConfusionMatrixDisplay(confusion_matrix(y,predictions,labels=[0,1]),display_labels=['Benign','Malignant']).plot(ax=ax,colorbar=False,cmap='Blues')
                st.pyplot(fig)
                plt.close(fig)
            with right:
                st.subheader('Classification report')
                st.code(classification_report(y,predictions,target_names=['Benign','Malignant'],zero_division=0))
            st.download_button('Download evaluated metrics (CSV)',pd.DataFrame([{'Model':selected,**results}]).to_csv(index=False),'evaluation_metrics.csv','text/csv')
        except Exception as exc:
            st.error(f'Could not evaluate uploaded CSV: {exc}')

st.divider()
st.caption('Academic demonstration only · Not intended for medical diagnosis · Dataset: UCI Breast Cancer Wisconsin (Diagnostic)')
