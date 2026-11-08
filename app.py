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
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"],.stApp {font-family:'DM Sans',sans-serif;}
.stApp {background:linear-gradient(180deg,#f2f7fc 0%,#ffffff 55%,#f5f8fc 100%);}
.block-container {padding-top:1.5rem;max-width:1220px;padding-bottom:3rem;}
.hero {background:linear-gradient(115deg,#10253f 0%,#184d70 60%,#087e8b 100%);color:#fff;padding:38px 42px;border-radius:24px;box-shadow:0 18px 42px rgba(18,50,75,.16);margin-bottom:22px;position:relative;overflow:hidden;}
.hero:after {content:'';position:absolute;width:270px;height:270px;border:1px solid #ffffff2b;border-radius:50%;right:-70px;top:-125px;box-shadow:0 0 0 48px #ffffff0a,0 0 0 95px #ffffff08;pointer-events:none;}
.hero h1 {font-size:clamp(1.8rem,3.1vw,2.8rem);line-height:1.16;letter-spacing:-.045em;color:white;margin:10px 0;max-width:780px;font-weight:800;}
.hero p {color:#d5eef7;font-size:1rem;margin:0;max-width:760px;}
.eyebrow {color:#8ce5ed;letter-spacing:.14em;font-size:.74rem;font-weight:800;text-transform:uppercase;}
.pill {display:inline-block;background:#ffffff18;border:1px solid #ffffff4a;color:#fff;padding:7px 13px;border-radius:99px;margin-top:18px;font-size:.78rem;font-weight:600;}
.section-head {margin:30px 0 12px;font-size:1.28rem;font-weight:800;color:#172d45;letter-spacing:-.02em;}
.section-sub {color:#64748b;margin-top:-7px;margin-bottom:16px;font-size:.9rem;}
.info-card {background:white;border:1px solid #e1eaf1;border-radius:17px;padding:20px;min-height:111px;box-shadow:0 5px 18px rgba(25,55,85,.045);}
.info-card b {font-size:1.75rem;color:#145b76;letter-spacing:-.035em;}
.info-card span {display:block;color:#63758a;margin-top:4px;font-size:.85rem;}
[data-testid="stMetric"] {background:white;border:1px solid #e1eaf1;border-radius:15px;padding:16px 17px;box-shadow:0 4px 15px #15385b0b;}
[data-testid="stMetricValue"] {color:#124d70;font-weight:800;}
[data-testid="stFileUploader"] {background:white;border:1px solid #dbe6f0;border-radius:17px;padding:13px;}
[data-testid="stDataFrame"] {border-radius:14px;overflow:hidden;}
.stButton>button,.stDownloadButton>button {border-radius:11px;font-weight:700;}
@media(max-width:640px){.hero{padding:26px 22px}.block-container{padding-top:.7rem}}
</style>""", unsafe_allow_html=True)
st.markdown("""<div class="hero"><div class="eyebrow">BITS PILANI · WILP · M.TECH AIML</div>
<h1>Machine Learning<br>Assignment 2</h1>
<p>Classification Model Benchmark · Compare five trained algorithms and evaluate a labeled test dataset in one workspace.</p>
<span class="pill">5 CLASSIFIERS</span> &nbsp; <span class="pill">6 EVALUATION METRICS</span> &nbsp; <span class="pill">UCI PUBLIC DATASET</span></div>""", unsafe_allow_html=True)

st.markdown('<div class="section-head">Dataset overview</div><div class="section-sub">The public dataset used for training and testing these five models.</div>', unsafe_allow_html=True)
st.markdown('**Breast Cancer Wisconsin (Diagnostic)** · [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) · loaded via `sklearn.datasets.load_breast_cancer`')
a,b,c,d=st.columns(4)
for col,val,label in [(a,'569','Total instances'),(b,'30','Numerical features'),(c,'2','Benign / malignant classes'),(d,'80 / 20','Training / testing split')]:
    with col:
        st.markdown(f'<div class="info-card"><b>{val}</b><span>{label}</span></div>',unsafe_allow_html=True)
st.caption('Binary target: 0 = benign, 1 = malignant (remapped for this experiment). Stratified random split, random_state=42; held-out test set = 114 records.')
with st.expander('About the experiment and evaluation methodology'):
    st.write('**Objective:** Compare five classifiers using the same 30 numerical features extracted from cell-nucleus images.')
    st.write('**Algorithms:** Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, and RBF Support Vector Machine.')
    st.write('**Evaluation:** Accuracy, ROC AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient. Appropriate preprocessing is stored with each fitted model.')
    st.write('The uploaded test CSV is evaluated using previously saved trained models. No retraining occurs in this application.')

st.markdown('<div class="section-head">Model performance comparison</div><div class="section-sub">Side-by-side performance of all five models on the same held-out test data.</div>',unsafe_allow_html=True)
metric_file=BASE/'model_metrics.csv'
if metric_file.exists():
    comparison=pd.read_csv(metric_file)
    st.dataframe(comparison.style.format({m:'{:.4f}' for m in METRICS}).background_gradient(subset=['Accuracy','F1','MCC'],cmap='Blues'),use_container_width=True,hide_index=True)
    chosen_metric=st.selectbox('Metric to visualize',METRICS,index=4)
    st.bar_chart(comparison.set_index('Model')[[chosen_metric]],color='#158a9b',horizontal=True)
else:
    st.warning('model_metrics.csv is missing. Run train_models.py to generate the comparison results.')

st.markdown('<div class="section-head">Evaluate uploaded test CSV</div><div class="section-sub">Select a saved classifier and upload a labeled CSV to calculate evaluation metrics without retraining.</div>',unsafe_allow_html=True)
selected=st.selectbox('Choose a trained classifier',MODEL_NAMES)
upload=st.file_uploader('Upload labeled test CSV',type=['csv'],help='Include the original 30 feature columns and target column.')
with st.expander('Expected CSV format and sample test data'):
    st.write('Your CSV needs the 30 original feature columns and a binary `target` column (0 = benign, 1 = malignant).')
    if (BASE/'test_data.csv').exists():
        st.download_button('Download sample held-out test CSV',(BASE/'test_data.csv').read_bytes(),'test_data.csv','text/csv')
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
