"""BITS WILP ML Assignment 2: evaluation of five saved classifiers (no retraining)."""
# Import libraries for file paths, model loading, tables, charts, and evaluation.
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix,
                             ConfusionMatrixDisplay, classification_report)

# Resolve files relative to this script so local and hosted deployments use the same paths.
BASE = Path(__file__).resolve().parent
MODEL_NAMES = ['Logistic Regression', 'Decision Tree', 'KNN', 'Naive Bayes', 'Support Vector Machine']
METRICS = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
# Set the browser title, icon, and wide layout before rendering page content.
st.set_page_config(page_title='ML Assignment 2 | Classification Benchmark', page_icon='📊', layout='wide')
# Customize the visual appearance of the Streamlit interface using CSS.
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
# Display the assignment banner and key experiment information.
st.markdown("""<div class="hero"><div class="eyebrow">BITS PILANI · WILP · M.TECH AIML</div>
<h1>Machine Learning<br>Assignment 2</h1>
<p>Classification Model Benchmark · Compare five trained algorithms and evaluate a labeled test dataset in one workspace.</p>
<span class="pill">5 CLASSIFIERS</span> &nbsp; <span class="pill">6 EVALUATION METRICS</span> &nbsp; <span class="pill">UCI PUBLIC DATASET</span></div>""", unsafe_allow_html=True)

# Summarize the dataset and class-label convention used by the trained models.
st.markdown('<div class="section-head">Dataset overview</div><div class="section-sub">The public dataset used for training and testing these five models.</div>', unsafe_allow_html=True)
st.markdown('**Breast Cancer Wisconsin (Diagnostic)** · [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) · loaded via `sklearn.datasets.load_breast_cancer`')
# Present four dataset facts in separate dashboard columns.
a,b,c,d=st.columns(4)
for col,val,label in [(a,'569','Total instances'),(b,'30','Numerical features'),(c,'2','Benign / malignant classes'),(d,'80 / 20','Training / testing split')]:
    with col:
        st.markdown(f'<div class="info-card"><b>{val}</b><span>{label}</span></div>',unsafe_allow_html=True)
st.caption('Binary target: 0 = benign, 1 = malignant (remapped for this experiment). Stratified random split, random_state=42; held-out test set = 114 records.')
# Provide optional methodological details without crowding the main dashboard.
st.markdown('### About the experiment and evaluation methodology')
with st.container(border=True):
    st.write('**Objective:** Compare five classifiers using the same 30 numerical features extracted from cell-nucleus images.')
    st.write('**Algorithms:** Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, and RBF Support Vector Machine.')
    st.write('**Evaluation:** Accuracy, ROC AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient. Appropriate preprocessing is stored with each fitted model.')
    st.write('The uploaded test CSV is evaluated using previously saved trained models. No retraining occurs in this application.')

# Load the saved models and calculate evaluation scores from actual predictions.
# No metric values are read from a precomputed model_metrics.csv file.
@st.cache_resource
# Cache fitted models to avoid reloading them on every Streamlit interaction.
def load_saved_models():
    loaded = {}
    for model_name in MODEL_NAMES:
        filename = model_name.lower().replace(' ', '_') + '.joblib'
        loaded[model_name] = joblib.load(BASE / 'model' / filename)
    return loaded

# Read and validate a labeled CSV before running inference.
def prepare_evaluation_data(dataset):
    feature_names = json.loads((BASE / 'model' / 'feature_columns.json').read_text())
    missing = [column for column in feature_names + ['target'] if column not in dataset.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    features = dataset[feature_names].apply(pd.to_numeric, errors='raise')
    targets = pd.to_numeric(dataset['target'], errors='raise')
    if features.isna().any().any() or targets.isna().any():
        raise ValueError('Missing feature or target values are not supported.')
    if not set(targets.unique()).issubset({0, 1}) or len(targets.unique()) != 2:
        raise ValueError('ROC AUC requires both target classes (0 and 1).')
    return features, targets

# Evaluate every classifier on the same labeled rows for a fair comparison.
def evaluate_all_models(dataset):
    features, targets = prepare_evaluation_data(dataset)
    saved_models = load_saved_models()
    records = []
    details = {}
    for model_name, estimator in saved_models.items():
        predicted = estimator.predict(features)
        positive_class_index = list(estimator.classes_).index(1)
        probability = estimator.predict_proba(features)[:, positive_class_index]
        scores = {
            'Accuracy': accuracy_score(targets, predicted),
            'AUC': roc_auc_score(targets, probability),
            'Precision': precision_score(targets, predicted, zero_division=0),
            'Recall': recall_score(targets, predicted, zero_division=0),
            'F1': f1_score(targets, predicted, zero_division=0),
            'MCC': matthews_corrcoef(targets, predicted),
        }
        records.append({'Model': model_name, **scores})
        details[model_name] = {'predictions': predicted, 'scores': scores}
    return pd.DataFrame(records), details, targets

# Display the initial comparison using the bundled held-out test set.
st.markdown('<div class="section-head">Model performance comparison</div><div class="section-sub">Metrics calculated from predictions on the bundled held-out test dataset.</div>', unsafe_allow_html=True)
try:
    benchmark_data = pd.read_csv(BASE / 'test_data.csv')
    benchmark_comparison, _, _ = evaluate_all_models(benchmark_data)
    st.dataframe(benchmark_comparison.style.format({m: '{:.4f}' for m in METRICS}).background_gradient(subset=['Accuracy', 'F1', 'MCC'], cmap='Blues'), use_container_width=True, hide_index=True)
    st.markdown('**All six metrics across five classifiers**')
    st.bar_chart(benchmark_comparison.set_index('Model')[METRICS], horizontal=True)
except Exception as exc:
    st.error(f'Could not calculate the bundled test-set comparison: {exc}')

# The uploader evaluates all five models, while a single selector controls detailed results.
st.markdown('<div class="section-head">Evaluate uploaded test CSV</div><div class="section-sub">Upload a labeled CSV and select a classifier to inspect its evaluation results.</div>', unsafe_allow_html=True)
upload = st.file_uploader('Upload labeled test CSV', type=['csv'], help='Include the original 30 feature columns and target column.')
with st.expander('Expected CSV format and sample test data'):
    st.write('Your CSV needs the 30 original feature columns and a binary `target` column (0 = benign, 1 = malignant).')
    if (BASE / 'test_data.csv').exists():
        st.download_button('Download sample held-out test CSV', (BASE / 'test_data.csv').read_bytes(), 'test_data.csv', 'text/csv')

if upload is None:
    st.info('Upload test_data.csv to evaluate the classifiers and select a model to inspect its results.')
else:
    try:
        uploaded_data = pd.read_csv(upload)
        uploaded_comparison, model_details, actual_targets = evaluate_all_models(uploaded_data)
        st.success(f'Evaluated {len(uploaded_data)} rows with all five saved models.')
        st.dataframe(uploaded_comparison.style.format({m: '{:.4f}' for m in METRICS}), use_container_width=True, hide_index=True)
        # Select one trained classifier to inspect; the comparison table still includes all five.
        model_name = st.selectbox('Choose a trained classifier', MODEL_NAMES)
        # Display the selected model's six metrics and diagnostic outputs.
        st.subheader(model_name)
        model_results = model_details[model_name]
        metric_columns = st.columns(6)
        for metric_column, (metric_name, metric_value) in zip(metric_columns, model_results['scores'].items()):
            metric_column.metric(metric_name, f'{metric_value:.4f}')
        left, right = st.columns(2)
        with left:
            st.markdown('**Confusion matrix**')
            figure, axis = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay(
                confusion_matrix(actual_targets, model_results['predictions'], labels=[0, 1]),
                display_labels=['Benign', 'Malignant'],
            ).plot(ax=axis, colorbar=False, cmap='Blues')
            st.pyplot(figure)
            plt.close(figure)
        with right:
            st.markdown('**Classification report**')
            st.code(classification_report(actual_targets, model_results['predictions'], labels=[0, 1], target_names=['Benign', 'Malignant'], zero_division=0))
        st.divider()
        st.download_button('Download evaluated metrics (CSV)', uploaded_comparison.to_csv(index=False), 'evaluation_metrics.csv', 'text/csv')
    except Exception as exc:
        st.error(f'Could not evaluate uploaded CSV: {exc}')

st.caption('Academic demonstration only · Not intended for medical diagnosis · Dataset: UCI Breast Cancer Wisconsin (Diagnostic)')
