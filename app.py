# Loan Default Prediction Notebook (with Streamlit Dashboard)

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import streamlit as st
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings("ignore")

# 2. Load Data
df = pd.read_csv("DOC-20250712-WA0001.csv")
df['default_ind'] = df['default_ind'].fillna(0)
y = df['default_ind']
X = df.drop(columns=['id', 'default_ind', 'status', 'funded_at', 'defaulted_at', 'default_date_min', 'default_date_max'])

# 3. Define Column Types
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# 4. Preprocessing
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, numerical_features),
    ('cat', cat_pipeline, categorical_features)
])

# 5. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 6. Train Final Model (XGBoost)
final_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])
final_model.fit(X_train, y_train)

# 7. STREAMLIT DASHBOARD
# To run: streamlit run this_notebook.py
st.set_page_config(page_title="Loan Default Dashboard")
st.title("🔍 Loan Default Prediction Dashboard")

st.markdown("---")
st.subheader("📌 Select a sample loan to analyze")
index = st.slider("Select row index from test set:", min_value=0, max_value=len(X_test)-1, value=0)
sample = X_test.iloc[[index]]
pred_prob = final_model.predict_proba(sample)[0][1]
pred_class = final_model.predict(sample)[0]

st.markdown(f"### 🧮 Prediction: {'Default' if pred_class == 1 else 'No Default'}")
st.markdown(f"### 🔢 Probability of Default: **{pred_prob:.2%}**")

# Show SHAP explanation
with st.expander("🔍 See SHAP Explanation"):
    explainer = shap.Explainer(final_model.named_steps['classifier'])
    X_transformed = final_model.named_steps['preprocessor'].transform(X_test)
    shap_values = explainer(X_transformed[index:index+1])
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.plots.waterfall(shap_values[0], max_display=10)
    st.pyplot(bbox_inches='tight')

# Show data sample
with st.expander("📄 See input feature values"):
    st.dataframe(sample.T)

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, XGBoost and SHAP")
