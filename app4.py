import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

# --- CHESS.COM THEME CONFIG ---
st.set_page_config(page_title="Checkmate Catalyst | Analysis", layout="wide")

# FIX: Changed unsafe_allow_index to unsafe_allow_html
st.markdown("""
    <style>
    .main { background-color: #312e2b; color: #ffffff; }
    .stMetric { background-color: #262421; padding: 15px; border-radius: 10px; border-left: 5px solid #81b64c; }
    div[data-testid="stSidebar"] { background-color: #262421; }
    .stButton>button { background-color: #81b64c; color: white; border-radius: 5px; width: 100%; border: none; }
    .stButton>button:hover { background-color: #a3d16a; color: white; }
    h1, h2, h3, p { color: #ffffff !important; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA ENGINE ---
@st.cache_data
def load_data():
    chess = fetch_openml(data_id=3, as_frame=True, parser='auto')
    df = chess.frame
    X = pd.get_dummies(df.drop('class', axis=1))
    y = df['class'].apply(lambda x: 1 if x == 'won' else 0)
    return X, y, list(X.columns)

X, y, feature_names = load_data()

# --- SIDEBAR ---
with st.sidebar:
    st.title("Settings")
    depth = st.slider("Engine Depth (Max Depth)", 1, 15, 5)
    crit = st.radio("Evaluation Metric", ["Gini", "Entropy"])

# --- MODEL LOGIC ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(max_depth=depth, criterion=crit.lower())
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)

# --- UI LAYOUT ---
col_board, col_stats = st.columns([1.5, 1])

with col_board:
    st.subheader("🌲 Decision Logic (The 'Engine' Path)")
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#312e2b')
    plot_tree(clf, max_depth=2, feature_names=feature_names, 
              class_names=['Draw', 'Win'], filled=True, rounded=True)
    st.pyplot(fig)

with col_stats:
    st.subheader("Game Analysis")
    st.metric(label="Evaluation Accuracy", value=f"{score:.1%}")
    
    st.write("### Top Winning Factors")
    importances = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=False).head(5)
    for feat, val in importances.items():
        st.write(f"**{feat.replace('_', ' ')}**")
        st.progress(float(val))

# --- FOOTER ---
st.divider()
# FIX: Changed unsafe_allow_index to unsafe_allow_html
st.markdown("""
<div style='text-align: center; color: #7d7a77;'>
    <p>Checkmate Catalyst | Powered by Scikit-Learn | Inspired by Unit III Decision Trees</p>
</div>
""", unsafe_allow_html=True)