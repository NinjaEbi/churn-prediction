import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
import json
import requests

# -----------------------------
# Page Config & Styling
# -----------------------------
st.set_page_config(page_title="Churn Prediction Pro", page_icon="📉", layout="wide")

st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #1f1c2c, #928DAB);
        color: #fff;
    }
    .title {
        font-size:42px !important;
        font-weight:800;
        text-align:center;
        color: #F39C12;
        margin-bottom:10px;
    }
    .subtitle {
        text-align:center;
        font-size:18px;
        margin-bottom:30px;
        color: #ECF0F1;
    }
    .metric-card {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        background-color: rgba(255,255,255,0.1);
        box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)


# -----------------------------
# Lottie Animation Loader
# -----------------------------
def load_lottie(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_url = "https://assets2.lottiefiles.com/packages/lf20_fyye8szy.json"
lottie_animation = load_lottie(lottie_url)

# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    le = LabelEncoder()
    for col in df.select_dtypes(include="object"):
        df[col] = le.fit_transform(df[col])
    return df


# -----------------------------
# Train model
# -----------------------------
@st.cache_resource
def train_model(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=150, random_state=42)
    rf.fit(X_train, y_train)

    return rf, scaler, X.columns, X_test, y_test


# -----------------------------
# App Layout
# -----------------------------
st.markdown('<div class="title">📉 Customer Churn Prediction Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">An AI-Powered Dashboard to Predict & Prevent Customer Loss</div>', unsafe_allow_html=True)

df = load_data()
model, scaler, feature_names, X_test, y_test = train_model(df)

# Navigation
menu = st.sidebar.radio("📂 Navigate", ["🏠 Home", "🔮 Prediction", "📊 Analytics", "ℹ️ About"])

# -----------------------------
# Home Page
# -----------------------------
if menu == "🏠 Home":
    st.write("### Welcome to Churn Prediction Pro")
    st.write("This dashboard helps businesses **predict churn**, analyze key drivers, and take proactive actions to improve retention.")
    
    if lottie_animation:
        from streamlit_lottie import st_lottie
        st_lottie(lottie_animation, height=300, key="churn")

    col1, col2, col3 = st.columns(3)
    churn_rate = df["Churn"].mean() * 100
    roc_auc = roc_auc_score(y_test, model.predict(X_test))
    col1.markdown(f"<div class='metric-card'><h3>Churn Rate</h3><h2>{churn_rate:.1f}%</h2></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><h3>ROC-AUC</h3><h2>{roc_auc:.2f}</h2></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-card'><h3>Dataset Size</h3><h2>{len(df):,}</h2></div>", unsafe_allow_html=True)


# -----------------------------
# Prediction Page
# -----------------------------
elif menu == "🔮 Prediction":
    st.subheader("Enter Customer Details")
    customer_data = {}
    for col in feature_names:
        if df[col].nunique() <= 10:
            customer_data[col] = st.selectbox(col, sorted(df[col].unique()))
        else:
            customer_data[col] = st.slider(col, int(df[col].min()), int(df[col].max()), int(df[col].median()))

    customer_df = pd.DataFrame([customer_data])
    customer_scaled = scaler.transform(customer_df)

    if st.button("🔮 Predict Now"):
        prediction = model.predict(customer_scaled)[0]
        prob = model.predict_proba(customer_scaled)[0][1]

        st.markdown("### 📋 Customer Profile")
        st.dataframe(customer_df)

        if prediction == 1:
            st.error(f"⚠ High Risk! This customer is **LIKELY to CHURN** (probability {prob:.2f})")
        else:
            st.success(f"✅ Safe! This customer is **NOT likely to churn** (probability {1-prob:.2f})")


# -----------------------------
# Analytics Page
# -----------------------------
elif menu == "📊 Analytics":
    st.subheader("Churn Analytics")

    # Distribution Plot
    fig = px.pie(df, names="Churn", title="Churn Distribution", color_discrete_sequence=["#27AE60", "#E74C3C"])
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    feat_df = feat_df.sort_values(by="Importance", ascending=False)

    fig = px.bar(feat_df.head(10), x="Importance", y="Feature", orientation="h", title="Top 10 Features Driving Churn")
    st.plotly_chart(fig, use_container_width=True)

    # Confusion matrix
    cm = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)


# -----------------------------
# About Page
# -----------------------------
elif menu == "ℹ️ About":
    st.write("### About this App")
    st.info("""
    **Churn Prediction Pro** is a demo app built with **Streamlit** and **Scikit-learn**.  
    It allows businesses to:
    - Predict customer churn in real time  
    - Analyze important factors driving churn  
    - Visualize customer data in a modern dashboard  

    Built with ❤️ using Python, Streamlit, and Machine Learning.
    """)
