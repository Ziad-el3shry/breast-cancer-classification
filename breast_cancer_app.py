
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.figure_factory as ff
import plotly.graph_objects as go

# -----------------------------------
# UI Setup
st.set_page_config(page_title="Breast Cancer Classifier", layout="wide")
st.title("🔬 Breast Cancer Classification App")
st.markdown("A professional tool to experiment with different models on the breast cancer dataset.")

# Sidebar
st.sidebar.title("🧠 Model Settings")
model_choice = st.sidebar.selectbox("Choose Classifier", ["Logistic Regression", "Random Forest", "SVM"])
test_size = st.sidebar.slider("Test Size (%)", 10, 50, 30, step=5)

# -----------------------------------
# Load and prepare data
@st.cache_data
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df, data

df, data = load_data()
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

# -----------------------------------
# Model Training
def train_model(model_name):
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "Random Forest":
        model = RandomForestClassifier()
    elif model_name == "SVM":
        model = SVC()
    model.fit(X_train, y_train)
    return model

if st.sidebar.button("🚀 Train Model"):
    model = train_model(model_choice)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    st.subheader(f"✅ Results for {model_choice}")
    st.metric("Accuracy", f"{acc:.2%}")

    # Confusion Matrix
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=["Benign", "Malignant"],
        y=["Benign", "Malignant"],
        colorscale="Blues",
        showscale=True
    )
    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📊 Feature Importance (Only for Random Forest)"):
        if model_choice == "Random Forest":
            importances = model.feature_importances_
            feat_df = pd.DataFrame({
                "Feature": X.columns,
                "Importance": importances
            }).sort_values("Importance", ascending=False).head(10)

            feat_fig = go.Figure([go.Bar(x=feat_df["Importance"], y=feat_df["Feature"], orientation='h')])
            feat_fig.update_layout(title="Top 10 Important Features", xaxis_title="Importance", yaxis_title="Feature")
            st.plotly_chart(feat_fig, use_container_width=True)
        else:
            st.info("Feature importance available only for Random Forest.")

# -----------------------------------
# 🎯 Prediction on Custom Input
st.markdown("---")
st.subheader("🔍 Predict Cancer Type for Custom Input")

with st.expander("📝 Input Features Manually"):
    input_data = {}
    col1, col2, col3 = st.columns(3)
    for i, feature in enumerate(X.columns):
        with [col1, col2, col3][i % 3]:
            input_data[feature] = st.number_input(feature, value=float(X[feature].mean()), format="%.4f")

    if st.button("🔬 Predict"):
        input_df = pd.DataFrame([input_data])
        model = train_model(model_choice)  # reuse training function
        prediction = model.predict(input_df)[0]
        label = data.target_names[prediction]
        st.success(f"🧾 The model predicts: **{label}**")

