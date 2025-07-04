import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon = "❤️",
    layout="wide",
    initial_sidebar_state="auto"
)

st.markdown(
    """
    <h1 >
        Heart Disease Predictor
    </h1>
    <hr style='border: 1px solid #ccc; margin-top: -10px; margin-bottom: 30px;'>
    """,
    unsafe_allow_html=True
)

st.markdown("""  
This tool uses a machine learning model trained on real patient data to predict the risk of heart disease based on clinical inputs.
""")

st.subheader("Enter patient details to check the risk of heart disease.")

# Load model
model = joblib.load("model.pkl")

# Input fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])

with col2:
    thalach = st.number_input("Max Heart Rate Achieved (thalach)", min_value=60, max_value=220)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, step=0.1)
    slope = st.selectbox("Slope of ST segment", [0, 1, 2])
    ca = st.selectbox("Number of major vessels (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thal", [0, 1, 2, 3])

# Converting inputs to numeric features
input_data = np.array([
    age,
    1 if sex == "Male" else 0,
    cp,
    trestbps,
    chol,
    fbs,
    restecg,
    thalach,
    exang,
    oldpeak,
    slope,
    ca,
    thal
]).reshape(1, -1)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]  # Probability of class 1 (heart disease)

    result = "🟢 No Heart Disease Detected" if prediction == 0 else "🔴 High Risk of Heart Disease"
    st.subheader(f"Prediction: {result}")
    st.info(f"🔍 Model Confidence (Risk Score): **{proba * 100:.2f}%**")


# Footer
st.markdown("---")
st.markdown(
    """
    Made with ❤️ by [Nishu Mehta](https://github.com/NishuMehta) · 
      
    📂 [Go to Project](https://github.com/NishuMehta/Heart-Disease-Prediction)
    """,
    unsafe_allow_html=True
)