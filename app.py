# streamlit 

import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('heart_disease_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Streamlit app title
st.title("Heart Disease Classification App")

# Input form for user data
st.header("Input Patient Data")
age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0 to 3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar (> 120 mg/dL, 1 = True, 0 = False)", [0, 1])
restecg = st.selectbox("Resting ECG Results (0 to 2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
exang = st.selectbox("Exercise-Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("Slope of ST Segment (0 to 2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0 to 3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)", [1, 2, 3])

# Create input DataFrame
input_data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "cp": [cp],
    "trestbps": [trestbps],
    "chol": [chol],
    "fbs": [fbs],
    "restecg": [restecg],
    "thalach": [thalach],
    "exang": [exang],
    "oldpeak": [oldpeak],
    "slope": [slope],
    "ca": [ca],
    "thal": [thal]
})

# Prediction button
if st.button("Predict"):
    prediction = loaded_model.predict(input_data)[0]  # Predict with the loaded model
    if prediction == 1:
        st.error("Prediction: The patient is likely to have heart disease.")
    else:
        st.success("Prediction: The patient is not likely to have heart disease.")
