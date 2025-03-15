import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Streamlit App Title
st.title("Heart Disease Prediction App ❤️")

# File Upload
st.sidebar.header("Upload Your CSV Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())

    # Display dataset info
    st.write("### Dataset Info")
    st.write(df.describe())

    # Target Variable Distribution
    st.write("### Target Variable Distribution")
    st.bar_chart(df["target"].value_counts())

    # Data Preprocessing
    X = df.drop("target", axis=1)
    y = df["target"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Selection
    model_choice = st.sidebar.selectbox("Choose a Model", ["Logistic Regression", "KNN", "Random Forest"])

    if st.sidebar.button("Train Model"):
        if model_choice == "Logistic Regression":
            model = LogisticRegression()
        elif model_choice == "KNN":
            model = KNeighborsClassifier()
        else:
            model = RandomForestClassifier()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Save trained model in session state
        st.session_state.model = model

        # Model Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"### Model Accuracy: {accuracy:.2f}")

        # Confusion Matrix
        st.write("### Confusion Matrix")
        st.write(confusion_matrix(y_test, y_pred))

        # Classification Report
        st.write("### Classification Report")
        st.text(classification_report(y_test, y_pred))

    # User Input for Predictions
    st.sidebar.header("Make a Prediction")
    user_input = [st.sidebar.number_input(col, value=float(df[col].mean())) for col in X.columns]

    if st.sidebar.button("Predict"):
        if "model" in st.session_state:
            prediction = st.session_state.model.predict([user_input])
            st.write("### Prediction Result")
            st.write("🚨 Heart Disease Detected!" if prediction[0] == 1 else "✅ No Heart Disease")
        else:
            st.sidebar.error("❌ Please train a model first before making predictions!")
