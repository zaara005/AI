# app.py

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model, scaler, and columns
model = joblib.load('model/rf_model.pkl')
scaler = joblib.load('model/scaler.pkl')
columns = joblib.load('model/train_columns.pkl')

st.title("Employee Salary Prediction App")

st.write("Fill the details below to predict the employee's salary (USD).")

# Create input fields matching your preprocessing
work_year = st.number_input("Work Year", min_value=2020, max_value=2030, value=2025)
remote_ratio = st.slider("Remote Ratio (%)", min_value=0, max_value=100, value=0)

# Example categorical inputs (modify choices as per your dataset)
experience_level = st.selectbox("Experience Level", ['EN', 'MI', 'SE', 'EX'])
employment_type = st.selectbox("Employment Type", ['PT', 'FT', 'CT', 'FL'])
job_title = st.text_input("Job Title", "Data Scientist")
employee_residence = st.text_input("Employee Residence", "US")
company_location = st.text_input("Company Location", "US")
company_size = st.selectbox("Company Size", ['S', 'M', 'L'])

# When predict button is clicked
if st.button("Predict Salary"):
    # Create DataFrame with input
    input_dict = {
        'work_year': [work_year],
        'remote_ratio': [remote_ratio],
        'experience_level': [experience_level],
        'employment_type': [employment_type],
        'job_title': [job_title],
        'employee_residence': [employee_residence],
        'company_location': [company_location],
        'company_size': [company_size]
    }
    input_df = pd.DataFrame(input_dict)

    # One-hot encode to match training columns
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Scale numerical features
    input_df[['work_year', 'remote_ratio']] = scaler.transform(input_df[['work_year', 'remote_ratio']])

    # Predict
    salary_pred = model.predict(input_df)[0]
    st.success(f"Predicted Salary: ${salary_pred:,.2f} USD")
