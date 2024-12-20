import joblib
import numpy as np
import streamlit as st
import pandas as pd

# Load Save Model
model = joblib.load("D:\Portofolio ML GDGoC Ian\linearmodel.pkl")


# Web Titile
st.title('Heart Disease Prediction App')

st.divider()

st.write("With this app, you can get prediction for heart disease.")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.text_input('Age')
with col2:
    sex = st.text_input('Sex')
with col3:
    cp = st.text_input('Chest Pain Type')
with col1:
    trestbps = st.text_input('Resting Blood Pressure')
with col2: 
    chol = st.text_input('Serum Cholestoral in mg/dl')
with col3:
    fbs = st.text_input('Fasting Blood Sugar')
with col1:
    restecg = st.text_input('Resting Electrocardiographic Results')
with col2:
    thalach = st.text_input('Maximum Heart Rate')
with col3:
    exang = st.text_input('Exercise Induced Angina')
with col1:
    oldpeak = st.text_input('ST Depression')
with col2:
    slope = st.text_input('Slope')
with col3:
    ca = st.text_input('Major Vessels Number')
with col1:
    thal = st.text_input('Thal Number')


# Convert all DataFrame to NUmerik
input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], 
                           columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
print(input_data.dtypes)
input_data = input_data.apply(pd.to_numeric, errors='coerce')
print(input_data.dtypes)

# Code for Prediction
heart_diagnosis =''

# Prediction Button
if st.button('Predict'):
    st.balloons()
    heart_prediction = model.predict(input_data)
    
    if (heart_prediction[0]==1):
        heart_diagnosis = 'The Person has Heart Disease'
    else:
        heart_diagnosis = 'The Person does not have a Heart Disease'
        
st.success(heart_diagnosis)