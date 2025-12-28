import streamlit as st
import pandas as pd
import numpy as np
from data_processing import load_and_preprocess_data, create_features
from model import train_risk_model, predict_risk_level
from utils import classify_risk_level

# Set page config
st.set_page_config(
    page_title="Bacterial Blight Risk Prediction",
    page_icon="🌱",
    layout="wide"
)

# App title and description
st.title("Bacterial Blight Risk Prediction")
st.markdown("""
This application predicts the risk level of bacterial blight based on weather parameters. 
The prediction is based on a machine learning model trained on historical weather data.
""")

# Load and preprocess data
try:
    data = load_and_preprocess_data("attached_assets/Final_W_&_B_Data_full_edited.csv")
    st.success("Data loaded successfully!")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Create features and target
X, y, feature_names = create_features(data)

# Train model
model, scaler = train_risk_model(X, y)

# Predict Bacterial Blight Risk
st.header("Predict Bacterial Blight Risk")
st.markdown("Enter weather parameters to predict the risk level for bacterial blight.")

# Create columns for input fields
col1, col2 = st.columns(2)

with col1:
    temp_min = st.number_input("Minimum Temperature (°C)", min_value=10.0, max_value=50.0, value=22.0, step=0.1)
    temp_max = st.number_input("Maximum Temperature (°C)", min_value=10.0, max_value=50.0, value=30.0, step=0.1)
    rh_min = st.number_input("Minimum Relative Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    rh_max = st.number_input("Maximum Relative Humidity (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.1)

with col2:
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=200.0, value=5.0, step=0.1)
    rainy_days = st.number_input("Number of Rainy Days", min_value=0, max_value=7, value=1, step=1)
    sunshine_hrs = st.number_input("Sunshine Hours", min_value=0.0, max_value=12.0, value=4.0, step=0.1)

# Create input data for prediction
input_data = pd.DataFrame({
    'Temp_Min': [temp_min],
    'Temp_Max': [temp_max],
    'RH_Min': [rh_min],
    'RH_Max': [rh_max],
    'Rainfall': [rainfall],
    'No_of_rainy_days': [rainy_days],
    'Sunshine_Hrs': [sunshine_hrs]
})

# Make prediction when user clicks the button
if st.button("Predict Risk Level"):
    # Make prediction
    risk_score = predict_risk_level(model, scaler, input_data)
    risk_level = classify_risk_level(risk_score[0])
    
    # Display prediction result
    st.subheader("Prediction Result")
    
    # Use different colors for different risk levels
    if risk_level == "Low Risk":
        color = "green"
    elif risk_level == "Medium Risk":
        color = "orange"
    else:  # High Risk
        color = "red"
    
    # Display risk level
    st.markdown(f"<h3 style='color: {color};'>Risk Level: {risk_level}</h3>", unsafe_allow_html=True)
    
    # Display risk score
    st.markdown(f"Risk Score: {risk_score[0]:.2f} (0-1 scale)")

# Footer
st.markdown("---")
st.markdown("Developed for bacterial blight risk prediction based on weather parameters")
