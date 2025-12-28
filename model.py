import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def train_risk_model(X, y):
    """
    Train a RandomForest model to predict risk level
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable (risk level)
    
    Returns:
        tuple: Trained model and scaler
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train the model
    model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model MSE: {mse:.4f}")
    print(f"Model R²: {r2:.4f}")
    
    return model, scaler

def predict_risk_level(model, scaler, input_data):
    """
    Predict risk level for new data
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        input_data (pd.DataFrame): Input data for prediction
    
    Returns:
        np.array: Predicted risk level
    """
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    
    return prediction

def calculate_blight_conditions_met(data):
    """
    Calculate how many of the blight-favorable conditions are met for each sample
    
    Args:
        data (pd.DataFrame): Input data
    
    Returns:
        pd.DataFrame: Data with additional column showing conditions met
    """
    df = data.copy()
    
    # Temperature: 24-30°C
    df['Temp_Favorable'] = ((df['Temp_Min'] >= 24) | (df['Temp_Max'] <= 30)).astype(int)
    
    # Relative Humidity: above 45%
    df['RH_Favorable'] = (df['RH_Min'] > 45).astype(int)
    
    # Rainfall: min 1mm
    df['Rainfall_Favorable'] = (df['Rainfall'] >= 1).astype(int)
    
    # Rainy days: min 1
    df['RainyDays_Favorable'] = (df['No_of_rainy_days'] >= 1).astype(int)
    
    # Sunshine hours: around 4
    df['Sunshine_Favorable'] = ((df['Sunshine_Hrs'] >= 3.5) & (df['Sunshine_Hrs'] <= 4.5)).astype(int)
    
    # Calculate total conditions met
    df['Conditions_Met'] = (
        df['Temp_Favorable'] + 
        df['RH_Favorable'] + 
        df['Rainfall_Favorable'] + 
        df['RainyDays_Favorable'] + 
        df['Sunshine_Favorable']
    )
    
    # Calculate risk score (0-1)
    df['Risk_Score'] = df['Conditions_Met'] / 5
    
    return df
