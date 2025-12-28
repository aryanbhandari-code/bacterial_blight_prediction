import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the CSV data
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: Preprocessed data
    """
    # Load the data
    data = pd.read_csv(file_path)
    
    # Remove Wind_Speed column
    if 'Wind_Speed' in data.columns:
        data = data.drop('Wind_Speed', axis=1)
    
    # Check for missing values
    data = data.fillna(data.median())
    
    # Check for outliers
    for col in data.columns:
        if data[col].dtype in [np.float64, np.int64]:
            # Calculate IQR
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define bounds for outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Replace outliers with bounds
            data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
            data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])
    
    return data

def create_features(data):
    """
    Create features and target variable for the model
    
    Args:
        data (pd.DataFrame): Preprocessed data
    
    Returns:
        tuple: X (features), y (target), feature_names (list of feature names)
    """
    # Features
    feature_columns = [
        'Temp_Min', 'Temp_Max', 'RH_Min', 'RH_Max', 
        'Rainfall', 'No_of_rainy_days', 'Sunshine_Hrs'
    ]
    
    X = data[feature_columns]
    feature_names = feature_columns
    
    # Create target variable based on favorable conditions
    # Temperature: 24-30°C
    temp_condition = ((data['Temp_Min'] >= 24) | (data['Temp_Max'] <= 30))
    
    # Relative Humidity: above 45%
    rh_condition = (data['RH_Min'] > 45)
    
    # Rainfall: min 1mm
    rainfall_condition = (data['Rainfall'] >= 1)
    
    # Rainy days: min 1
    rainy_days_condition = (data['No_of_rainy_days'] >= 1)
    
    # Sunshine hours: around 4
    sunshine_condition = ((data['Sunshine_Hrs'] >= 3.5) & (data['Sunshine_Hrs'] <= 4.5))
    
    # Calculate target - the number of conditions met (0-5)
    y = temp_condition.astype(int) + rh_condition.astype(int) + \
        rainfall_condition.astype(int) + rainy_days_condition.astype(int) + \
        sunshine_condition.astype(int)
    
    # Normalize to 0-1 range
    y = y / 5
    
    return X, y, feature_names

def create_additional_features(data):
    """
    Create additional features for the model
    
    Args:
        data (pd.DataFrame): Preprocessed data
    
    Returns:
        pd.DataFrame: Data with additional features
    """
    # Copy the data to avoid modifying the original
    df = data.copy()
    
    # Temperature range
    df['Temp_Range'] = df['Temp_Max'] - df['Temp_Min']
    
    # Humidity range
    df['RH_Range'] = df['RH_Max'] - df['RH_Min']
    
    # Average temperature
    df['Temp_Avg'] = (df['Temp_Max'] + df['Temp_Min']) / 2
    
    # Average humidity
    df['RH_Avg'] = (df['RH_Max'] + df['RH_Min']) / 2
    
    # Temperature and humidity interaction
    df['Temp_RH_Interaction'] = df['Temp_Avg'] * df['RH_Avg'] / 100
    
    # Rainfall intensity (rainfall per rainy day)
    # Avoid division by zero by adding a small value
    df['Rainfall_Intensity'] = df['Rainfall'] / (df['No_of_rainy_days'] + 0.1)
    
    return df
