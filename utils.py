def classify_risk_level(risk_score):
    """
    Classify risk score into risk level
    
    Args:
        risk_score (float): Risk score between 0 and 1
    
    Returns:
        str: Risk level (Low, Medium, or High)
    """
    if risk_score < 0.4:
        return "Low Risk"
    elif risk_score < 0.7:
        return "Medium Risk"
    else:
        return "High Risk"

def get_risk_color(risk_level):
    """
    Get color for risk level
    
    Args:
        risk_level (str): Risk level
    
    Returns:
        str: Color code
    """
    if risk_level == "Low Risk":
        return "green"
    elif risk_level == "Medium Risk":
        return "orange"
    else:  # High Risk
        return "red"

def get_condition_status(condition_met):
    """
    Get symbol for condition status
    
    Args:
        condition_met (bool): Whether condition is met
    
    Returns:
        str: Symbol (✅ or ❌)
    """
    return "✅" if condition_met else "❌"

def format_float(value, precision=2):
    """
    Format float to a specific precision
    
    Args:
        value (float): Value to format
        precision (int): Number of decimal places
    
    Returns:
        str: Formatted value
    """
    return f"{value:.{precision}f}"

def calculate_avg_risk_by_month(data):
    """
    Calculate average risk by month
    
    Args:
        data (pd.DataFrame): Dataset with date column and risk scores
    
    Returns:
        pd.DataFrame: Average risk by month
    """
    # Extract month from date
    if 'Date' in data.columns and pd.api.types.is_datetime64_any_dtype(data['Date']):
        data['Month'] = data['Date'].dt.month_name()
        
        # Calculate average risk by month
        monthly_risk = data.groupby('Month')['Risk_Score'].mean().reset_index()
        
        # Add risk level
        monthly_risk['Risk_Level'] = monthly_risk['Risk_Score'].apply(classify_risk_level)
        
        return monthly_risk
    else:
        return None

def calculate_threshold_impact(data, parameter, thresholds):
    """
    Calculate the impact of different thresholds on risk score
    
    Args:
        data (pd.DataFrame): Dataset
        parameter (str): Parameter to analyze
        thresholds (list): List of threshold values
    
    Returns:
        dict: Dictionary of threshold values and corresponding average risk scores
    """
    result = {}
    
    for threshold in thresholds:
        # Create mask based on threshold
        mask = data[parameter] >= threshold
        
        # Calculate average risk score for samples meeting the threshold
        avg_risk = data.loc[mask, 'Risk_Score'].mean()
        
        result[threshold] = avg_risk
    
    return result
