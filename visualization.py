import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import seaborn as sns
from utils import classify_risk_level

def plot_correlation_heatmap(data):
    """
    Create a correlation heatmap for the features
    
    Args:
        data (pd.DataFrame): Dataset
    
    Returns:
        matplotlib.figure.Figure: Correlation heatmap
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, 
        mask=mask, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        square=True, 
        linewidths=0.5, 
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    plt.title("Correlation Heatmap of Weather Parameters")
    plt.tight_layout()
    return fig

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance from the model
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (list): List of feature names
    
    Returns:
        matplotlib.figure.Figure: Feature importance plot
    """
    # Get feature importances
    importances = model.feature_importances_
    
    # Sort importances
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(sorted_importances)), sorted_importances, align='center')
    ax.set_yticks(range(len(sorted_importances)))
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    plt.tight_layout()
    return fig

def plot_risk_distribution(y):
    """
    Plot the distribution of risk levels in the dataset
    
    Args:
        y (pd.Series): Risk scores
    
    Returns:
        plotly.graph_objects.Figure: Risk distribution plot
    """
    # Convert risk scores to risk levels
    risk_levels = [classify_risk_level(score) for score in y]
    
    # Count the number of samples in each risk level
    risk_counts = pd.Series(risk_levels).value_counts().reset_index()
    risk_counts.columns = ['Risk Level', 'Count']
    
    # Create a color map for risk levels
    color_map = {
        'Low Risk': 'green',
        'Medium Risk': 'orange',
        'High Risk': 'red'
    }
    
    # Order risk levels
    risk_order = ['Low Risk', 'Medium Risk', 'High Risk']
    risk_counts['Risk Level'] = pd.Categorical(
        risk_counts['Risk Level'], 
        categories=risk_order, 
        ordered=True
    )
    risk_counts = risk_counts.sort_values('Risk Level')
    
    # Create plot
    fig = px.bar(
        risk_counts, 
        x='Risk Level', 
        y='Count',
        color='Risk Level',
        color_discrete_map=color_map,
        title='Distribution of Bacterial Blight Risk Levels'
    )
    
    return fig

def plot_parameter_distribution(data, parameter):
    """
    Plot the distribution of a specific parameter
    
    Args:
        data (pd.DataFrame): Dataset
        parameter (str): Parameter to plot
    
    Returns:
        plotly.graph_objects.Figure: Distribution plot
    """
    fig = px.histogram(
        data, 
        x=parameter,
        nbins=30,
        title=f'Distribution of {parameter}',
        opacity=0.7
    )
    
    fig.update_layout(
        xaxis_title=parameter,
        yaxis_title='Frequency',
        bargap=0.1
    )
    
    return fig

def plot_risk_factors_importance(data):
    """
    Create a visualization showing the importance of different risk factors
    
    Args:
        data (pd.DataFrame): Dataset with favorable condition columns
    
    Returns:
        plotly.graph_objects.Figure: Risk factors importance plot
    """
    # Check which conditions are met most often
    condition_columns = [
        'Temp_Favorable', 
        'RH_Favorable', 
        'Rainfall_Favorable', 
        'RainyDays_Favorable', 
        'Sunshine_Favorable'
    ]
    
    condition_names = [
        'Temperature (24-30°C)', 
        'Relative Humidity (>45%)', 
        'Rainfall (≥1mm)', 
        'Rainy Days (≥1)', 
        'Sunshine Hours (~4)'
    ]
    
    # Calculate the percentage of samples where each condition is met
    percentages = data[condition_columns].mean() * 100
    
    # Create a DataFrame for plotting
    plot_data = pd.DataFrame({
        'Condition': condition_names,
        'Percentage': percentages.values
    })
    
    # Create plot
    fig = px.bar(
        plot_data,
        x='Percentage',
        y='Condition',
        orientation='h',
        title='Percentage of Samples Meeting Each Favorable Condition',
        labels={'Percentage': 'Percentage of Samples (%)'}
    )
    
    fig.update_layout(
        yaxis={'categoryorder':'total ascending'}
    )
    
    return fig
