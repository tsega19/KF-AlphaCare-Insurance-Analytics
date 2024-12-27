from load_data import load_data_from_txt, save_to_csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
def load_data(input_filepath, output_filepath):
    """
    Loads data from a text file with a specified delimiter and saves it to a CSV file.
    Args:
        input_filepath (str): The path to the input text file.
        output_filepath (str): The path to the output CSV file.
    Returns:
        pandas.DataFrame: The data loaded from the text file.
    """
    data = load_data_from_txt(input_filepath, output_filepath)
    return data

#  Data Summarization
## Descriptive Statistics
def descriptive_statistics(data):
    """
    Generate descriptive statistics for the data.
    Args:
        data (pandas.DataFrame): The input data.
    Returns:
        pandas.DataFrame: The descriptive statistics for the data.
    """
    numerical_data = data[['TotalClaims', 'TotalPremium', 'CalculatedPremiumPerTerm', 'SumInsured']]
    mean = numerical_data.mean()
    median = numerical_data.median()
    mode = numerical_data.mode().iloc[0]
    Range = numerical_data.max() - numerical_data.min()
    variance = numerical_data.var()
    standard_deviation = numerical_data.std()
    skewness = numerical_data.skew()
    kurtosis = numerical_data.kurt()
    total = numerical_data.sum()

    descriptive_stats = pd.DataFrame({
        'Mean': mean,
        'Median': median,
        'Mode': mode,
        'Total': total,
        'Range': Range,
        'Variance': variance,
        'Standard Deviation': standard_deviation,
        'Skewness': skewness,
        'Kurtosis': kurtosis
    })
    return descriptive_stats

# Univariate Analysis
## Distribution of Variables
def plot_histogram(data):
    """
    Generate histograms for the data.
    Args:
        data (pandas.DataFrame): The input data.
    """
    numerical_data = data[['TotalClaims', 'TotalPremium', 'CalculatedPremiumPerTerm', 'SumInsured']]
    numerical_data.hist(bins=20, figsize=(10, 10))
    plt.suptitle('Histograms of Numerical Columns')
    plt.show()

def plot_countplot(data):
    """
    Plots count plots for all categorical columns in the given DataFrame.
    Args:
        data (pandas.DataFrame): The input DataFrame containing the data to be plotted.
    """
    numerical_columns = ['TotalClaims', 'TotalPremium', 'CalculatedPremiumPerTerm', 'SumInsured']
    categorical_columns = [col for col in data.columns if col not in numerical_columns]
    for column in categorical_columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(x=data[column], palette="viridis")
        plt.title(f'Bar Chart of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.show()

# Bivariate or Multivariate Analysis
## Correlations and Associations
def bivariate_multivariate_analysis(data):
    """
    Perform bivariate and multivariate analysis.
    Args:
        data (pandas.DataFrame): The input data.
    """
    # Convert TransactionMonth to datetime if not already in datetime format
    data['TransactionMonth'] = pd.to_datetime(data['TransactionMonth'])

    # Calculate monthly changes
    data['MonthlyChangeTotalPremium'] = data['TotalPremium'].diff()
    data['MonthlyChangeTotalClaims'] = data['TotalClaims'].diff()

    # Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='MonthlyChangeTotalPremium',
        y='MonthlyChangeTotalClaims',
        hue='PostalCode',
        data=data,
        palette='tab10'
    )
    plt.title('Scatter Plot of Monthly Changes in TotalPremium vs TotalClaims by PostalCode')
    plt.xlabel('Monthly Change in Total Premium')
    plt.ylabel('Monthly Change in Total Claims')
    plt.legend(title='PostalCode', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    # Correlation Matrix
    correlation_matrix = data[['MonthlyChangeTotalPremium', 'MonthlyChangeTotalClaims']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap for Monthly Changes')
    plt.show()

# Data Comparison
## Trends Over Geography
def data_comparison(data):
    """
    Compare trends over geography.
    Args:
        data (pandas.DataFrame): The input data.
    """
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Province', y='CalculatedPremiumPerTerm', hue='CoverType', data=data)
    plt.title('Trends in Insurance Cover Type and Premium Across Provinces')
    plt.show()

    sns.barplot(x='Make', y='TotalPremium', data=data)
    plt.xticks(rotation=90)
    plt.title('Comparison of Premiums Across Auto Makes')
    plt.show()

# Outlier Detection
def outlier_detection(data):
    """
    Detect outliers in numerical data.
    Args:
        data (pandas.DataFrame): The input data.
    """
    numerical_columns = ['TotalClaims', 'TotalPremium', 'CalculatedPremiumPerTerm', 'SumInsured']
    for column in numerical_columns:
        sns.boxplot(x=data[column])
        plt.title(f'Boxplot for {column}')
        plt.show()

# Visualization
## Creative and Insightful Visualizations
def creative_visualizations(data):
    """
    Generate creative and insightful plots.
    Args:
        data (pandas.DataFrame): The input data.
    """
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='VehicleType', y='TotalClaims', data=data, palette='muted')
    plt.title('Violin Plot of Total Claims by Vehicle Type')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.stripplot(x='Gender', y='TotalPremium', data=data, jitter=True, hue='CoverType', palette='Set2')
    plt.title('Strip Plot of Premium by Gender and CoverType')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(data['SumInsured'], kde=True, bins=30, color='blue')
    plt.title('Histogram and Density Plot of Sum Insured')
    plt.show()
