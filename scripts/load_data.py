import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_data_from_txt(input_filepath, output_filepath):
    """
    Loads data from a text file with a specified delimiter and saves it to a CSV file.
    Args:
        input_filepath (str): The path to the input text file.
        output_filepath (str): The path to the output CSV file.
    Returns:
        pandas.DataFrame: The data loaded from the text file.
    """
    data = pd.read_csv(input_filepath, delimiter='|') 
    data.to_csv(output_filepath, index=False)
    return data
def missing_values(data):
    """
    Identify columns with missing values.
    Args:
        data (pandas.DataFrame): The input data.
    Returns:
        pandas.DataFrame: A DataFrame with columns containing missing values.
    """
    missing = data.isnull().sum()
    missing = missing[missing > 0]
    return missing

def fill_missing_values(data):
    """
    Fill missing values in the given DataFrame.
    This function fills missing values in both categorical and numerical columns.
    For categorical columns, it fills missing values with the mode (most frequent value).
    For numerical columns, it fills missing values with the mean of the column.
    Parameters:
    data (pd.DataFrame): The input DataFrame with potential missing values.
    Returns:
    pd.DataFrame: The DataFrame with missing values filled.
    """ 
    categorical_cols = ['Bank', 'AccountType', 'MaritalStatus', 'Gender', 'mmcode', 
                        'VehicleType', 'make', 'Model', 'bodytype', 'NumberOfDoors', 
                        'VehicleIntroDate', 'NewVehicle', 'WrittenOff', 'Rebuilt', 
                        'Converted', 'CrossBorder']

    for col in categorical_cols:
        if col in data.columns:
            data[col].fillna(data[col].mode()[0], inplace=True)
    
    # Numerical columns: Fill with mean
    numerical_cols = ['Cylinders', 'cubiccapacity', 'kilowatts', 'CustomValueEstimate', 
                      'CapitalOutstanding', 'NumberOfVehiclesInFleet']
    
    for col in numerical_cols:
        if col in data.columns:
            # Ensure the column is numeric
            data[col] = pd.to_numeric(data[col], errors='coerce')  # Convert non-numeric to NaN
            data[col].fillna(data[col].mean(), inplace=True)  # Fill with mean

    return data

# data structure
def data_structure(data):
    """
    Display the structure of the data in a single table.
    Args:
        data (pandas.DataFrame): The input data.
    """
    data_results = pd.DataFrame({
        'Column': data.columns,
        'Missing Values': data.isnull().sum(),
        'Unique Values': data.nunique(),
        'Data Type': data.dtypes
    })
    return data_results


def save_to_csv(df, output_filepath):
    """
    Save a DataFrame to a CSV file.
    Parameters:
    df (pandas.DataFrame): The DataFrame to be saved.
    output_filepath (str): The file path where the CSV file will be saved.

    Returns:
    None
    Prints a message indicating whether the data was successfully saved or if an error occurred.
    """
    if df is not None:
        try:
            df.to_csv(output_filepath, index=False)
            print(f"Data saved to {output_filepath}")
        except Exception as e:
            print(f"Error saving file: {e}")
    else:
        print("No data available to save.")
