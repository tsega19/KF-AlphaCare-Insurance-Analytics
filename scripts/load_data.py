import pandas as pd


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
