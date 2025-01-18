# imports
import pandas as pd
import numpy as np
import os
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data as a DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    logging.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

def check_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic checks and statistics on the data.

    Args:
        data (pd.DataFrame): Input data.

    Returns:
        pd.DataFrame: Data check results.
    """
    logging.info("Checking data")
    data_check = pd.DataFrame()
    data_check['Data Types'] = data.dtypes
    data_check['Missing Values'] = data.isnull().sum()
    data_check['Unique Values'] = data.nunique()
    data_check['Duplicates'] = [data.duplicated().sum()] + [None] * (data.shape[1] - 1)

    def detect_outliers(column: pd.Series) -> Optional[int]:
        if np.issubdtype(column.dtype, np.number):
            z_scores = (column - column.mean()) / column.std()
            return ((z_scores.abs() > 3).sum())
        return None
    
    data_check['Outliers (Z>3)'] = data.apply(detect_outliers)
    data_check['Mean'] = data.select_dtypes(include=[np.number]).mean()
    data_check['Median'] = data.select_dtypes(include=[np.number]).median()
    data_check['Min'] = data.select_dtypes(include=[np.number]).min()
    data_check['Max'] = data.select_dtypes(include=[np.number]).max()
    data_check.loc['Shape'] = [''] * len(data_check.columns)
    data_check.loc['Shape', 'Data Types'] = f"Rows: {data.shape[0]}, Columns: {data.shape[1]}"
    
    return data_check

def save_data(data: pd.DataFrame, file_path: str) -> None:
    """
    Save data to a CSV file.

    Args:
        data (pd.DataFrame): Data to save.
        file_path (str): Path to the CSV file.
    """
    logging.info(f"Saving data to {file_path}")
    data.to_csv(file_path, index=False)

def main() -> None:
    """
    Main function to load, check, and save data.
    """
    data_path = os.path.join(os.path.dirname(__file__),'..', 'data', 'processed','prepared_football_data.csv')
    data = load_data(data_path)
    
    data_for_back_testing = data[['implied_home_win_prob', 'implied_draw_prob', 'implied_away_win_prob']]
    data = data.drop(columns=['implied_home_win_prob', 'implied_draw_prob', 'implied_away_win_prob'])
    
    data_check = check_data(data)
    print(data_check)
    
    save_data(data, os.path.join(os.path.dirname(__file__),'..', 'data', 'processed', 'data_for_model.csv'))
    save_data(data_for_back_testing, os.path.join(os.path.dirname(__file__),'..', 'data', 'processed', 'data_for_back_testing.csv'))

if __name__ == "__main__":
    main()
