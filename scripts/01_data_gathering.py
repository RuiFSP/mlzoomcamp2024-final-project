import os
import logging
from typing import List

import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_URL = "https://www.football-data.co.uk/mmz4281/{}/E0.csv"
SEASONS = ["0506", "0607", "0708", "0809", "0910", "1011", "1112", "1213", "1314",
           "1415", "1516", "1617", "1718", "1819", "1920", "2021", "2122", "2223",
           "2324", "2425"]
RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw_data")
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
COLUMNS_TO_SELECT = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
                     'HTHG', 'HTAG', 'HTR', 'Referee', 'HS', 'AS', 'HST', 'AST',
                     'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'B365H', 'B365D', 'B365A']

def ensure_directory_exists(path: str) -> None:
    """
    Ensure the directory exists, create if it doesn't.

    Args:
        path (str): Path to the directory.
    """
    os.makedirs(path, exist_ok=True)

def download_csv_files(seasons: List[str], base_url: str, data_path: str) -> None:
    """
    Download CSV files for each season and save them to the specified path.

    Args:
        seasons (List[str]): List of seasons.
        base_url (str): Base URL for downloading the files.
        data_path (str): Path to save the downloaded files.
    """
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    
    for season in seasons:
        url = base_url.format(season)
        try:
            response = session.get(url)
            response.raise_for_status()
            file_path = os.path.join(data_path, f"premier_league_{season}.csv")
            with open(file_path, "wb") as f:
                f.write(response.content)
            logging.info(f"Successfully downloaded {url} to {file_path}")
        except requests.RequestException as e:
            logging.error(f"Failed to download {url}: {e}")

def check_columns_in_files(directory: str, columns: List[str]) -> None:
    """
    Check if all specified columns are present in each CSV file in the directory.

    Args:
        directory (str): Directory containing the CSV files.
        columns (List[str]): List of columns to check.
    """
    csv_files = [file for file in os.listdir(directory) if file.endswith(".csv")]
    for file in csv_files:
        file_path = os.path.join(directory, file)
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            missing_columns = [column for column in columns if column not in df.columns]
            if not missing_columns:
                logging.info(f"{file} contains all columns")
            else:
                logging.warning(f"{file} is missing columns: {missing_columns}")
        except pd.errors.ParserError as e:
            logging.error(f"Error parsing {file}: {e}")

def read_and_concatenate_csv_files(seasons: List[str], directory: str, columns: List[str]) -> pd.DataFrame:
    """
    Read and concatenate CSV files for each season, selecting specified columns.

    Args:
        seasons (List[str]): List of seasons.
        directory (str): Directory containing the CSV files.
        columns (List[str]): List of columns to select.

    Returns:
        pd.DataFrame: Concatenated DataFrame.
    """
    dfs = []
    for season in seasons:
        file_path = os.path.join(directory, f"premier_league_{season}.csv")
        try:
            df = pd.read_csv(file_path, usecols=columns, encoding='utf-8')
            df["Date"] = df["Date"].astype(str).apply(lambda x: x if len(x.split('/')[-1]) == 4 else x[:-2] + "20" + x[-2:])
            df["Season"] = "20" + season[:2]
            dfs.append(df)
        except pd.errors.ParserError as e:
            logging.error(f"Error parsing {file_path}: {e}")
        except ValueError as e:
            logging.error(f"Error processing {file_path}: {e}")
    concatenated_df = pd.concat(dfs, ignore_index=True).dropna()
    logging.info("Successfully concatenated all CSV files")
    return concatenated_df

def save_dataframe_to_csv(df: pd.DataFrame, path: str) -> None:
    """
    Save the DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame to save.
        path (str): Path to the CSV file.
    """
    df.to_csv(path, index=False)

def main() -> None:
    """
    Main function to orchestrate data gathering and processing.
    """
    ensure_directory_exists(RAW_DATA_PATH)
    download_csv_files(SEASONS, BASE_URL, RAW_DATA_PATH)
    check_columns_in_files(RAW_DATA_PATH, COLUMNS_TO_SELECT)
    result = read_and_concatenate_csv_files(SEASONS, RAW_DATA_PATH, COLUMNS_TO_SELECT)
    ensure_directory_exists(PROCESSED_DATA_PATH)
    save_dataframe_to_csv(result, os.path.join(PROCESSED_DATA_PATH, "all_concat_football_data.csv"))

if __name__ == "__main__":
    main()
