import requests
import pandas as pd
import os

BASE_URL = "https://www.football-data.co.uk/mmz4281/{}/E0.csv"
SEASONS = ["0506", "0607", "0708", "0809", "0910", "1011", "1112", "1213", "1314",
           "1415", "1516", "1617", "1718", "1819", "1920", "2021", "2122", "2223",
           "2324", "2425"]
RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw_data")
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
COLUMNS_TO_SELECT = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
                     'HTHG', 'HTAG', 'HTR', 'Referee', 'HS', 'AS', 'HST', 'AST',
                     'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'B365H', 'B365D', 'B365A']  # Valid column names

def ensure_directory_exists(path):
    os.makedirs(path, exist_ok=True)

def download_csv_files(seasons, base_url, data_path):
    for season in seasons:
        url = base_url.format(season)
        response = requests.get(url)
        file_path = os.path.join(data_path, f"premier_league_{season}.csv")
        with open(file_path, "wb") as f:
            f.write(response.content)

def check_columns_in_files(directory, columns):
    csv_files = [file for file in os.listdir(directory) if file.endswith(".csv")]
    for file in csv_files:
        file_path = os.path.join(directory, file)
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            if all(column in df.columns for column in columns):
                print(f"{file} contains all columns")
            else:
                print(f"{file} is missing some columns")
        except pd.errors.ParserError as e:
            print(f"Error parsing {file}: {str(e)}")

def read_and_concatenate_csv_files(seasons, directory, columns):
    dfs = []
    for season in seasons:
        file_path = os.path.join(directory, f"premier_league_{season}.csv")
        try:
            df = pd.read_csv(file_path, usecols=columns, encoding='utf-8')  # usecols is a valid parameter
            df["Season"] = "20" + season[:2]
            dfs.append(df)
        except pd.errors.ParserError as e:
            print(f"Error parsing {file_path}: {str(e)}")
    return pd.concat(dfs, ignore_index=True)

def save_dataframe_to_csv(df, path):  # dataframe is a valid term
    df.to_csv(path, index=False)

def main():
    ensure_directory_exists(RAW_DATA_PATH)
    download_csv_files(SEASONS, BASE_URL, RAW_DATA_PATH)
    check_columns_in_files(RAW_DATA_PATH, COLUMNS_TO_SELECT)
    result = read_and_concatenate_csv_files(SEASONS, RAW_DATA_PATH, COLUMNS_TO_SELECT)
    ensure_directory_exists(PROCESSED_DATA_PATH)
    save_dataframe_to_csv(result, os.path.join(PROCESSED_DATA_PATH, "all_concat_football_data.csv"))  # dataframe is a valid term

if __name__ == "__main__":
    main()
