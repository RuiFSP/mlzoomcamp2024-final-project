# imports
import pandas as pd
import numpy as np
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    return pd.read_csv(file_path)

def check_data(data):
    data_check = pd.DataFrame()
    data_check['Data Types'] = data.dtypes
    data_check['Missing Values'] = data.isnull().sum()
    data_check['Unique Values'] = data.nunique()
    data_check['Duplicates'] = [data.duplicated().sum()] + [None] * (data.shape[1] - 1)

    def detect_outliers(column):
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

def find_highly_correlated_features(data, threshold=0.8):
    numerical_data = data.select_dtypes(include=[np.number])
    corr_matrix = numerical_data.corr()
    high_corr_pairs = corr_matrix.where((corr_matrix.abs() > threshold) & (corr_matrix != 1))
    high_corr_pairs = high_corr_pairs.stack().reset_index()
    high_corr_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']
    return high_corr_pairs

def calculate_vif(data):
    numerical_data = data.select_dtypes(include=[np.number])
    X = add_constant(numerical_data)
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_data.sort_values(by='VIF', ascending=False, inplace=True)
    return vif_data

def remove_high_vif_features(data, vif_data, threshold=20):
    high_vif_features = vif_data[(vif_data['VIF'] == float('inf')) | (vif_data['VIF'] > threshold)]['Feature'].tolist()
    high_vif_features = [feature for feature in high_vif_features if feature != 'const']
    data.drop(columns=high_vif_features, inplace=True)
    return data

def save_data(data, file_path):
    data.to_csv(file_path, index=False)

def main():
    data_path = os.path.join(os.path.dirname(__file__),'..', 'data', 'processed','prepared_football_data.csv')
    data = load_data(data_path)
    
    data_for_back_testing = data[['implied_home_win_prob', 'implied_draw_prob', 'implied_away_win_prob']]
    data = data.drop(columns=['implied_home_win_prob', 'implied_draw_prob', 'implied_away_win_prob'])
    
    data_check = check_data(data)
    print(data_check)
    
    high_corr_pairs = find_highly_correlated_features(data)
    print(f"We have {high_corr_pairs.shape[0]} pairs of highly correlated features")
    
    vif_data = calculate_vif(data)
    data = remove_high_vif_features(data, vif_data)
    
    save_data(data, os.path.join(os.path.dirname(__file__),'..', 'data', 'processed', 'data_for_model.csv'))
    save_data(data_for_back_testing, os.path.join(os.path.dirname(__file__),'..', 'data', 'processed', 'data_for_back_testing.csv'))

if __name__ == "__main__":
    main()
