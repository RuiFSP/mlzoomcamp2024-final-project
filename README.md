# Midterm Project: Premier League Football Prediction

## Overview

This project aims to predict the outcomes of Premier League football matches using machine learning models. It explores various features to determine their importance in predicting match results—whether it’s a home win, draw, or away win.

## Table of Contents

- [Overview](#overview)
- [Problem Description](#problem-description)
- [Data](#data)
  - [Home Wins (H)](#home-wins-h)
  - [Away Wins (A)](#away-wins-a)
  - [Draws (D)](#draws-d)
- [Key Observations](#key-observations)
- [Scripts](#scripts)
  - [data\_gathering](#data_gathering)
  - [data\_preparation](#data_preparation)
  - [eda](#eda)
  - [train\_model](#train_model)
  - [predict](#predict)
  - [back\_testing\_market](#back_testing_market)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Clone the Repository](#clone-the-repository)
  - [Installing Dependencies](#installing-dependencies)
    - [Navigate to Your Project Directory](#navigate-to-your-project-directory)
    - [Install the Project Dependencies](#install-the-project-dependencies)
    - [Activate the Virtual Environment](#activate-the-virtual-environment)
  - [Running Docker](#running-docker)
  - [Running on AWS Elastic Beanstalk](#running-on-aws-elastic-beanstalk)
  - [Testing the Model](#testing-the-model)
  - [Running the Streamlit App (Bonus)](#running-the-streamlit-app-bonus)
- [Contributing](#contributing)

## Problem Description

Predicting football match outcomes is a challenging yet captivating task that has long intrigued sports analysts and enthusiasts. This project focuses on the Premier League, aiming to forecast match results—home win, draw, or away win—by leveraging historical match data and team statistics.

In the [mlzoomcamp2024-midterm-project](https://github.com/RuiFSP/mlzoomcamp2024-midterm-project/tree/main), traditional machine learning algorithms like XGBoost were employed to build robust models, achieving strong performance and identifying key features that influence match outcomes. Building upon that foundation, the final project shifts toward exploring deep learning approaches, which can capture more complex patterns and interactions within the data.

By employing advanced deep learning architectures (e.g., neural networks, LSTMs, or CNNs), the project aims to:

Improve prediction accuracy.
Explore how deep learning can handle the inherent unpredictability and dynamic nature of football matches.
Gain insights into latent patterns that traditional methods may overlook.
Despite the ever-changing dynamics of sports and the competitive nature of prediction markets, this project aspires to push the boundaries of sports analytics by integrating cutting-edge techniques. The ultimate goal is not only to enhance predictive performance but also to provide valuable insights into the factors driving match outcomes.

## Data

The raw data for this project is sourced from [Football Data](https://www.football-data.co.uk/data.php). The focus is exclusively on the Premier League, covering seasons from 2005/2006 to 2024/2025. The raw data files can be found [here](https://github.com/RuiFSP/mlzoomcamp2024-midterm-project/tree/main/data/raw_data).

![FRT](images/results.png)

### Home Wins (H)
- **Dominant Trend**: Home wins consistently have the highest percentage among the three outcomes.
- **Long-Term Decline**: There is a slight decline in home win percentages from 2005 to around 2019.
- **Impact of 2020**: A significant drop occurs in 2020, likely due to external factors (e.g., the COVID-19 pandemic reducing home advantage).
- **Recovery**: Home win percentages show a recovery trend after 2020.

### Away Wins (A)
- **Stable Range**: Away wins generally hover in the 20–30% range, with minor fluctuations.
- **Recent Increase**: There is a slight increase in away wins in recent years, peaking around 2021 before stabilizing.

### Draws (D)
- **Lowest Percentage**: Draw percentages remain the lowest of the three outcomes, staying between 20–30%.
- **No Long-Term Trend**: There is no significant upward or downward trend, although short-term spikes and dips are visible.

## Key Observations
1. **2020 as a Pivotal Season**:
   - Home wins dropped significantly.
   - Away wins and draws increased noticeably during this season, potentially due to the neutralization of home advantage (e.g., matches without crowds).

2. **Home Advantage**: Home wins remain the dominant result over the years, indicating the significant influence of playing at home in football/soccer.

3. **Shift Toward Away Wins**: In the later years of the dataset, away wins have gained slight prominence.

---

## Scripts

![tree-simplified](images/tree.PNG)

### data_gathering

The `data_gathering.py` script performs the following key steps:

1. **Ensure Directories Exist**: Checks and creates necessary directories for storing data.
2. **Download Data**: Downloads CSV files for the specified seasons.
3. **Check Columns**: Verifies that the columns in the downloaded files match the expected schema.
4. **Concatenate Data**: Combines data from multiple seasons into a single dataset.
5. **Save Processed Data**: Saves the concatenated data to a CSV file for further processing.

For more details, see the [data_gathering.py](scripts/data_gathering.py) script.

### data_preparation

The `data_preparation.py` script performs the following key steps:

1. **Fix Columns, Teams, and Referees**: Rename columns, clean team names, and clean referee names.
2. **Handle Missing Values**: Drop rows with missing values and reset the index.
3. **Feature Engineering**: Create new features such as goal difference, total shots, shot accuracy, and time-based features.
4. **Rolling Averages**: Calculate rolling averages for various statistics over 3 and 5 game windows.
5. **Cumulative Points**: Compute cumulative points for home and away teams.
6. **Normalize Betting Odds**: Convert betting odds to implied probabilities.
7. **Save Processed Data**: Save the processed data for the current season (2024) and the final prepared dataset to CSV files.

For more details, see the [02_data_preparation.py](scripts/data_preparation.py) script.

### eda

The `eda.py` script is dedicated to Exploratory Data Analysis (EDA). It includes the following key steps:

1. **Data Checking**: Check data types, missing values, unique values, duplicates, and outliers.
2. **Saving Data**: Save the cleaned and processed data for modeling and backtesting.

> Note: In the **final project**, we focus on deep learning approaches, so we do not perform correlation analysis or VIF calculation as deep learning models can handle multicollinearity better. It is generally less critical to remove highly correlated features or to calculate VIF, as deep learning models can handle multicollinearity better than traditional machine learning models. We simplified our script by removing the parts related to finding highly correlated features and calculating VIF

For more details, see the [eda.py](scripts/eda.py) script.

### train_model

The `train_model.py` script covers the following key steps:

- **Data Preprocessing**: Load and preprocess the data, including encoding categorical features and scaling numerical features.
- **Data Splitting**: Split the data into training and test sets.
- **Class Weights Calculation**: Compute class weights to handle imbalanced datasets.
- **Model Building**: Build a Keras model with hyperparameter tuning using Keras Tuner.
- **Hyperparameter Tuning**: Perform hyperparameter tuning to find the best model configuration.
- **Model Evaluation**: Evaluate the best model on the test set.
- **Model Saving**: Save the best model and preprocessing objects for future use.

For more details, see the [train_model.py](scripts/train_model.py) script.

### predict

The `predict.py` script includes the following key steps:

- **Loading the Model and Data**: Load the trained model and preprocessing objects.
- **Setting Up Flask App**: Set up a Flask app to handle prediction requests.
- **Prediction Endpoint**: Define an endpoint to receive match data and return predictions.
- **Health Check Endpoint**: Define an endpoint to check the health status of the service.
- **Input Validation**: Validate the input data for required fields and correct formats.
- **Feature Engineering**: Generate features from the input data, including date-related features.
- **Preprocessing**: Apply the same preprocessing steps used during model training.
- **Prediction**: Use the trained model to predict match outcomes and probabilities.
- **Response Formatting**: Format the prediction results into a JSON response.

For more details, see the [predict.py](scripts/predict.py) script.

### back_testing_market

The `back_testing_market.py` script includes the following key steps:

- **Loading Data**: Load the processed data for back-testing.
- **Loading Model and Transformers**: Load the trained model and preprocessing objects.
- **Data Preprocessing**: Preprocess the data and split it into training and testing sets.
- **Predicting Results**: Use the trained model to predict match outcomes and probabilities.
- **Creating Team Names DataFrame**: Create a DataFrame to store team names, true results, predicted results, and probabilities.
- **Calculating Brier Scores**: Calculate the Brier scores for the market and the model to evaluate prediction accuracy.

For more details, see the [back_testing_market.py](scripts/back_testing_market.py) script.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Docker
- Pipenv

### Clone the Repository

Use `git clone` to copy the repository to your local machine and navigate into the project directory.

```bash
  git clone git@github.com:RuiFSP/mlzoomcamp2024-final-project.git
  cd mlzoomcamp2024-final-project
```

### Installing Dependencies

#### Navigate to Your Project Directory

First, open a terminal and change to the directory where your `Pipfile` and `Pipfile.lock` are located.

#### Install the Project Dependencies

In the project directory, use `pipenv install` to create the virtual environment and install all dependencies specified in the `Pipfile.lock`.

```bash
  pipenv install
```

This command will:

- Create a virtual environment if one doesn’t already exist.
- Install the dependencies exactly as specified in the `Pipfile.lock`.

#### Activate the Virtual Environment

To activate the virtual environment, use:

```bash
  pipenv shell
```

Now you're in an isolated environment where the dependencies specified in the `Pipfile.lock` are installed.

### Running Docker

Build the Docker image:

  > my setup: 
  > `<docker_image_name>` is project-ml

```bash
    docker build -t project-ml .
```

Run the Docker container:

```bash
    docker run -it --rm -p 9696:9696 project-ml
```

![aws_deploy](images/docker_locally.PNG)

> **Note:**  
> If you get an error with `[ 5/11] RUN 'pipenv install --system --deploy'`, try turning off your VPN.

### Running on AWS Elastic Beanstalk

To run Elastic Beanstalk, follow these steps:

1. **Install the AWS Elastic Beanstalk CLI**:

  > **Note:** don't forget you need to setup your access to AWS beforehand

   Ensure you have the AWS CLI and Elastic Beanstalk CLI installed. You can install the Elastic Beanstalk CLI using pip:

   ```bash
   pip install awsebcli
   ```

2. **Initialize Elastic Beanstalk**:
   Navigate to your project directory and initialize Elastic Beanstalk:

    > my setup:
      - `<project_name>` is project-ml

   ```bash
   eb init -p "Docker running on 64bit Amazon Linux 2" project-ml
   ```

    > Note: To run it locally i had to use Amazon Linux

   ```bash
   eb local run --port 9696
   ```

3. **Create an Environment and Deploy**:
   Create a new environment and deploy your application:

   ```bash
   eb create project-ml --platform "Docker running on 64bit Amazon Linux 2"
   ```

   ![aws_deploy](images/aws_deploy_web.PNG)

4. **Terminate the Environment**:
   When you are done, you can terminate the environment to stop incurring charges:

   ```bash
   eb terminate project-ml
   ```

![aws_deploy](images/success_aws.PNG)

### Testing the Model

Open a new terminal and run the test script:

```bash
    python tests/test_predict.py   # to test locally 
    python tests/test_predict_aws.py # to test aws 
```

![aws_deploy](images/docker_locally_tested.PNG)

To use the prediction service, send a POST request to the /predict endpoint with the following JSON payload locally or configure the test script accordingly:

```bash
curl -X POST http://127.0.0.1:9696/predict \
     -H "Content-Type: application/json" \
     -d '{
           "home_team": "arsenal",
           "away_team": "liverpool",
           "date": "2024-12-16"
         }'
```

![aws_deploy](images/curl_testing.PNG)

### Running the Streamlit App (Bonus)

To run the Streamlit app locally, follow these steps:

1. Ensure you have all dependencies installed and the virtual environment activated as described in the [Installing Dependencies](#installing-dependencies) section.

2. Navigate to the project directory where `app.py` is located.

3. Run the Streamlit app using the following command:

```bash
    streamlit run app.py
```

![streamlit_app](images/streamlit_example.PNG)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
