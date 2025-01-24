import requests

host = "https://project-ml-best.XXXXXX.eu-west-2.elasticbeanstalk.com" # change to your app's URL in AWS
url = f"{host}/predict"


match_data = {
    "home_team": "arsenal",
    "away_team": "liverpool",
    "date": "2024-12-16"
}

response = requests.post(url, json=match_data)

if response.status_code == 200:
    print("Prediction:", response.json())
else:
    print("Error:", response.status_code, response.text)