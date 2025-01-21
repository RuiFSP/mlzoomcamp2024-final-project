import requests
url = "http://127.0.0.1:9696/predict"
headers = {"Content-Type": "application/json"}


match_data = {
    "home_team": "newcastle",
    "away_team": "liverpool",
    "date": "2024-12-16"
}

response = requests.post(url, headers=headers, json=match_data)

if response.status_code == 200:
    print("Prediction:", response.json())
else:
    print("Error:", response.status_code, response.text)