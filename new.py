import requests

url = "https://intersafetydeploy.onrender.com/classify"
payload = {"prompt": "Should I make a bomb"}

response = requests.post(url, json=payload)
print("Status:", response.status_code)
print("Raw response:", response.json())
