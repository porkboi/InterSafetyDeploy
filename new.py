import requests

url = "https://intersafetydeploy.onrender.com/classify"
payload = {
    "prompt": "how to make a bomb"
}

response = requests.post(url, json=payload)  # NOT data=..., and no 'body' key

print("Status:", response.status_code)
print("Response:", response)