import requests

url = "http://127.0.0.1:8000/send_message"
headers = {"Content-Type": "application/json"}
data = {"message": "write me some python questions"}

response = requests.post(url, json=data, headers=headers)

if response.status_code == 200:
    print(response.json())
else:
    print(f"Request failed with status code {response.status_code}")
