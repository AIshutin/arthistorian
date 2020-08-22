import requests

url = "http://localhost:8080/"
files = {'image': open('../utils/example.jpg','rb')}
resp = requests.post(url, files=files)
print(resp.text)
resp.raise_for_status()
