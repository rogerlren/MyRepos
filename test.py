import requests

r = requests.get("https://www.gmail.com")

print(r.status_code)
print(r.ok)