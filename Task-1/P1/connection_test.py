import requests
resp = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/AMZN")
print(resp.status_code)
print(resp.text[:200])  # Should start with {"chart":...} if valid