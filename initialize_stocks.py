import json
import requests

with open('stocks.json', 'r') as file:
    stocks = json.load(file)

# Base URL of your Flask application
base_url = 'http://localhost:5000'  


def select_stock(symbol):
    url = f"{base_url}/select_stock"
    data = {'symbol': symbol}
    try:
        response = requests.post(url, data=data)
        response.raise_for_status()  
        print(f"Successfully processed {symbol}: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Error processing {symbol}: {e}")

for stock in stocks:
    symbol = stock['symbol']
    print(f"Processing {symbol}...")
    select_stock(symbol)

print("All stocks processed.")