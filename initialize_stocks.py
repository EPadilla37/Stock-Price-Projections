import json
import requests

# Load the stocks from the JSON file
with open('stocks.json', 'r') as file:
    stocks = json.load(file)

# Base URL of your Flask application
base_url = 'http://localhost:5000'  # Adjust this if your server is running on a different port or host

# Function to call the /select_stock endpoint
def select_stock(symbol):
    url = f"{base_url}/select_stock"
    data = {'symbol': symbol}
    try:
        response = requests.post(url, data=data)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        print(f"Successfully processed {symbol}: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Error processing {symbol}: {e}")

# Iterate through the stocks and call /select_stock for each
for stock in stocks:
    symbol = stock['symbol']
    print(f"Processing {symbol}...")
    select_stock(symbol)

print("All stocks processed.")