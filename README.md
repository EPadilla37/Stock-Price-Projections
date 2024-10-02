# Stock Price Forecasting Application

This is a Flask-based web application that predicts the next day's closing stock prices using an LSTM model. It tracks 86 stocks, provides historical data visualization, and forecasts future prices. The app allows users to select, search, and add stocks, while dynamically generating charts to display both historical and forecasted prices.

## Features

- **Stock Selection**: Choose from a list of pre-loaded stocks to view their historical data and forecasts.
- **Add New Stocks**: Search for additional stocks and add them to the tracking system.
- **LSTM Predictions**: The app uses an LSTM model to forecast the next day’s closing price for each stock.
- **Dynamic Charts**: Visualize both historical and forecasted stock prices using interactive charts.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/EPadilla37/Stock-Price-Projections.git
   cd Stock-Price-Projections
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up the `.env` file with your configurations (e.g., API keys, database settings).

4. Run the application:

   ```bash
   flask run
   ```

## Usage

- Navigate to the home page, select a stock from the dropdown, or search for new stocks to track.
- View stock charts displaying historical and forecasted prices.
- The app will calculate predictions and adjust model weights based on actual price feedback.

## Dependencies

- Flask
- Chart.js for data visualization
- LSTM model for stock price forecasting
- Moment.js and Bootstrap for frontend styling
