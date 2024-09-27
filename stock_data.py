# stock_data.py

import yfinance as yf
from models import Stock, HistoricalData
import pandas as pd
import logging
import requests
from sqlalchemy.exc import IntegrityError
from lstm_predictor import update_actual_prices, generate_and_store_forecast
from utils import get_latest_market_close
from datetime import timedelta

def search_stock_info(symbol):
    stock = yf.Ticker(symbol)
    info = stock.info
    return {
        'symbol': symbol,
        'name': info.get('longName', 'N/A'),
        'sector': info.get('sector', 'N/A'),
        'industry': info.get('industry', 'N/A')
    }


def fetch_and_store_latest_data(symbol, db_session):
    try:
        stock = db_session.query(Stock).filter_by(symbol=symbol).first()
        if not stock:
            raise ValueError(f"Stock {symbol} not found in database")

        latest_close_date = get_latest_market_close()
        
        # Fetch latest data (up to the latest market close)
        historical_data = yf.Ticker(symbol).history(start=latest_close_date, end=latest_close_date + timedelta(days=1))

        if not historical_data.empty:
            latest_data = historical_data.iloc[-1]
            data_entry = HistoricalData(
                stock_id=stock.id,
                date=latest_close_date,
                open=float(latest_data['Open']) if not pd.isna(latest_data['Open']) else None,
                high=float(latest_data['High']) if not pd.isna(latest_data['High']) else None,
                low=float(latest_data['Low']) if not pd.isna(latest_data['Low']) else None,
                close=float(latest_data['Close']) if not pd.isna(latest_data['Close']) else None,
                volume=float(latest_data['Volume']) if not pd.isna(latest_data['Volume']) else None,
                adj_close=float(latest_data['Close']) if not pd.isna(latest_data['Close']) else None
            )
            db_session.add(data_entry)
            db_session.commit()

        return stock
    except Exception as e:
        db_session.rollback()
        logging.error(f"Failed to fetch/store latest data for {symbol}: {e}")
        raise

def fetch_and_store_stock_data(symbol, db_session):
    try:
        # Fetch stock information
        stock_info = yf.Ticker(symbol).info
        stock = db_session.query(Stock).filter_by(symbol=symbol).first()

        if not stock:
            stock = Stock(symbol=symbol, name=stock_info.get('longName', 'N/A'))
            db_session.add(stock)
            db_session.flush()  # This will assign an ID to the stock without committing the transaction

        # Fetch historical data
        historical_data = yf.Ticker(symbol).history(period="max")
        for date, row in historical_data.iterrows():
            data_entry = HistoricalData(
                stock_id=stock.id,
                date=date.date(),
                open=float(row['Open']) if not pd.isna(row['Open']) else None,
                high=float(row['High']) if not pd.isna(row['High']) else None,
                low=float(row['Low']) if not pd.isna(row['Low']) else None,
                close=float(row['Close']) if not pd.isna(row['Close']) else None,
                volume=float(row['Volume']) if not pd.isna(row['Volume']) else None,
                adj_close=float(row['Close']) if not pd.isna(row['Close']) else None
            )
            db_session.add(data_entry)

        db_session.commit()

        # Call the generate_forecast route
        generate_forecast_url = 'http://localhost:5000/generate_forecast'
        response = requests.post(generate_forecast_url, json={'stock_id': stock.id})
        if response.status_code == 200:
            logging.info(f"Forecast generated successfully for {symbol}")
        else:
            logging.error(f"Failed to generate forecast for {symbol}: {response.text}")
        
        return stock  # Return the stock object
    except IntegrityError as e:
        db_session.rollback()
        logging.error(f"Integrity Error for {symbol}: {e}")
        raise
    except Exception as e:
        db_session.rollback()
        logging.error(f"Failed to fetch/store data for {symbol}: {e}")
        raise