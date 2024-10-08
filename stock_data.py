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
        
        logging.info(f"Fetching data for {symbol} from {latest_close_date}")
        historical_data = yf.Ticker(symbol).history(start=latest_close_date, end=latest_close_date + timedelta(days=1))
        logging.info(f"Fetched data: {historical_data}")

        if not historical_data.empty:
            latest_data = historical_data.iloc[-1]
            logging.info(f"Latest data for {symbol}: {latest_data}")
            
            # Check if 'Adj Close' column exists, if not use 'Close'
            adj_close_column = 'Adj Close' if 'Adj Close' in latest_data.index else 'Close'
            logging.info(f"Using {adj_close_column} for adj_close")
            
            data_entry = HistoricalData(
                stock_id=stock.id,
                date=latest_close_date,
                open=float(latest_data['Open']) if not pd.isna(latest_data['Open']) else None,
                high=float(latest_data['High']) if not pd.isna(latest_data['High']) else None,
                low=float(latest_data['Low']) if not pd.isna(latest_data['Low']) else None,
                close=float(latest_data['Close']) if not pd.isna(latest_data['Close']) else None,
                volume=float(latest_data['Volume']) if not pd.isna(latest_data['Volume']) else None,
                adj_close=float(latest_data[adj_close_column]) if not pd.isna(latest_data[adj_close_column]) else None
            )
            db_session.add(data_entry)
            db_session.commit()
            logging.info(f"Successfully stored latest data for {symbol}")

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
            db_session.flush() 

        # Fetch historical data
        historical_data = yf.Ticker(symbol).history(period="max")
        adj_close_column = 'Adj Close' if 'Adj Close' in historical_data.columns else 'Close'
        
        for date, row in historical_data.iterrows():
            existing_entry = db_session.query(HistoricalData).filter_by(stock_id=stock.id, date=date.date()).first()
            
            if existing_entry:
                # Update the existing record if needed
                existing_entry.open = float(row['Open']) if not pd.isna(row['Open']) else None
                existing_entry.high = float(row['High']) if not pd.isna(row['High']) else None
                existing_entry.low = float(row['Low']) if not pd.isna(row['Low']) else None
                existing_entry.close = float(row['Close']) if not pd.isna(row['Close']) else None
                existing_entry.volume = float(row['Volume']) if not pd.isna(row['Volume']) else None
                existing_entry.adj_close = float(row[adj_close_column]) if not pd.isna(row[adj_close_column]) else None
            else:
                # Insert new data if no record exists for the date
                data_entry = HistoricalData(
                    stock_id=stock.id,
                    date=date.date(),
                    open=float(row['Open']) if not pd.isna(row['Open']) else None,
                    high=float(row['High']) if not pd.isna(row['High']) else None,
                    low=float(row['Low']) if not pd.isna(row['Low']) else None,
                    close=float(row['Close']) if not pd.isna(row['Close']) else None,
                    volume=float(row['Volume']) if not pd.isna(row['Volume']) else None,
                    adj_close=float(row[adj_close_column]) if not pd.isna(row[adj_close_column]) else None
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
        
        return stock  
    except IntegrityError as e:
        db_session.rollback()
        logging.error(f"Integrity Error for {symbol}: {e}")
        raise
    except Exception as e:
        db_session.rollback()
        logging.error(f"Failed to fetch/store data for {symbol}: {e}")
        raise