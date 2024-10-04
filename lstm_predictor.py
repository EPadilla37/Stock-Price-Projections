import traceback
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import os
from models import db, Stock, Forecast, HistoricalData
from utils import get_latest_market_close
from dotenv import load_dotenv
import logging

load_dotenv()

database_url = os.getenv('DATABASE_URL')
engine = create_engine(database_url)

scaler = MinMaxScaler(feature_range=(0, 1))

MODEL_PATH = os.getenv('MODEL_PATH', 'general_stock_model')
MODEL_PATH = os.path.splitext(MODEL_PATH)[0] + '.h5'

def fetch_stock_data(stock_id):
    query = """
    SELECT h.date, h.adj_close
    FROM historical_data h
    WHERE h.stock_id = %s
    ORDER BY h.date
    """
    df = pd.read_sql_query(query, engine, params=(stock_id,))
    df['date'] = pd.to_datetime(df['date'])
    return df

def prepare_data(data, look_back=60):
    if 'actual_price' in data.columns:
        price_column = 'actual_price'
    elif 'adj_close' in data.columns:
        price_column = 'adj_close'
    else:
        raise ValueError("Neither 'actual_price' nor 'adj_close' column found in data")

    scaled_data = scaler.fit_transform(data[price_column].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y

def create_model(look_back):
    inputs = Input(shape=(look_back, 1))
    x = LSTM(units=50, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(units=50, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(units=50)(x)
    x = Dropout(0.2)(x)
    outputs = Dense(units=1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model

def train_or_load_model(X, y, look_back):
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        model = create_model(look_back)
        model.fit(X, y, epochs=100, batch_size=32, verbose=0)
        model.save(MODEL_PATH)
        print(f"New model trained and saved to {MODEL_PATH}")
    return model

def update_model(old_model, X, y):
    # Create a new model with the same architecture
    new_model = create_model(X.shape[1])
    
    # Set the weights of the new model to be the same as the old model
    new_model.set_weights(old_model.get_weights())
    
    # Train the new model on the new data
    new_model.fit(X, y, epochs=1, batch_size=32, verbose=0)
    
    # Save the updated model
    new_model.save(MODEL_PATH)
    print(f"Model updated and saved to {MODEL_PATH}")
    return new_model


def predict_next_day(model, last_60_days):
    current_batch = last_60_days[-60:].reshape((1, 60, 1))
    prediction = model.predict(current_batch)[0]
    return scaler.inverse_transform(prediction.reshape(-1, 1))[0][0]

def generate_forecast(stock_id):
    try:
        stock_data = fetch_stock_data(stock_id)
        
        if stock_data.empty:
            raise ValueError(f"No historical data found for stock_id {stock_id}")
        
        if 'adj_close' not in stock_data.columns:
            logging.warning(f"'adj_close' column not found for stock_id {stock_id}. Using 'close' instead.")
            stock_data['adj_close'] = stock_data['close']
        
        X, _ = prepare_data(stock_data)
        model = load_model(MODEL_PATH)
        
        last_60_days = scaler.fit_transform(stock_data['adj_close'].values[-60:].reshape(-1, 1))
        forecast = predict_next_day(model, last_60_days)
        
        latest_data_date = get_latest_market_close()
        forecast_date = latest_data_date + timedelta(days=1)
        
        return [{'date': forecast_date.strftime('%Y-%m-%d'), 'forecast': float(forecast)}]
    except Exception as e:
        logging.error(f"Error in generate_forecast for stock_id {stock_id}: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def fetch_recent_forecasts():
    try:
        query = """
        SELECT f.stock_id, f.date, f.predicted_price, h.adj_close as actual_price
        FROM forecasts f
        JOIN historical_data h ON f.stock_id = h.stock_id AND f.date = h.date
        WHERE f.date = CURRENT_DATE - INTERVAL '1 day'
        ORDER BY f.stock_id, f.date
        """
        df = pd.read_sql(query, engine)
        if df.empty:
            logging.warning("No recent forecasts found")
        elif 'actual_price' not in df.columns:
            logging.error("'actual_price' column not found in fetched data")
        return df
    except Exception as e:
        logging.error(f"Error in fetch_recent_forecasts: {str(e)}")
        logging.error(traceback.format_exc())
        return pd.DataFrame()

def update_model_with_feedback():
    try:
        recent_forecasts = fetch_recent_forecasts()
        if recent_forecasts.empty:
            logging.info("No recent forecasts to learn from")
            return

        if 'actual_price' not in recent_forecasts.columns:
            logging.error("'actual_price' column not found in recent forecasts")
            return

        X, y = prepare_data(recent_forecasts[['actual_price']])
        old_model = load_model(MODEL_PATH)
        updated_model = update_model(old_model, X, y)
        return updated_model
    except Exception as e:
        logging.error(f"Error in update_model_with_feedback: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def generate_and_store_forecast(stock_id):
    try:
        forecast = generate_forecast(stock_id)
        store_forecast(stock_id, forecast)
        logging.info(f"Generated and stored new forecast for stock_id {stock_id}")
    except Exception as e:
        logging.error(f"Error generating forecast for stock_id {stock_id}: {str(e)}")
        raise

def store_forecast(stock_id, forecast):
    for forecast_item in forecast:
        forecast_date = datetime.strptime(forecast_item['date'], '%Y-%m-%d').date()
        
        # Remove any existing forecast for the same date
        Forecast.query.filter_by(stock_id=stock_id, date=forecast_date).delete()
        
        new_forecast = Forecast(
            stock_id=stock_id,
            date=forecast_date,
            predicted_price=forecast_item['forecast'],
            actual_price=None,  
            residual=None,  
            created_at=datetime.now()
        )
        db.session.add(new_forecast)
    
    db.session.commit()

def update_actual_prices(stock_id):
    latest_close_date = get_latest_market_close()
    
    # Get the latest historical data
    latest_historical = HistoricalData.query.filter_by(stock_id=stock_id, date=latest_close_date).first()
    
    if latest_historical:
        # Update the forecast for this date with the actual price
        forecast = Forecast.query.filter_by(stock_id=stock_id, date=latest_close_date).first()
        if forecast:
            if latest_historical.adj_close is not None:
                forecast.actual_price = latest_historical.adj_close
                forecast.residual = forecast.predicted_price - forecast.actual_price
                db.session.commit()
                logging.info(f"Updated actual price and residual for stock_id {stock_id} on {latest_close_date}")
            else:
                logging.warning(f"No adj_close data for stock_id {stock_id} on {latest_close_date}")
        else:
            logging.warning(f"No forecast found for stock_id {stock_id} on {latest_close_date}")
    else:
        logging.warning(f"No historical data found for stock_id {stock_id} on {latest_close_date}")

def daily_model_update():
    try:
        logging.info("Starting daily model update")
        
        update_model_with_feedback()
        logging.info("Model feedback update completed")
        
        for stock in Stock.query.all():
            try:
                logging.info(f"Processing stock {stock.symbol}")
                
                update_actual_prices(stock.id)
                logging.info(f"Updated actual prices for stock {stock.symbol}")
                
                generate_and_store_forecast(stock.id)
                logging.info(f"Generated new forecast for stock {stock.symbol}")
                
            except Exception as e:
                logging.error(f"Error processing stock {stock.symbol}: {str(e)}")
                logging.error(traceback.format_exc())
                # Continue with the next stock even if there's an error
                continue
        
        logging.info("Daily model update completed successfully")
    except Exception as e:
        logging.error(f"Error in daily_model_update: {str(e)}")
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    daily_model_update()