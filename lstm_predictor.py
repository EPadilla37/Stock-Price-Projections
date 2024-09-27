import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import os
from models import db, Stock, Forecast, HistoricalData
from utils import get_latest_market_close
from dotenv import load_dotenv

load_dotenv()

database_url = os.getenv('DATABASE_URL')
# Database connection
engine = create_engine(database_url)

# Global scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Set the path for saving the model
MODEL_PATH = os.getenv('PATH')

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
    scaled_data = scaler.fit_transform(data['adj_close'].values.reshape(-1, 1))
    
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
    model.compile(optimizer='adam', loss='mean_squared_error')
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

def update_model(model, X, y):
    model.fit(X, y, epochs=1, batch_size=32, verbose=0)
    model.save(MODEL_PATH)
    print(f"Model updated and saved to {MODEL_PATH}")
    return model

def predict_next_day(model, last_60_days):
    current_batch = last_60_days[-60:].reshape((1, 60, 1))
    prediction = model.predict(current_batch)[0]
    return scaler.inverse_transform(prediction.reshape(-1, 1))[0][0]

def generate_forecast(stock_id):
    try:
        stock_data = fetch_stock_data(stock_id)
        
        if stock_data.empty:
            raise ValueError(f"No historical data found for stock_id {stock_id}")
        
        X, y = prepare_data(stock_data)
        model = train_or_load_model(X, y, look_back=60)
        
        last_60_days = scaler.fit_transform(stock_data['adj_close'].values[-60:].reshape(-1, 1))
        forecast = predict_next_day(model, last_60_days)
        
        latest_data_date = get_latest_market_close()
        forecast_date = latest_data_date + timedelta(days=1)
        
        return [{'date': forecast_date.strftime('%Y-%m-%d'), 'forecast': float(forecast)}]
    except Exception as e:
        print(f"Error in generate_forecast: {str(e)}")
        raise

def fetch_recent_forecasts():
    query = """
    SELECT f.stock_id, f.date, f.predicted_price, h.adj_close as actual_price
    FROM forecasts f
    JOIN historical_data h ON f.stock_id = h.stock_id AND f.date = h.date
    WHERE f.date = CURRENT_DATE - INTERVAL '1 day'
    ORDER BY f.stock_id, f.date
    """
    return pd.read_sql(query, engine)

def update_model_with_feedback():
    recent_forecasts = fetch_recent_forecasts()
    if recent_forecasts.empty:
        print("No recent forecasts to learn from")
        return
    
    X, y = prepare_data(recent_forecasts[['actual_price']])
    model = load_model(MODEL_PATH)
    updated_model = update_model(model, X, y)
    return updated_model

def generate_and_store_forecast(stock_id):
    forecast = generate_forecast(stock_id)
    store_forecast(stock_id, forecast)
    return forecast

def store_forecast(stock_id, forecast):
    for forecast_item in forecast:
        forecast_date = datetime.strptime(forecast_item['date'], '%Y-%m-%d').date()
        
        # Remove any existing forecast for the same date
        Forecast.query.filter_by(stock_id=stock_id, date=forecast_date).delete()
        
        new_forecast = Forecast(
            stock_id=stock_id,
            date=forecast_date,
            predicted_price=forecast_item['forecast'],
            actual_price=None,  # Set to None initially
            residual=None,  # Set to None initially
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
            forecast.actual_price = latest_historical.adj_close
            forecast.residual = forecast.predicted_price - forecast.actual_price
            db.session.commit()

def daily_model_update():
    update_model_with_feedback()
    # After updating the model, update actual prices and regenerate forecasts for all stocks
    for stock in Stock.query.all():
        update_actual_prices(stock.id)
        generate_and_store_forecast(stock.id)

# def daily_model_update():
#     update_model_with_feedback()
#     # After updating the model, regenerate forecasts for all stocks for the next day
#     for stock in Stock.query.all():
#         generate_and_store_forecast(stock.id)

if __name__ == "__main__":
    daily_model_update()