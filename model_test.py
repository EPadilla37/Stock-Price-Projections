# lstm_predictor.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sqlalchemy import create_engine
from datetime import datetime, timedelta
from models import db, Stock, Forecast, HistoricalData
import os

# Database connection
database_url = os.getenv('DATABASE_URL')
engine = create_engine(database_url)

# Global scaler
scaler = MinMaxScaler(feature_range=(0, 1))

def fetch_all_stock_data():
    query = """
    SELECT s.id as stock_id, h.date, h.adj_close
    FROM stocks s
    JOIN historical_data h ON s.id = h.stock_id
    ORDER BY s.id, h.date
    """
    df = pd.read_sql(query, engine)
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
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_or_load_model(X, y, look_back):
    model_path = 'general_stock_model.h5'
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = create_model(look_back)
        model.fit(X, y, epochs=100, batch_size=32, verbose=0)
        model.save(model_path)
    return model

def update_model(model, X, y):
    model.fit(X, y, epochs=1, batch_size=32, verbose=0)
    model.save('general_stock_model.h5')
    return model

def predict_next_month(model, last_60_days):
    predictions = []
    current_batch = last_60_days[-60:].reshape((1, 60, 1))
    
    for _ in range(30):  # Predict next 30 days
        current_pred = model.predict(current_batch)[0]
        predictions.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
    
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

def generate_forecast(stock_id):
    all_data = fetch_all_stock_data()
    stock_data = all_data[all_data['stock_id'] == stock_id]
    
    X, y = prepare_data(stock_data)
    model = train_or_load_model(X, y, look_back=60)
    
    last_60_days = scaler.transform(stock_data['adj_close'].values[-60:].reshape(-1, 1))
    forecast = predict_next_month(model, last_60_days)
    
    forecast_dates = [stock_data['date'].iloc[-1] + timedelta(days=i+1) for i in range(30)]
    
    return [
        {'date': date.strftime('%Y-%m-%d'), 'forecast': float(price)}
        for date, price in zip(forecast_dates, forecast.flatten())
    ]

def fetch_recent_forecasts():
    query = """
    SELECT f.stock_id, f.date, f.predicted_price, h.adj_close as actual_price
    FROM forecasts f
    JOIN historical_data h ON f.stock_id = h.stock_id AND f.date = h.date
    WHERE f.date >= CURRENT_DATE - INTERVAL '30 days'
    ORDER BY f.stock_id, f.date
    """
    return pd.read_sql(query, engine)

def update_model_with_feedback():
    recent_forecasts = fetch_recent_forecasts()
    if recent_forecasts.empty:
        return  # No recent forecasts to learn from
    
    X, y = prepare_data(recent_forecasts[['actual_price']])
    model = load_model('general_stock_model.h5')
    updated_model = update_model(model, X, y)
    return updated_model

def generate_and_store_forecast(stock_id):
    forecast = generate_forecast(stock_id)
    store_forecast(stock_id, forecast)
    return forecast

def store_forecast(stock_id, forecast):
    Forecast.query.filter_by(stock_id=stock_id).delete()
    
    for forecast_item in forecast:
        new_forecast = Forecast(
            stock_id=stock_id,
            date=datetime.strptime(forecast_item['date'], '%Y-%m-%d').date(),
            predicted_price=forecast_item['forecast'],
            created_at=datetime.now()
        )
        db.session.add(new_forecast)
    
    db.session.commit()

# Run this function periodically (e.g., daily) to update the model
def daily_model_update():
    update_model_with_feedback()
    # After updating the model, you might want to regenerate forecasts for all stocks
    for stock in Stock.query.all():
        generate_and_store_forecast(stock.id)