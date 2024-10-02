from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_apscheduler import APScheduler
from sqlalchemy.exc import OperationalError
import logging
from models import db, Stock, HistoricalData, Forecast
from stock_data import search_stock_info, fetch_and_store_stock_data, fetch_and_store_latest_data
from lstm_predictor import generate_forecast, store_forecast, daily_model_update
from utils import get_latest_market_close
from datetime import datetime, timedelta
import os

app = Flask(__name__)
database_url = os.getenv('DATABASE_URL')
app.config['SQLALCHEMY_DATABASE_URI'] = database_url
db.init_app(app)
CORS(app)

scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

try:
    with app.app_context():
        db.create_all()
except OperationalError as e:
    logger.error("Error connecting to the database: %s", str(e))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_stocks', methods=['GET'])
def get_stocks():
    stocks = Stock.query.all()
    return jsonify({
        'stocks': [{'id': stock.id, 'symbol': stock.symbol, 'name': stock.name} for stock in stocks]
    })

@app.route('/search_stock', methods=['POST'])
def search_stock():
    symbol = request.form['symbol']
    try:
        stock_info = search_stock_info(symbol)
        return jsonify(stock_info)
    except Exception as e:
        logger.error("Error searching stock: %s", str(e))
        return jsonify({'error': 'An error occurred while searching for the stock'}), 500

@app.route('/select_stock', methods=['POST'])
def select_stock():
    symbol = request.form['symbol']
    
    try:
        # Check if stock already exists in database
        stock = Stock.query.filter_by(symbol=symbol).first()
        if stock:
            # Fetch historical data
            historical_data = HistoricalData.query.filter_by(stock_id=stock.id).order_by(HistoricalData.date.desc()).limit(30).all()
            
            # Fetch forecast data
            latest_close_date = get_latest_market_close()
            forecast_data = Forecast.query.filter(
                Forecast.stock_id == stock.id,
                Forecast.date >= latest_close_date
            ).order_by(Forecast.date).all()
            
            if historical_data and forecast_data:
                return jsonify({
                    'message': 'Data already exists',
                    'stock_id': stock.id,
                    'historical_data': [{'date': hd.date.strftime('%Y-%m-%d'), 'price': hd.adj_close} for hd in reversed(historical_data)],
                    'forecast_data': [{'date': fd.date.strftime('%Y-%m-%d'), 'forecast': fd.predicted_price, 'actual': fd.actual_price} for fd in forecast_data]
                })
        
        # If stock doesn't exist or data is incomplete, proceed with current logic
        new_stock = fetch_and_store_stock_data(symbol, db.session)
        forecast = generate_forecast(new_stock.id)
        store_forecast(new_stock.id, forecast)
        
        return jsonify({'message': 'Stock added successfully', 'stock_id': new_stock.id})
    except Exception as e:
        logger.exception(f"Error selecting stock: {str(e)}")
        return jsonify({'error': f'An error occurred while adding the stock: {str(e)}'}), 500

@app.route('/generate_forecast', methods=['POST'])
def generate_stock_forecast():
    try:
        data = request.get_json()
        stock_id = data['stock_id']
        stock = db.session.get(Stock, stock_id)
        if not stock:
            return jsonify({'error': 'Stock not found'}), 404
        
        forecast = generate_forecast(stock_id)
        store_forecast(stock_id, forecast)
        return jsonify({
            'message': 'Forecast generated successfully',
            'forecast': forecast
        })
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}")
        return jsonify({'error': 'An error occurred while generating the forecast'}), 500
    
@app.route('/get_stock_data/<int:stock_id>', methods=['GET'])
def get_stock_data(stock_id):
    try:
        # Fetch historical data
        historical_data = HistoricalData.query.filter_by(stock_id=stock_id).order_by(HistoricalData.date.desc()).limit(30).all()
        
        # Fetch forecast data
        latest_close_date = get_latest_market_close()
        forecast_data = Forecast.query.filter(
            Forecast.stock_id == stock_id,
            Forecast.date >= latest_close_date
        ).order_by(Forecast.date).all()
        
        return jsonify({
            'historical_data': [{'date': hd.date.strftime('%Y-%m-%d'), 'price': hd.adj_close} for hd in reversed(historical_data)],
            'forecast_data': [{'date': fd.date.strftime('%Y-%m-%d'), 'forecast': fd.predicted_price} for fd in forecast_data]
        })
    except Exception as e:
        logger.exception(f"Error fetching stock data: {str(e)}")
        return jsonify({'error': f'An error occurred while fetching stock data: {str(e)}'}), 500

@app.route('/update', methods=['POST'])
def update_stocks():
    try:
        for stock in Stock.query.all():
            # Fetch and store latest data, then generate new forecast
            fetch_and_store_latest_data(stock.symbol, db.session)
        
        daily_model_update()
        
        return jsonify({'message': 'Stocks updated successfully'})
    except Exception as e:
        logger.error(f"Error updating stocks: {str(e)}")
        return jsonify({'error': 'An error occurred while updating stocks'}), 500
    
def scheduled_update():
    with app.app_context():
        update_stocks()

@app.route('/start_scheduler', methods=['GET'])
def start_scheduler():
    if not scheduler.get_job('update_stocks'):
        scheduler.add_job(id='update_stocks', func=scheduled_update, trigger='cron', hour=18, minute=0, timezone='US/Eastern')
        return jsonify({'message': 'Scheduler started successfully'})
    else:
        return jsonify({'message': 'Scheduler is already running'})

if __name__ == '__main__':
    with app.app_context():
        # scheduled_update() //use only to manually check stock after 3:10PM PST
        if not scheduler.get_job('update_stocks'):
            scheduler.add_job(id='update_stocks', func=scheduled_update, trigger='cron', hour=18, minute=19, timezone='US/Eastern')
            print("Scheduler started")
    app.run(debug=True)