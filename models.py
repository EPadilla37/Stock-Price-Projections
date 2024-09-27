from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Stock(db.Model):
    __tablename__ = 'stocks'
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String, unique=True, nullable=False)
    name = db.Column(db.String)

class HistoricalData(db.Model):
    __tablename__ = 'historical_data'
    id = db.Column(db.Integer, primary_key=True)
    stock_id = db.Column(db.Integer, db.ForeignKey('stocks.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    open = db.Column(db.Float)
    high = db.Column(db.Float)
    low = db.Column(db.Float)
    close = db.Column(db.Float)
    volume = db.Column(db.BigInteger)
    adj_close = db.Column(db.Float)

class Forecast(db.Model):
    __tablename__ = 'forecasts'
    id = db.Column(db.Integer, primary_key=True)
    stock_id = db.Column(db.Integer, db.ForeignKey('stocks.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    predicted_price = db.Column(db.Float, nullable=False)
    actual_price = db.Column(db.Float, nullable=True)
    residual = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False)