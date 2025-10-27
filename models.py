from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import jwt
from time import time
from flask import current_app

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    predictions = db.relationship('Prediction', backref='user', lazy=True, cascade='all, delete-orphan')
    harvests = db.relationship('Harvest', backref='user', lazy=True, cascade='all, delete-orphan')
    inventory_items = db.relationship('Inventory', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def get_reset_token(self, expires_in=600):
        return jwt.encode(
            {'reset_password': self.id, 'exp': time() + expires_in},
            current_app.config['SECRET_KEY'], algorithm='HS256'
        )
    
    @staticmethod
    def verify_reset_token(token):
        try:
            id = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])['reset_password']
        except:
            return None
        return db.session.get(User, id)

    def __repr__(self):
        return f'<User {self.username}>'

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    input_data = db.Column(db.JSON, nullable=False)
    predicted_usd_price = db.Column(db.Float, nullable=False)
    predicted_rwf_price = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Prediction {self.id} - User {self.user_id}>'

class Inventory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    item_name = db.Column(db.String(50), nullable=False)  # e.g., Potatoes, Fertilizer
    quantity = db.Column(db.Float, nullable=False)  # e.g., kg for crops, units for supplies
    unit = db.Column(db.String(20), nullable=False)  # e.g., kg, liters, units
    storage_location = db.Column(db.String(100), nullable=True)  # e.g., Warehouse A
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Inventory {self.item_name} - User {self.user_id}>'

class IrrigationDevice(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    device_name = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(200), nullable=False)
    latitude = db.Column(db.Float, default=0.0)
    longitude = db.Column(db.Float, default=0.0)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    valves = db.relationship('IrrigationValve', backref='device', lazy=True, cascade='all, delete-orphan')
    moisture_sensors = db.relationship('MoistureSensor', backref='device', lazy=True, cascade='all, delete-orphan')
    dht_sensors = db.relationship('DHTSensor', backref='device', lazy=True, cascade='all, delete-orphan')

    def __repr__(self):
        return f'<IrrigationDevice {self.device_name}>'

class IrrigationValve(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    device_id = db.Column(db.Integer, db.ForeignKey('irrigation_device.id'), nullable=False)
    valve_number = db.Column(db.Integer, nullable=False)  # 1-4
    valve_name = db.Column(db.String(50), nullable=False)
    is_open = db.Column(db.Boolean, default=False)
    flow_rate = db.Column(db.Float, default=0.0)  # L/min
    total_runtime = db.Column(db.Integer, default=0)  # minutes
    last_activated = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<IrrigationValve {self.valve_name} - Device {self.device_id}>'

class MoistureSensor(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    device_id = db.Column(db.Integer, db.ForeignKey('irrigation_device.id'), nullable=False)
    sensor_number = db.Column(db.Integer, nullable=False)  # 1-4
    sensor_name = db.Column(db.String(50), nullable=False)
    moisture_level = db.Column(db.Float, default=0.0)  # percentage
    threshold_min = db.Column(db.Float, default=30.0)  # minimum moisture threshold
    threshold_max = db.Column(db.Float, default=80.0)  # maximum moisture threshold
    last_reading = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<MoistureSensor {self.sensor_name} - Device {self.device_id}>'

class DHTSensor(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    device_id = db.Column(db.Integer, db.ForeignKey('irrigation_device.id'), nullable=False)
    sensor_number = db.Column(db.Integer, nullable=False)  # 1-4
    sensor_name = db.Column(db.String(50), nullable=False)
    temperature = db.Column(db.Float, default=0.0)  # Celsius
    humidity = db.Column(db.Float, default=0.0)  # percentage
    last_reading = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<DHTSensor {self.sensor_name} - Device {self.device_id}>'

class Harvest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    crop = db.Column(db.String(50), nullable=False)
    quantity = db.Column(db.Float, nullable=False)
    harvest_date = db.Column(db.Date, nullable=False)
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Harvest {self.crop} - User {self.user_id}>'
