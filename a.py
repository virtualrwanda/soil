import pandas as pd
import joblib
import sklearn
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, current_app
import os
import numpy as np
from datetime import datetime
from flask_mail import Mail, Message
from flask_login import LoginManager, login_user, logout_user, current_user, login_required, UserMixin
from flask_caching import Cache
from functools import wraps
import json
import secrets
import logging
import sqlite3

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from custom modules
from config import Config
from models import db, User, Prediction, IrrigationDevice, IrrigationValve, MoistureSensor, DHTSensor, Harvest
from forms import LoginForm, SignupForm, ForgotPasswordForm, ResetPasswordForm

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize Flask extensions
db.init_app(app)
mail = Mail(app)

# Initialize caching
app.config['CACHE_TYPE'] = 'SimpleCache'
cache = Cache(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# Flask-Login User Loader
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# --- Model Loading ---
MODEL_DIR = "trained_models"
MODEL_FILENAME = "Linear_Regression_pipeline.joblib"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

model_pipeline = None
try:
    if os.path.exists(MODEL_PATH):
        model_pipeline = joblib.load(MODEL_PATH)
        logger.info(f"Model '{MODEL_FILENAME}' loaded successfully.")
    else:
        logger.warning(f"Main prediction model not found at {MODEL_PATH}. Prediction functionality will be unavailable.")
except Exception as e:
    logger.error(f"An error occurred while loading the main model: {e}")

# ARIMA models for crop price forecasting
arima_models = {}
try:
    arima_models['potatoes'] = joblib.load(os.path.join(MODEL_DIR, 'arima_potato_price_model.pkl'))
    arima_models['carrots'] = joblib.load(os.path.join(MODEL_DIR, 'arima_Carrots_price_model.pkl'))
    arima_models['beans'] = joblib.load(os.path.join(MODEL_DIR, 'arima_Beans_price_model.pkl'))
    arima_models['tomatoes'] = joblib.load(os.path.join(MODEL_DIR, 'arima_Tomatoes_price_model.pkl'))
    logger.info("ARIMA models for forecasting loaded successfully.")
except FileNotFoundError:
    logger.warning("ARIMA model files not found. Crop price forecasting will be unavailable.")

# Define expected model features
EXPECTED_FEATURES = [
    'year', 'month', 'day_of_week', 'day_of_year',
    'admin1', 'admin2', 'market', 'category', 'commodity', 'unit', 'pricetype'
]

# Exchange Rate (Fixed for demonstration)
USD_TO_RWF_EXCHANGE_RATE = 1250

# --- Utility Functions ---
def admin_required(f):
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if not current_user.is_admin:
            flash('You do not have permission to access this page.', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

# Data Loading for Visualization
DATASET_PATH = os.path.join('data', 'food_prices_data.csv')

@cache.memoize(timeout=3600)
def load_dataset():
    try:
        df = pd.read_csv(DATASET_PATH, low_memory=False)
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
        df.dropna(subset=['date', 'price'], inplace=True)
        for col in ['price', 'usdprice', 'latitude', 'longitude', 'market_id', 'commodity_id']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        logger.info(f"Dataset '{DATASET_PATH}' loaded successfully for visualization.")
        return df
    except FileNotFoundError:
        logger.error(f"Dataset not found at {DATASET_PATH}.")
        flash(f"Error: Dataset not found at {DATASET_PATH}.", 'danger')
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        flash(f"Error loading dataset: {str(e)}", 'danger')
        return pd.DataFrame()

def create_dynamic_price_trend_plot(df, commodity, admin1, pricetype, market=None, start_date=None, end_date=None):
    if df.empty:
        return {}
    filtered_df = df.copy()
    if commodity: filtered_df = filtered_df[filtered_df['commodity'] == commodity]
    if admin1: filtered_df = filtered_df[filtered_df['admin1'] == admin1]
    if pricetype: filtered_df = filtered_df[filtered_df['pricetype'] == pricetype]
    if market: filtered_df = filtered_df[filtered_df['market'] == market]
    if start_date: filtered_df = filtered_df[filtered_df['date'] >= pd.to_datetime(start_date)]
    if end_date: filtered_df = filtered_df[filtered_df['date'] <= pd.to_datetime(end_date)]
    
    filtered_df = filtered_df.sort_values('date')
    if filtered_df.empty: return {}
    
    title = f"Price Trend of {commodity or 'All Commodities'} in {admin1 or 'All Regions'} ({pricetype or 'All Price Types'})"
    if market: title += f" at {market}"

    labels = filtered_df['date'].dt.strftime('%Y-%m-%d').tolist()
    prices = filtered_df['price'].tolist()

    chart_config = {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [{
                "label": "Price (RWF)", "data": prices, "borderColor": "#10b981", "backgroundColor": "#10b98166",
                "fill": False, "tension": 0.1
            }]
        },
        "options": {
            "scales": {"x": {"title": {"display": True, "text": "Date"}},
                       "y": {"title": {"display": True, "text": "Price (RWF)"}, "beginAtZero": True}},
            "plugins": {"legend": {"display": True}, "title": {"display": True, "text": title}}
        }
    }
    return json.dumps(chart_config)

# Nutrient requirements for crops
crop_nutrient_requirements = {
    'Potatoes': {'pH': (5.0, 6.5), 'Nitrogen': (30, 50), 'Phosphorus': (20, 40), 'Potassium': (150, 250)},
    'Carrots': {'pH': (6.0, 7.0), 'Nitrogen': (20, 40), 'Phosphorus': (20, 40), 'Potassium': (120, 200)},
    'Beans': {'pH': (6.0, 7.5), 'Nitrogen': (10, 20), 'Phosphorus': (15, 30), 'Potassium': (100, 180)},
    'Tomatoes': {'pH': (6.0, 6.8), 'Nitrogen': (50, 70), 'Phosphorus': (40, 60), 'Potassium': (200, 300)},
    'Rice': {'pH': (5.5, 7.0), 'Nitrogen': (60, 100), 'Phosphorus': (20, 40), 'Potassium': (30, 60)},
}

def suggest_soil_nutrients(crop_name, current_soil_data):
    if crop_name not in crop_nutrient_requirements:
        return [f"Sorry, we do not have nutrient data for {crop_name}"]

    suggestions = []
    crop_requirements = crop_nutrient_requirements[crop_name]
    
    for nutrient, (min_value, max_value) in crop_requirements.items():
        current_value = current_soil_data.get(nutrient)
        if current_value is None:
            suggestions.append(f"Missing {nutrient} data.")
        elif current_value < min_value:
            suggestions.append(f"Increase {nutrient}. Current: {current_value:.2f}, Recommended: {min_value}-{max_value}.")
        elif current_value > max_value:
            suggestions.append(f"Decrease {nutrient}. Current: {current_value:.2f}, Recommended: {min_value}-{max_value}.")
        else:
            suggestions.append(f"{nutrient} level is optimal. Current: {current_value:.2f}.")

    return suggestions

def make_arima_prediction(model, crop, steps):
    if model is None:
        return []
    try:
        forecast = model.forecast(steps=steps).tolist()
        if crop == 'beans':
            forecast = [price + 430 for price in forecast]
        return forecast
    except Exception as e:
        logger.error(f"Error during ARIMA prediction for {crop}: {e}")
        return []

def send_reset_email(user):
    token = user.get_reset_token()
    msg = Message('Password Reset Request', sender=current_app.config['MAIL_USERNAME'], recipients=[user.email])
    msg.body = f'''To reset your password, visit the following link:
{url_for('reset_token', token=token, _external=True)}

If you did not make this request then simply ignore this email and no changes will be made.
'''
    try:
        mail.send(msg)
    except Exception as e:
        logger.error(f"Failed to send email to {user.email}: {e}")
        flash('Failed to send password reset email.', 'danger')

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html', current_user=current_user)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        return predict_price()
    
    return render_template('predict.html', current_user=current_user)

@app.route('/predict_price', methods=['POST'])
@login_required
def predict_price():
    if model_pipeline is None:
        return jsonify({"error": "Prediction model not loaded. Please contact support."}), 500

    try:
        json_data = request.get_json(force=True)
        input_df = pd.DataFrame([json_data])
        for feature in EXPECTED_FEATURES:
            if feature not in input_df.columns:
                input_df[feature] = 0 if feature in ['year', 'month', 'day_of_week', 'day_of_year'] else 'unknown'
        input_df = input_df[EXPECTED_FEATURES]
        
        predicted_usd_price = model_pipeline.predict(input_df)[0]
        predicted_rwf_price = predicted_usd_price * USD_TO_RWF_EXCHANGE_RATE

        prediction = Prediction(
            user_id=current_user.id,
            input_data=json_data,
            predicted_usd_price=predicted_usd_price,
            predicted_rwf_price=predicted_rwf_price
        )
        db.session.add(prediction)
        db.session.commit()
        logger.info(f"Prediction saved for user {current_user.username}: USD {predicted_usd_price}, RWF {predicted_rwf_price}")

        return jsonify({
            "predicted_usdprice": float(predicted_usd_price),
            "predicted_rwfprice": float(predicted_rwf_price),
            "input_data": json_data
        })
    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}")
        return jsonify({"error": f"An unexpected error occurred during prediction: {str(e)}"}), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.check_password(form.password.data):
            login_user(user, remember=form.remember_me.data)
            next_page = request.args.get('next')
            flash('Logged in successfully!', 'success')
            return redirect(next_page or url_for('index'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html', title='Login', form=form)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = SignupForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html', title='Sign Up', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password_request():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = ForgotPasswordForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user:
            send_reset_email(user)
            flash('An email has been sent with instructions to reset your password.', 'info')
        else:
            flash('If an account with that email exists, a password reset email has been sent.', 'info')
        return redirect(url_for('login'))
    return render_template('forgot_password.html', title='Forgot Password', form=form)

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_token(token):
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    user = User.verify_reset_token(token)
    if user is None:
        flash('That is an invalid or expired token', 'warning')
        return redirect(url_for('forgot_password_request'))
    form = ResetPasswordForm()
    if form.validate_on_submit():
        user.set_password(form.password.data)
        db.session.commit()
        flash('Your password has been updated! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('reset_password.html', title='Reset Password', form=form)

@app.route('/admin_dashboard')
@admin_required
def admin_dashboard():
    users = User.query.all()
    return render_template('admin_dashboard.html', title='Admin Dashboard', users=users, current_user=current_user)

@app.route('/delete_user/<int:user_id>', methods=['POST'])
@admin_required
def delete_user(user_id):
    user_to_delete = db.session.get(User, user_id)
    if not user_to_delete or user_to_delete.is_admin or user_to_delete.id == current_user.id:
        flash('Cannot delete this user.', 'danger')
        return redirect(url_for('admin_dashboard'))
    db.session.delete(user_to_delete)
    db.session.commit()
    flash(f'User "{user_to_delete.username}" has been deleted.', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/visualize_data', methods=['GET', 'POST'])
@login_required
def visualize_data():
    df = load_dataset()
    plot_json = {}
    
    commodities = sorted(df['commodity'].unique().tolist()) if not df.empty else []
    admin1s = sorted(df['admin1'].unique().tolist()) if not df.empty else []
    pricetypes = sorted(df['pricetype'].unique().tolist()) if not df.empty else []
    markets = sorted(df['market'].unique().tolist()) if not df.empty else []

    selected_commodity, selected_admin1, selected_pricetype, selected_market, selected_start_date, selected_end_date = None, None, None, None, None, None

    if request.method == 'POST':
        selected_commodity = request.form.get('commodity')
        selected_admin1 = request.form.get('admin1')
        selected_pricetype = request.form.get('pricetype')
        selected_market = request.form.get('market')
        selected_start_date = request.form.get('start_date')
        selected_end_date = request.form.get('end_date')

    if not df.empty:
        plot_json = create_dynamic_price_trend_plot(
            df, selected_commodity, selected_admin1, selected_pricetype, selected_market, selected_start_date, selected_end_date
        )
        if not plot_json:
            flash('No data available for the selected filters.', 'warning')
    else:
        flash('Could not load data for visualization.', 'danger')

    return render_template('visualization.html',
                           plot_json=plot_json, current_user=current_user,
                           commodities=commodities, admin1s=admin1s, pricetypes=pricetypes, markets=markets,
                           selected_commodity=selected_commodity, selected_admin1=selected_admin1,
                           selected_pricetype=selected_pricetype, selected_market=selected_market,
                           selected_start_date=selected_start_date, selected_end_date=selected_end_date)

@app.route('/prediction_history', methods=['GET'])
@login_required
def prediction_history():
    selected_commodity = request.args.get('commodity')
    selected_start_date = request.args.get('start_date')
    selected_end_date = request.args.get('end_date')

    query = Prediction.query.filter_by(user_id=current_user.id)
    if selected_commodity:
        query = query.filter(Prediction.input_data.as_json_path('$.commodity') == selected_commodity)
    if selected_start_date:
        query = query.filter(Prediction.timestamp >= pd.to_datetime(selected_start_date))
    if selected_end_date:
        query = query.filter(Prediction.timestamp <= pd.to_datetime(selected_end_date))

    predictions = query.order_by(Prediction.timestamp.desc()).all()
    commodities = sorted({pred.input_data.get('commodity', '') for pred in Prediction.query.filter_by(user_id=current_user.id).all() if pred.input_data.get('commodity')})

    if not predictions:
        flash('No predictions found for the selected filters.', 'info')

    labels = [pred.timestamp.strftime('%Y-%m-%d %H:%M:%S') for pred in predictions]
    rwf_prices = [pred.predicted_rwf_price for pred in predictions]
    usd_prices = [pred.predicted_usd_price for pred in predictions]

    chart_config = {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [
                {"label": "Price (RWF)", "data": rwf_prices, "borderColor": "#10b981", "yAxisID": "y-rwf", "tension": 0.1},
                {"label": "Price (USD)", "data": usd_prices, "borderColor": "#f59e0b", "yAxisID": "y-usd", "tension": 0.1}
            ]
        },
        "options": {
            "scales": {
                "x": {"title": {"display": True, "text": "Timestamp"}},
                "y-rwf": {"type": "linear", "position": "left", "title": {"display": True, "text": "Price (RWF)"}},
                "y-usd": {"type": "linear", "position": "right", "title": {"display": True, "text": "Price (USD)"}, "grid": {"drawOnChartArea": False}}
            },
            "plugins": {"legend": {"display": True}, "title": {"display": True, "text": f"Prediction History for {current_user.username}"}}
        }
    }
    plot_json = json.dumps(chart_config)

    return render_template('prediction_history.html',
                           plot_json=plot_json, predictions=predictions, current_user=current_user,
                           commodities=commodities, selected_commodity=selected_commodity,
                           selected_start_date=selected_start_date, selected_end_date=selected_end_date)

@app.route('/sensor_dashboard')
def sensor_dashboard():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM sensor_data ORDER BY date DESC')
    rows = cursor.fetchall()
    conn.close()

    sensor_data = [{
        'serial_number': row[1], 'temperature': row[2], 'humidity': row[3], 'nitrogen': row[4],
        'potassium': row[5], 'moisture': row[6], 'eclec': row[7], 'phosphorus': row[8],
        'soilPH': row[9], 'latitude': row[10], 'longitude': row[11], 'date': row[12]
    } for row in rows]

    chart_data = {
        'temperature': [row[2] for row in rows], 'humidity': [row[3] for row in rows], 'nitrogen': [row[4] for row in rows],
        'potassium': [row[5] for row in rows], 'moisture': [row[6] for row in rows], 'phosphorus': [row[8] for row in rows],
        'soilPH': [row[9] for row in rows], 'dates': [row[12] for row in rows]
    }

    map_data = [{
        "lat": row[10], "lng": row[11], "serial_number": row[1], "temperature": row[2],
        "humidity": row[3], "nitrogen": row[4], "potassium": row[5], "moisture": row[6],
        "eclec": row[7], "phosphorus": row[8], "soilPH": row[9], "date": row[12]
    } for row in rows]

    latest_data = sensor_data[0] if sensor_data else {}
    current_soil_data = {
        'pH': latest_data.get('soilPH'), 'Nitrogen': latest_data.get('nitrogen'),
        'Phosphorus': latest_data.get('phosphorus'), 'Potassium': latest_data.get('potassium')
    }
    
    suggestions_for_all_crops = {
        crop: suggest_soil_nutrients(crop, current_soil_data)
        for crop in crop_nutrient_requirements.keys()
    }

    return render_template('sensor_dashboard.html',
                           sensor_data=sensor_data, chart_data=chart_data, map_data=map_data,
                           suggestions_for_all_crops=suggestions_for_all_crops,
                           suggestions=None, price_predictions=None, current_user=current_user)

@app.route('/store_sensor_data', methods=['POST'])
def store_sensor_data():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    data = request.get_json()
    try:
        cursor.execute('''
            INSERT INTO sensor_data (serial_number, temperature, humidity, nitrogen, potassium, moisture, eclec, phosphorus, soilPH, latitude, longitude, date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.get('serial_number', 'unknown'), data.get('temperature', 0.0), data.get('humidity', 0.0),
            data.get('nitrogen', 0.0), data.get('potassium', 0.0), data.get('moisture', 0.0),
            data.get('eclec', 0.0), data.get('phosphorus', 0.0), data.get('soilPH', 0.0),
            data.get('latitude', 0.0), data.get('longitude', 0.0), datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
        conn.commit()
        return jsonify({"status": "success", "message": "Data stored successfully"}), 200
    except Exception as e:
        logger.error(f"Error storing sensor data: {e}")
        return jsonify({"status": "error", "message": "Failed to store data"}), 500
    finally:
        conn.close()

@app.route('/suggest_nutrients', methods=['POST'])
def suggest_nutrients():
    selected_crop = request.form.get('crop').capitalize()

    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM sensor_data ORDER BY date DESC LIMIT 1')
    latest_row = cursor.fetchone()
    conn.close()

    current_soil_data = {}
    if latest_row:
        current_soil_data = {
            'pH': latest_row[9], 'Nitrogen': latest_row[4], 'Phosphorus': latest_row[8], 'Potassium': latest_row[5]
        }

    suggestions = suggest_soil_nutrients(selected_crop, current_soil_data)
    price_predictions = make_arima_prediction(arima_models.get(selected_crop.lower()), selected_crop.lower(), steps=12)

    return render_template('sensor_dashboard.html',
                           suggestions=suggestions, selected_crop=selected_crop, price_predictions=price_predictions,
                           sensor_data=[], chart_data={}, map_data={}, current_user=current_user)

@app.route('/irrigation_control')
@login_required
def irrigation_control():
    devices = IrrigationDevice.query.filter_by(is_active=True).all()
    return render_template('irrigation_control.html', devices=devices, current_user=current_user)

@app.route('/add_irrigation_device', methods=['POST'])
@login_required
def add_irrigation_device():
    try:
        device_name = request.form.get('device_name')
        location = request.form.get('location')
        latitude = float(request.form.get('latitude', 0))
        longitude = float(request.form.get('longitude', 0))
        
        device = IrrigationDevice(
            device_name=device_name,
            location=location,
            latitude=latitude,
            longitude=longitude
        )
        db.session.add(device)
        db.session.flush()
        
        for i in range(1, 5):
            valve = IrrigationValve(
                device_id=device.id,
                valve_number=i,
                valve_name=f"Valve {i}"
            )
            db.session.add(valve)
        
        for i in range(1, 5):
            moisture_sensor = MoistureSensor(
                device_id=device.id,
                sensor_number=i,
                sensor_name=f"Moisture Sensor {i}"
            )
            db.session.add(moisture_sensor)
        
        for i in range(1, 5):
            dht_sensor = DHTSensor(
                device_id=device.id,
                sensor_number=i,
                sensor_name=f"DHT Sensor {i}"
            )
            db.session.add(dht_sensor)
        
        db.session.commit()
        flash(f'Irrigation device "{device_name}" added successfully!', 'success')
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error adding irrigation device: {e}")
        flash('Error adding irrigation device. Please try again.', 'danger')
    
    return redirect(url_for('irrigation_control'))

@app.route('/toggle_valve/<int:valve_id>', methods=['POST'])
@login_required
def toggle_valve(valve_id):
    try:
        valve = IrrigationValve.query.get_or_404(valve_id)
        valve.is_open = not valve.is_open
        
        if valve.is_open:
            valve.last_activated = datetime.utcnow()
        
        db.session.commit()
        
        status = "opened" if valve.is_open else "closed"
        flash(f'Valve {valve.valve_name} has been {status}!', 'success')
        
        return jsonify({
            'success': True,
            'valve_id': valve_id,
            'is_open': valve.is_open,
            'message': f'Valve {status} successfully'
        })
        
    except Exception as e:
        logger.error(f"Error toggling valve {valve_id}: {e}")
        return jsonify({
            'success': False,
            'message': 'Error controlling valve'
        }), 500

@app.route('/update_sensor_data/<int:device_id>', methods=['POST'])
@login_required
def update_sensor_data(device_id):
    try:
        data = request.get_json()
        device = IrrigationDevice.query.get_or_404(device_id)
        
        for i, moisture_data in enumerate(data.get('moisture_sensors', []), 1):
            sensor = MoistureSensor.query.filter_by(device_id=device_id, sensor_number=i).first()
            if sensor:
                sensor.moisture_level = moisture_data.get('level', 0)
                sensor.last_reading = datetime.utcnow()
        
        for i, dht_data in enumerate(data.get('dht_sensors', []), 1):
            sensor = DHTSensor.query.filter_by(device_id=device_id, sensor_number=i).first()
            if sensor:
                sensor.temperature = dht_data.get('temperature', 0)
                sensor.humidity = dht_data.get('humidity', 0)
                sensor.last_reading = datetime.utcnow()
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Sensor data updated successfully'
        })
        
    except Exception as e:
        logger.error(f"Error updating sensor data for device {device_id}: {e}")
        return jsonify({
            'success': False,
            'message': 'Error updating sensor data'
        }), 500

@app.route('/get_device_data/<int:device_id>')
@login_required
def get_device_data(device_id):
    try:
        device = IrrigationDevice.query.get_or_404(device_id)
        
        device_data = {
            'id': device.id,
            'name': device.device_name,
            'location': device.location,
            'valves': [{
                'id': valve.id,
                'number': valve.valve_number,
                'name': valve.valve_name,
                'is_open': valve.is_open,
                'flow_rate': valve.flow_rate,
                'total_runtime': valve.total_runtime
            } for valve in device.valves],
            'moisture_sensors': [{
                'id': sensor.id,
                'number': sensor.sensor_number,
                'name': sensor.sensor_name,
                'level': sensor.moisture_level,
                'threshold_min': sensor.threshold_min,
                'threshold_max': sensor.threshold_max,
                'last_reading': sensor.last_reading.isoformat() if sensor.last_reading else None
            } for sensor in device.moisture_sensors],
            'dht_sensors': [{
                'id': sensor.id,
                'number': sensor.sensor_number,
                'name': sensor.sensor_name,
                'temperature': sensor.temperature,
                'humidity': sensor.humidity,
                'last_reading': sensor.last_reading.isoformat() if sensor.last_reading else None
            } for sensor in device.dht_sensors]
        }
        
        return jsonify(device_data)
        
    except Exception as e:
        logger.error(f"Error getting device data for device {device_id}: {e}")
        return jsonify({'error': 'Device not found'}), 404

@app.route('/record_harvest', methods=['GET', 'POST'])
@login_required
def record_harvest():
    if request.method == 'POST':
        try:
            crop = request.form.get('crop')
            quantity = float(request.form.get('quantity'))
            harvest_date = request.form.get('harvest_date')
            notes = request.form.get('notes')

            if not crop or not quantity or not harvest_date:
                flash('Crop, quantity, and harvest date are required.', 'danger')
                return redirect(url_for('record_harvest'))

            if crop not in crop_nutrient_requirements:
                flash(f'Invalid crop selected: {crop}.', 'danger')
                return redirect(url_for('record_harvest'))

            if quantity <= 0:
                flash('Quantity must be greater than zero.', 'danger')
                return redirect(url_for('record_harvest'))

            try:
                harvest_date = pd.to_datetime(harvest_date).to_pydatetime()
            except ValueError:
                flash('Invalid harvest date format. Use YYYY-MM-DD.', 'danger')
                return redirect(url_for('record_harvest'))

            harvest = Harvest(
                user_id=current_user.id,
                crop=crop,
                quantity=quantity,
                harvest_date=harvest_date,
                notes=notes
            )
            db.session.add(harvest)
            db.session.commit()
            flash(f'Harvest of {quantity} kg of {crop} recorded successfully!', 'success')
            return redirect(url_for('harvest_history'))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error recording harvest: {e}")
            flash('Error recording harvest. Please try again.', 'danger')
            return redirect(url_for('record_harvest'))

    crops = sorted(crop_nutrient_requirements.keys())
    return render_template('harvest.html', current_user=current_user, crops=crops, mode='record')

@app.route('/harvest_history', methods=['GET', 'POST'])
@login_required
def harvest_history():
    selected_crop = request.form.get('crop') if request.method == 'POST' else request.args.get('crop')
    selected_start_date = request.form.get('start_date') if request.method == 'POST' else request.args.get('start_date')
    selected_end_date = request.form.get('end_date') if request.method == 'POST' else request.args.get('end_date')

    query = Harvest.query.filter_by(user_id=current_user.id)
    if selected_crop:
        query = query.filter_by(crop=selected_crop)
    if selected_start_date:
        query = query.filter(Harvest.harvest_date >= pd.to_datetime(selected_start_date))
    if selected_end_date:
        query = query.filter(Harvest.harvest_date <= pd.to_datetime(selected_end_date))

    harvests = query.order_by(Harvest.harvest_date.desc()).all()
    crops = sorted({harvest.crop for harvest in Harvest.query.filter_by(user_id=current_user.id).all()})

    if not harvests:
        flash('No harvests found for the selected filters.', 'info')

    labels = [harvest.harvest_date.strftime('%Y-%m-%d') for harvest in harvests]
    quantities = [harvest.quantity for harvest in harvests]

    chart_config = {
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": [{
                "label": "Harvest Quantity (kg)",
                "data": quantities,
                "backgroundColor": "#10b981",
                "borderColor": "#10b981",
                "borderWidth": 1
            }]
        },
        "options": {
            "scales": {
                "x": {"title": {"display": True, "text": "Harvest Date"}},
                "y": {"title": {"display": True, "text": "Quantity (kg)"}, "beginAtZero": True}
            },
            "plugins": {
                "legend": {"display": True},
                "title": {"display": True, "text": f"Harvest History for {current_user.username}"}
            }
        }
    }
    plot_json = json.dumps(chart_config)

    return render_template('harvest.html',
                           plot_json=plot_json, harvests=harvests, crops=crops,
                           selected_crop=selected_crop, selected_start_date=selected_start_date,
                           selected_end_date=selected_end_date, current_user=current_user, mode='history')

@app.route('/delete_harvest/<int:harvest_id>', methods=['POST'])
@login_required
def delete_harvest(harvest_id):
    try:
        harvest = Harvest.query.get_or_404(harvest_id)
        if harvest.user_id != current_user.id and not current_user.is_admin:
            flash('You do not have permission to delete this harvest.', 'danger')
            return redirect(url_for('harvest_history'))
        db.session.delete(harvest)
        db.session.commit()
        flash(f'Harvest of {harvest.quantity} kg of {harvest.crop} deleted successfully.', 'success')
    except Exception as e:
        logger.error(f"Error deleting harvest {harvest_id}: {e}")
        flash('Error deleting harvest. Please try again.', 'danger')
    return redirect(url_for('harvest_history'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sensor_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            serial_number TEXT, temperature REAL, humidity REAL, nitrogen REAL,
            potassium REAL, moisture REAL, eclec REAL, phosphorus REAL,
            soilPH REAL, latitude REAL, longitude REAL, date TEXT
        )
        ''')
        conn.commit()
        conn.close()

        if not User.query.filter_by(username='admin').first():
            admin_password = secrets.token_urlsafe(16)
            admin_user = User(username='admin', email='admin@example.com', is_admin=True)
            admin_user.set_password(admin_password)
            db.session.add(admin_user)
            db.session.commit()
            logger.info("\n--- Initial Setup ---")
            logger.info("Default admin user created:")
            logger.info("Username: admin")
            logger.info(f"Password: {admin_password}")
            logger.info("!!! PLEASE CHANGE THIS PASSWORD IMMEDIATELY AFTER FIRST LOGIN IN PRODUCTION !!!")
            logger.info("---------------------\n")
        if not User.query.filter_by(username='testuser').first():
            default_user = User(username='testuser', email='test@example.com', is_admin=False)
            default_user.set_password('testpass')
            db.session.add(default_user)
            db.session.commit()
            logger.info("Default regular user created: username 'testuser', password 'testpass'")

    app.run(debug=os.getenv('FLASK_DEBUG', 'True') == 'True', host='0.0.0.0', port=5000)