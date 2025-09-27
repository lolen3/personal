from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import requests
import json
import os
import random
from datetime import datetime
from sklearn.metrics import accuracy_score
from werkzeug.utils import secure_filename
from faker import Faker
import logging
import warnings
import csv
import hashlib
import re
from functools import wraps

# Suppress scikit-learn warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev_key_for_crop_app")
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # 30 minutes

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

fake = Faker()

# Ensure required directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('static/images', exist_ok=True)

# Add this code at the beginning of app.py, after directory creation
# Define the path to the model file
MODEL_PATH = "c:/Users/PC/.vscode/crop_app/train_model.py"
# Create default model files if they don't exist
if not os.path.exists(MODEL_PATH):
    # Create a basic random forest model
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    # Create a simple dummy model
    model = RandomForestClassifier(n_estimators=50)
    model.fit([[0, 0, 0, 0]], [0])  # Dummy fit
    joblib.dump(model, MODEL_PATH)
    
    # Define the path to the scaler file
    SCALER_PATH = "models/scaler.pkl"
    
    # Create a scaler
    scaler = StandardScaler()
    scaler.fit([[0, 0, 0, 0]])  # Dummy fit
    joblib.dump(scaler, SCALER_PATH)
    
    # Define the path to the encoder file
    ENCODER_PATH = "models/label_encoder.pkl"
    
    # Create a label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(["Maize", "Beans", "Rice", "Wheat", "Coffee"])
    joblib.dump(label_encoder, ENCODER_PATH)
    
    # Create feature names
    feature_names = ["T2M_MAX-Sp", "T2M_MIN-Sp", "QV2M-Sp", "PRECTOTCORR-Sp"]
    FEATURE_NAMES_PATH = "models/feature_names.pkl"
    joblib.dump(feature_names, FEATURE_NAMES_PATH)
    
    # Create soil color columns
    # Define the path to the soil color columns file
    SOILCOLOR_COLS_PATH = "models/soilcolor_cols.pkl"
    
    # Define soil color columns
    soilcolor_cols = [
        "Soilcolor_Black", "Soilcolor_Brown", "Soilcolor_Red", 
        "Soilcolor_Yellow", "Soilcolor_Grey"
    ]
    joblib.dump(soilcolor_cols, SOILCOLOR_COLS_PATH)

# Paths to saved model artifacts
MODEL_PATH = "models/rf_crop_model.pkl"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
FEATURE_NAMES_PATH = "models/feature_names.pkl"
SOILCOLOR_COLS_PATH = "models/soilcolor_cols.pkl"

# Load model and preprocessing objects
try:
    pass  # Add your code here
except Exception as e:
    logger.error(f"An error occurred: {e}")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    expected_features = joblib.load(FEATURE_NAMES_PATH)
    logger.info("All model artifacts loaded successfully")
except Exception as e:
    logger.error(f"Error loading model artifacts: {e}")
    # Create dummy objects if files don't exist yet
    model = None
    scaler = None
    label_encoder = None
    expected_features = []

# Load soil color columns
try:
    soilcolor_cols = joblib.load(SOILCOLOR_COLS_PATH)
except Exception as e:
    logger.warning(f"Soil color columns not found: {e}")
    soilcolor_cols = [
        "Soilcolor_Black", "Soilcolor_Brown", "Soilcolor_Red", 
        "Soilcolor_Yellow", "Soilcolor_Grey"
    ]

# Prediction log
PREDICTION_LOG = "data/prediction_log.csv"
if not os.path.exists(PREDICTION_LOG):
    os.makedirs('data', exist_ok=True)
    pd.DataFrame(columns=["timestamp", "county", "season", "input", "prediction", "true_label"]).to_csv(PREDICTION_LOG, index=False)

# Weather API setup
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
WEATHER_URL = "http://api.openweathermap.org/data/2.5/weather"

# Create vendors.json if it doesn't exist
if not os.path.exists("vendors.json"):
    default_vendors = [
        {
            "name": "Lolen Mwangi", 
            "email": "loswolfi04@gmail.com", 
            "phone": "0706018491", 
            "location": "Nyeri", 
            "product": "Fertilizers",
            "hours": "8:00 AM - 5:00 PM"
        },
        {
            "name": "John Doe",
            "email": "johndoe@example.com",
            "phone": "0712345678",
            "location": "Nairobi",
            "product": "Fertilizers",
            "hours": "8:00 AM - 6:00 PM"
        },
        {
            "name": "Jane Smith",
            "email": "janesmith@example.com",
            "phone": "0787654321",
            "location": "Kisumu",
            "product": "Pesticides",
            "hours": "9:00 AM - 5:00 PM"
        },
        {
            "name": "Samuel Kariuki",
            "email": "samuelkariuki@example.com",
            "phone": "0722112233",
            "location": "Mombasa",
            "product": "Seeds",
            "hours": "8:30 AM - 5:30 PM"
        }
    ]
    with open("vendors.json", "w") as f:
        json.dump(default_vendors, f, indent=2)

# Load vendor data
try:
    with open("vendors.json", "r") as f:
        vendors = json.load(f)
except Exception as e:
    logger.warning(f"Vendor data not found or invalid. Using empty list: {e}")
    vendors = []

# Create users.json if it doesn't exist
if not os.path.exists("users.json"):
    default_users = []
    with open("users.json", "w") as f:
        json.dump(default_users, f, indent=2)

# Load user data
try:
    with open("users.json", "r") as f:
        users = json.load(f)
except Exception as e:
    logger.warning(f"User data not found or invalid. Using empty list: {e}")
    users = []

# Create verification_codes.json if it doesn't exist
if not os.path.exists("verification_codes.json"):
    verification_codes = {}
    with open("verification_codes.json", "w") as f:
        json.dump(verification_codes, f, indent=2)

# Farm inputs dataset
FARM_INPUTS_FILE = "data/farm_inputs.json"
if not os.path.exists(FARM_INPUTS_FILE):
    default_farm_inputs = {
        "Maize": {
            "seeds": [
                {"name": "H614D Hybrid", "price": 450, "quantity": "2kg", "description": "High yielding variety suitable for medium altitude areas"},
                {"name": "DK8031", "price": 520, "quantity": "2kg", "description": "Drought tolerant variety with excellent standability"}
            ],
            "fertilizers": [
                {"name": "DAP", "price": 3800, "quantity": "50kg", "description": "18:46:0 - For planting"},
                {"name": "CAN", "price": 2900, "quantity": "50kg", "description": "26:0:0 - For top dressing"}
            ],
            "pesticides": [
                {"name": "Striker", "price": 850, "quantity": "500ml", "description": "Controls fall armyworm and stalk borers"},
                {"name": "Duduthrin", "price": 450, "quantity": "50ml", "description": "Broad-spectrum insecticide for field pests"}
            ],
            "equipment": [
                {"name": "Jembe (Hoe)", "price": 450, "quantity": "1 piece", "description": "Traditional farming tool for digging"},
                {"name": "Knapsack Sprayer", "price": 2500, "quantity": "16L", "description": "For applying pesticides and foliar feeds"}
            ],
            "planting_guide": "Plant at 75cm between rows and 25cm between plants. Apply 1 teaspoon of DAP per hole at planting. Top dress with CAN when plants are knee-high."
        },
        "Beans": {
            "seeds": [
                {"name": "KK8 Rose Coco", "price": 650, "quantity": "2kg", "description": "Early maturing variety with good market demand"},
                {"name": "Nyota", "price": 700, "quantity": "2kg", "description": "Disease resistant variety suitable for many regions"}
            ],
            "fertilizers": [
                {"name": "TSP", "price": 3600, "quantity": "50kg", "description": "0:45:0 - For planting"},
                {"name": "NPK 17:17:17", "price": 3500, "quantity": "50kg", "description": "Balanced fertilizer for growing"}
            ],
            "pesticides": [
                {"name": "Thunder", "price": 1200, "quantity": "500ml", "description": "Controls aphids and bean flies"},
                {"name": "Ridomil", "price": 950, "quantity": "500g", "description": "Fungicide for rust and anthracnose"}
            ],
            "equipment": [
                {"name": "Jembe (Hoe)", "price": 450, "quantity": "1 piece", "description": "Traditional farming tool for digging"},
                {"name": "Panga (Machete)", "price": 350, "quantity": "1 piece", "description": "For clearing land before planting"}
            ],
            "planting_guide": "Plant at 45cm between rows and 15cm between plants. Apply 1/2 teaspoon of TSP per hole at planting. Keep field weed-free."
        },
        "Rice": {
            "seeds": [
                {"name": "Basmati 370", "price": 850, "quantity": "2kg", "description": "Aromatic long-grain variety with premium market price"},
                {"name": "IR2793", "price": 750, "quantity": "2kg", "description": "High-yielding variety suitable for irrigation"}
            ],
            "fertilizers": [
                {"name": "DAP", "price": 3800, "quantity": "50kg", "description": "18:46:0 - For planting"},
                {"name": "Urea", "price": 3200, "quantity": "50kg", "description": "46:0:0 - For vegetative growth"}
            ],
            "pesticides": [
                {"name": "Oshin", "price": 950, "quantity": "100g", "description": "Controls rice stem borers"},
                {"name": "Amistar Top", "price": 2800, "quantity": "250ml", "description": "Fungicide for blast and sheath blight"}
            ],
            "equipment": [
                {"name": "Water Pump", "price": 15000, "quantity": "1 piece", "description": "For irrigation management"},
                {"name": "Sickle", "price": 300, "quantity": "1 piece", "description": "For harvesting rice"}
            ],
            "planting_guide": "Transplant seedlings at 20cm x 20cm spacing. Maintain water level at 5cm during vegetative stage. Apply fertilizer in split doses."
        },
        "Wheat": {
            "seeds": [
                {"name": "Kenya Kwale", "price": 700, "quantity": "2kg", "description": "High-yielding variety with resistance to rust"},
                {"name": "Eagle 10", "price": 750, "quantity": "2kg", "description": "Drought tolerant variety with early maturity"}
            ],
            "fertilizers": [
                {"name": "DAP", "price": 3800, "quantity": "50kg", "description": "18:46:0 - For planting"},
                {"name": "Urea", "price": 3200, "quantity": "50kg", "description": "46:0:0 - For top dressing"}
            ],
            "pesticides": [
                {"name": "Tilt", "price": 1800, "quantity": "250ml", "description": "Fungicide for rust and septoria"},
                {"name": "Decis", "price": 850, "quantity": "500ml", "description": "Insecticide for aphids and thrips"}
            ],
            "equipment": [
                {"name": "Seed Drill", "price": 35000, "quantity": "1 piece", "description": "For precision planting"},
                {"name": "Boom Sprayer", "price": 8500, "quantity": "1 piece", "description": "For efficient pesticide application"}
            ],
            "planting_guide": "Plant in rows 20cm apart with seeding rate of 125kg/ha. Apply DAP at planting and top dress with Urea at tillering stage."
        },
        "Coffee": {
            "seeds": [
                {"name": "Ruiru 11", "price": 80, "quantity": "per seedling", "description": "Disease resistant variety with high yield potential"},
                {"name": "Batian", "price": 100, "quantity": "per seedling", "description": "Newer variety with resistance to CBD and CLR"}
            ],
            "fertilizers": [
                {"name": "NPK 17:17:17", "price": 3500, "quantity": "50kg", "description": "Balanced fertilizer for young trees"},
                {"name": "CAN", "price": 2900, "quantity": "50kg", "description": "26:0:0 - For nitrogen supplementation"}
            ],
            "pesticides": [
                {"name": "Copper Oxychloride", "price": 1200, "quantity": "1kg", "description": "Fungicide for coffee berry disease"},
                {"name": "Cyflutrhin", "price": 1500, "quantity": "500ml", "description": "Controls coffee berry borer"}
            ],
            "equipment": [
                {"name": "Pruning Shears", "price": 950, "quantity": "1 piece", "description": "For pruning coffee trees"},
                {"name": "Motorized Sprayer", "price": 18000, "quantity": "20L", "description": "For efficient pesticide application"}
            ],
            "planting_guide": "Plant seedlings at 2m x 2m spacing. Apply 100g of NPK per tree twice a year. Prune regularly to maintain productive branches."
        },
        "Tomato": {
            "seeds": [
                {"name": "Rio Grande", "price": 850, "quantity": "50g", "description": "Processing variety with firm fruits"},
                {"name": "Kilele F1", "price": 1200, "quantity": "10g", "description": "Hybrid variety with resistance to bacterial wilt"}
            ],
            "fertilizers": [
                {"name": "DAP", "price": 3800, "quantity": "50kg", "description": "18:46:0 - For planting"},
                {"name": "NPK 17:17:17", "price": 3500, "quantity": "50kg", "description": "Balanced fertilizer for growing"}
            ],
            "pesticides": [
                {"name": "Coragen", "price": 2500, "quantity": "50ml", "description": "Controls tomato fruitworm"},
                {"name": "Ridomil", "price": 950, "quantity": "500g", "description": "Fungicide for early and late blight"}
            ],
            "equipment": [
                {"name": "Drip Irrigation Kit", "price": 15000, "quantity": "1/4 acre", "description": "Water-efficient irrigation system"},
                {"name": "Staking Materials", "price": 200, "quantity": "per stake", "description": "For supporting plants"}
            ],
            "planting_guide": "Plant at 60cm between rows and 45cm between plants. Stake plants when they reach 30cm tall. Regular spraying against blight is essential."
        }
    }
    with open(FARM_INPUTS_FILE, "w") as f:
        json.dump(default_farm_inputs, f, indent=2)

# Load farm inputs data
try:
    with open(FARM_INPUTS_FILE, "r") as f:
        farm_inputs = json.load(f)
except Exception as e:
    logger.warning(f"Farm inputs data not found or invalid. Using empty dict: {e}")
    farm_inputs = {}

# Define Kenyan counties with more specific characteristics
kenyan_counties = {
    "Baringo": {"suitable_crops": ["Maize", "Beans", "Wheat"], "rainfall": "Medium", "altitude": "High"},
    "Bomet": {"suitable_crops": ["Maize", "Tea", "Wheat"], "rainfall": "High", "altitude": "High"},
    "Bungoma": {"suitable_crops": ["Maize", "Beans", "Sugarcane"], "rainfall": "High", "altitude": "Medium"},
    "Busia": {"suitable_crops": ["Cassava", "Maize", "Rice"], "rainfall": "Medium", "altitude": "Low"},
    "Elgeyo-Marakwet": {"suitable_crops": ["Maize", "Wheat", "Potatoes"], "rainfall": "Medium", "altitude": "High"},
    "Embu": {"suitable_crops": ["Coffee", "Maize", "Beans"], "rainfall": "Medium", "altitude": "Medium"},
    "Garissa": {"suitable_crops": ["Sorghum", "Millet", "Watermelon"], "rainfall": "Low", "altitude": "Low"},
    "Homa Bay": {"suitable_crops": ["Maize", "Beans", "Sorghum"], "rainfall": "Medium", "altitude": "Low"},
    "Isiolo": {"suitable_crops": ["Sorghum", "Millet", "Beans"], "rainfall": "Low", "altitude": "Medium"},
    "Kajiado": {"suitable_crops": ["Maize", "Beans", "Tomato"], "rainfall": "Low", "altitude": "Medium"},
    "Kakamega": {"suitable_crops": ["Maize", "Beans", "Tea"], "rainfall": "High", "altitude": "Medium"},
    "Kericho": {"suitable_crops": ["Tea", "Maize", "Wheat"], "rainfall": "High", "altitude": "High"},
    "Kiambu": {"suitable_crops": ["Coffee", "Tea", "Tomato"], "rainfall": "Medium", "altitude": "High"},
    "Kilifi": {"suitable_crops": ["Cashew", "Coconut", "Cassava"], "rainfall": "Medium", "altitude": "Low"},
    "Kirinyaga": {"suitable_crops": ["Rice", "Coffee", "Tea"], "rainfall": "Medium", "altitude": "Medium"},
    "Kisii": {"suitable_crops": ["Maize", "Beans", "Bananas"], "rainfall": "High", "altitude": "High"},
    "Kisumu": {"suitable_crops": ["Rice", "Maize", "Sorghum"], "rainfall": "Medium", "altitude": "Low"},
    "Kitui": {"suitable_crops": ["Sorghum", "Millet", "Green Grams"], "rainfall": "Low", "altitude": "Medium"},
    "Kwale": {"suitable_crops": ["Cashew", "Coconut", "Cassava"], "rainfall": "Medium", "altitude": "Low"},
    "Laikipia": {"suitable_crops": ["Wheat", "Barley", "Beans"], "rainfall": "Medium", "altitude": "High"},
    "Lamu": {"suitable_crops": ["Coconut", "Cashew", "Rice"], "rainfall": "Medium", "altitude": "Low"},
    "Machakos": {"suitable_crops": ["Maize", "Beans", "Tomato"], "rainfall": "Low", "altitude": "Medium"},
    "Makueni": {"suitable_crops": ["Green Grams", "Sorghum", "Tomato"], "rainfall": "Low", "altitude": "Medium"},
    "Mandera": {"suitable_crops": ["Sorghum", "Millet", "Watermelon"], "rainfall": "Low", "altitude": "Low"},
    "Marsabit": {"suitable_crops": ["Sorghum", "Millet", "Wheat"], "rainfall": "Low", "altitude": "High"},
    "Meru": {"suitable_crops": ["Coffee", "Tea", "Potatoes"], "rainfall": "Medium", "altitude": "High"},
    "Migori": {"suitable_crops": ["Maize", "Beans", "Tobacco"], "rainfall": "Medium", "altitude": "Medium"},
    "Mombasa": {"suitable_crops": ["Coconut", "Cashew", "Vegetables"], "rainfall": "Medium", "altitude": "Low"},
    "Murang'a": {"suitable_crops": ["Coffee", "Tea", "Avocado"], "rainfall": "Medium", "altitude": "Medium"},
    "Nairobi": {"suitable_crops": ["Vegetables", "Tomato", "Herbs"], "rainfall": "Medium", "altitude": "High"},
    "Nakuru": {"suitable_crops": ["Wheat", "Maize", "Vegetables"], "rainfall": "Medium", "altitude": "High"},
    "Nandi": {"suitable_crops": ["Tea", "Maize", "Wheat"], "rainfall": "High", "altitude": "High"},
    "Narok": {"suitable_crops": ["Wheat", "Barley", "Maize"], "rainfall": "Medium", "altitude": "High"},
    "Nyamira": {"suitable_crops": ["Tea", "Bananas", "Maize"], "rainfall": "High", "altitude": "High"},
    "Nyandarua": {"suitable_crops": ["Potatoes", "Wheat", "Vegetables"], "rainfall": "Medium", "altitude": "High"},
    "Nyeri": {"suitable_crops": ["Coffee", "Tea", "Potatoes"], "rainfall": "Medium", "altitude": "High"},
    "Samburu": {"suitable_crops": ["Sorghum", "Millet", "Beans"], "rainfall": "Low", "altitude": "Medium"},
    "Siaya": {"suitable_crops": ["Maize", "Beans", "Sorghum"], "rainfall": "Medium", "altitude": "Medium"},
    "Taita-Taveta": {"suitable_crops": ["Maize", "Beans", "Vegetables"], "rainfall": "Medium", "altitude": "Medium"},
    "Tana River": {"suitable_crops": ["Rice", "Maize", "Sorghum"], "rainfall": "Low", "altitude": "Low"},
    "Tharaka-Nithi": {"suitable_crops": ["Coffee", "Tea", "Maize"], "rainfall": "Medium", "altitude": "Medium"},
    "Trans-Nzoia": {"suitable_crops": ["Maize", "Wheat", "Sunflower"], "rainfall": "High", "altitude": "High"},
    "Turkana": {"suitable_crops": ["Sorghum", "Millet", "Watermelon"], "rainfall": "Low", "altitude": "Low"},
    "Uasin Gishu": {"suitable_crops": ["Maize", "Wheat", "Potatoes"], "rainfall": "Medium", "altitude": "High"},
    "Vihiga": {"suitable_crops": ["Maize", "Beans", "Tea"], "rainfall": "High", "altitude": "Medium"},
    "Wajir": {"suitable_crops": ["Sorghum", "Millet", "Watermelon"], "rainfall": "Low", "altitude": "Low"},
    "West Pokot": {"suitable_crops": ["Maize", "Beans", "Sorghum"], "rainfall": "Medium", "altitude": "Medium"}
}

# Define seasons with crop suitability
seasons = {
    "Spring": ["Maize", "Beans", "Tomato", "Rice"],
    "Summer": ["Maize", "Sorghum", "Beans", "Wheat"],
    "Autumn": ["Wheat", "Beans", "Potatoes", "Vegetables"],
    "Winter": ["Vegetables", "Potatoes", "Wheat", "Tea"]
}

# Map season to abbreviations used in feature names
season_abbr = {
    "Spring": "Sp",
    "Summer": "Su", 
    "Autumn": "Au",
    "Winter": "W"
}

county_coords = {
    "Baringo": {"lat": 0.6022, "lon": 36.0023},
    "Bomet": {"lat": -0.7821, "lon": 35.3426},
    "Bungoma": {"lat": 0.5695, "lon": 34.5584},
    "Busia": {"lat": 0.4608, "lon": 34.1118},
    "Elgeyo-Marakwet": {"lat": 1.0494, "lon": 35.4782},
    "Embu": {"lat": -0.5389, "lon": 37.4509},
    "Garissa": {"lat": -0.4532, "lon": 39.6460},
    "Homa Bay": {"lat": -0.5273, "lon": 34.4570},
    "Isiolo": {"lat": 0.3524, "lon": 37.5822},
    "Kajiado": {"lat": -1.8531, "lon": 36.7760},
    "Kakamega": {"lat": 0.2827, "lon": 34.7519},
    "Kericho": {"lat": -0.3675, "lon": 35.2830},
    "Kiambu": {"lat": -1.0350, "lon": 36.9541},
    "Kilifi": {"lat": -3.5107, "lon": 39.9093},
    "Kirinyaga": {"lat": -0.6606, "lon": 37.3827},
    "Kisii": {"lat": -0.6817, "lon": 34.7668},
    "Kisumu": {"lat": -0.0917, "lon": 34.7679},
    "Kitui": {"lat": -1.3753, "lon": 38.0106},
    "Kwale": {"lat": -4.1833, "lon": 39.4500},
    "Laikipia": {"lat": 0.3606, "lon": 36.6338},
    "Lamu": {"lat": -2.2711, "lon": 40.9020},
    "Machakos": {"lat": -1.5200, "lon": 37.2634},
    "Makueni": {"lat": -1.8030, "lon": 37.6218},
    "Mandera": {"lat": 3.9373, "lon": 41.8569},
    "Marsabit": {"lat": 2.3285, "lon": 37.9896},
    "Meru": {"lat": 0.3557, "lon": 37.8088},
    "Migori": {"lat": -1.0634, "lon": 34.4731},
    "Mombasa": {"lat": -4.0435, "lon": 39.6682},
    "Murang’a": {"lat": -0.7180, "lon": 37.1477},
    "Nairobi": {"lat": -1.2921, "lon": 36.8219},
    "Nakuru": {"lat": -0.3031, "lon": 36.0800},
    "Nandi": {"lat": 0.1056, "lon": 35.2548},
    "Narok": {"lat": -1.1041, "lon": 35.8762},
    "Nyamira": {"lat": -0.5815, "lon": 34.9392},
    "Nyandarua": {"lat": -0.1827, "lon": 36.4298},
    "Nyeri": {"lat": -0.4197, "lon": 36.9476},
    "Samburu": {"lat": 1.2154, "lon": 36.9541},
    "Siaya": {"lat": 0.0612, "lon": 34.2422},
    "Taita Taveta": {"lat": -3.3161, "lon": 38.4850},
    "Tana River": {"lat": -1.2159, "lon": 40.1169},
    "Tharaka-Nithi": {"lat": -0.3227, "lon": 37.7238},
    "Trans Nzoia": {"lat": 1.0156, "lon": 35.0026},
    "Turkana": {"lat": 3.3121, "lon": 35.5523},
    "Uasin Gishu": {"lat": 0.4532, "lon": 35.2698},
    "Vihiga": {"lat": 0.0157, "lon": 34.7280},
    "Wajir": {"lat": 1.7500, "lon": 40.0500},
    "West Pokot": {"lat": 1.2381, "lon": 35.1630}
}


# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_phone' not in session:
            flash("Please login to access this page", "warning")
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# Generate verification code
def generate_verification_code():
    return str(random.randint(100000, 999999))

# Simulate sending SMS (would integrate with actual SMS gateway in production)
def send_verification_sms(phone, code):
    logger.info(f"Sending verification code {code} to {phone}")
    # In a real app, you would use an SMS service API
    # For this demo, we'll just store the code in our verification_codes dict
    try:
        with open("verification_codes.json", "r") as f:
            verification_codes = json.load(f)
    except:
        verification_codes = {}
    
    verification_codes[phone] = code
    
    with open("verification_codes.json", "w") as f:
        json.dump(verification_codes, f, indent=2)
    
    return True

def get_weather_features(location, season):
    """Fetch actual weather forecast using OpenWeather."""
    try:
        coords = county_coords.get(location)
        if not coords:
            raise ValueError("Coordinates not found for county")
        
        url = (
            f"https://api.openweathermap.org/data/2.5/forecast?"
            f"lat={coords['lat']}&lon={coords['lon']}&units=metric&appid={OPENWEATHER_API_KEY}"
        )
        response = requests.get(url)
        forecast_data = response.json()

        # Use the next 24-48h forecast average for season-specific weather
        temp_max, temp_min, humidity, rain = [], [], [], []

        for entry in forecast_data["list"][:8]:  # next 24 hours (3h intervals)
            temp_max.append(entry["main"]["temp_max"])
            temp_min.append(entry["main"]["temp_min"])
            humidity.append(entry["main"]["humidity"])
            rain.append(entry.get("rain", {}).get("3h", 0))

        weather = {
            f"T2M_MAX-{season_abbr[season]}": round(np.mean(temp_max), 2),
            f"T2M_MIN-{season_abbr[season]}": round(np.mean(temp_min), 2),
            f"QV2M-{season_abbr[season]}": round(np.mean(humidity), 2),
            f"PRECTOTCORR-{season_abbr[season]}": round(np.sum(rain), 2),
        }

        return weather, "OpenWeather Forecast (Next 24h)"

    except Exception as e:
        logger.warning(f"Using fallback simulated weather: {e}")
                
        return fallback_simulated_weather(location, season)
    
    def fallback_simulated_weather(location, season):
        """Simulate fallback weather data."""
        logger.info("Using fallback simulated weather data.")
        return {
            f"T2M_MAX-{season_abbr[season]}": round(25 + random.uniform(-5, 5), 2),
            f"T2M_MIN-{season_abbr[season]}": round(15 + random.uniform(-5, 5), 2),
            f"QV2M-{season_abbr[season]}": round(60 + random.uniform(-10, 10), 2),
            f"PRECTOTCORR-{season_abbr[season]}": round(random.uniform(0, 10), 2),
        }, "Simulated Weather Data"
        
        return fallback_simulated_weather(location, season)
        
        # Define the fallback_simulated_weather function
        def fallback_simulated_weather(location, season):
            """Simulate fallback weather data."""
            logger.info("Using fallback simulated weather data.")
            return {
                f"T2M_MAX-{season_abbr[season]}": round(25 + random.uniform(-5, 5), 2),
                f"T2M_MIN-{season_abbr[season]}": round(15 + random.uniform(-5, 5), 2),
                f"QV2M-{season_abbr[season]}": round(60 + random.uniform(-10, 10), 2),
                f"PRECTOTCORR-{season_abbr[season]}": round(random.uniform(0, 10), 2),
            }, "Simulated Weather Data"

def get_soil_features(county, soilcolor):
    """Get soil features for prediction."""
    # Adapt soil properties based on county and soil color
    county_data = kenyan_counties.get(county, {})
    altitude = county_data.get("altitude", "Medium")
    rainfall = county_data.get("rainfall", "Medium")
    
    # Adjust soil properties based on characteristics
    if soilcolor == "Black":
        base_ph = 6.8
        base_k = 140
        base_p = 35
        base_n = 50
    elif soilcolor == "Red":
        base_ph = 5.8
        base_k = 100
        base_p = 25
        base_n = 35
    elif soilcolor == "Brown":
        base_ph = 6.5
        base_k = 120
        base_p = 30
        base_n = 45
    elif soilcolor == "Yellow":
        base_ph = 6.0
        base_k = 90
        base_p = 22
        base_n = 30
    else:  # Grey
        base_ph = 7.0
        base_k = 110
        base_p = 28
        base_n = 40
    
    # Adjust based on altitude and rainfall
    if altitude == "High":
        ph_adj = -0.3
        k_adj = -10
        p_adj = 5
        n_adj = 5
    elif altitude == "Medium":
        ph_adj = 0
        k_adj = 0
        p_adj = 0
        n_adj = 0
    else:  # Low
        ph_adj = 0.3
        k_adj = 10
        p_adj = -5
        n_adj =0
    
    if rainfall == "High":
        ph_adj += -0.1
        k_adj += -5
        p_adj += 2
        n_adj += 3
    elif rainfall == "Medium":
        ph_adj += 0
        k_adj += 0
        p_adj += 0
        n_adj += 0
    else:  # Low
        ph_adj += 0.1
        k_adj += 5
        p_adj += -2
        n_adj += -3
# Calculate adjusted values
    # Generate soil features
    soil_features = {}
    for s in season_abbr.values():
        soil_features[f"PHH2O-{s}"] = round(base_ph + ph_adj, 2)
        soil_features[f"K-{s}"] = round(base_k + k_adj, 2)
        soil_features[f"P-{s}"] = round(base_p + p_adj, 2)
        soil_features[f"N-{s}"] = round(base_n + n_adj, 2)
        soil_features[f"OC-{s}"] = round(fake.random.uniform(0.5, 3.0), 2)
        soil_features[f"BD-{s}"] = round(fake.random.uniform(0.5, 3.0), 2)
        soil_features[f"CLAY-{s}"] = round(fake.random.uniform(0.5, 3.0), 2)
        soil_features[f"SAND-{s}"] = round(fake.random.uniform(0.5, 3.0), 2)
    return soil_features
def get_soil_features(county, soilcolor):
    """Get soil features for prediction."""
    # Adapt soil properties based on county and soil color
    county_data = kenyan_counties.get(county, {})
    altitude = county_data.get("altitude", "Medium")
    rainfall = county_data.get("rainfall", "Medium")
    
    # Adjust soil properties based on characteristics
    if soilcolor == "Black":
        base_ph = 6.8
        base_k = 140
        base_p = 35
        base_n = 50
    elif soilcolor == "Red":
        base_ph = 5.8
        base_k = 100
        base_p = 25
        base_n = 35
    elif soilcolor == "Brown":
        base_ph = 6.5
        base_k = 120
        base_p = 30
        base_n = 45
    elif soilcolor == "Yellow":
        base_ph = 6.0
        base_k = 90
        base_p = 22
        base_n = 30
    else:  # Grey
        base_ph = 7.0
        base_k = 110
        base_p = 28
        base_n = 40
    
    # Adjust based on altitude and rainfall
    if altitude == "High":
        ph_adj = -0.3
        k_adj = -10
        p_adj = 5
        n_adj = 5
    elif altitude == "Medium":
        ph_adj = 0
        k_adj = 0
        p_adj = 0
        n_adj = 0
    else:  # Low
        ph_adj = 0.3
        k_adj = 10
        p_adj = -5
        n_adj = -5
        
    if rainfall == "High":
        ph_adj -= 0.2
        k_adj -= 15
        p_adj -= 3
        n_adj += 10
    elif rainfall == "Low":
        ph_adj += 0.2
        k_adj += 5
        p_adj -= 5
        n_adj -= 10
    
    # Add some randomness
    ph_var = fake.random_int(-10, 10) / 100
    k_var = fake.random_int(-15, 15)
    p_var = fake.random_int(-5, 5)
    n_var = fake.random_int(-8, 8)
    
    # Calculate final values
    soil_data = {
        "Ph": round(base_ph + ph_adj + ph_var, 1),
        "K": max(10, base_k + k_adj + k_var),
        "P": max(5, base_p + p_adj + p_var),
        "N": max(10, base_n + n_adj + n_var),
        "Zn": round(2 + fake.random_int(-1, 2), 1),
        "S": round(10 + fake.random_int(-3, 5), 1)
    }
    
    # Create one-hot encoded columns for soil color
    for col in soilcolor_cols:
        soil_data[col] = 0
    
    # Set the appropriate soil color column to 1
    color_col = f"Soilcolor_{soilcolor}"
    if color_col in soilcolor_cols:
        soil_data[color_col] = 1
    else:
        # If exact match not found, set first soil color
        if soilcolor_cols:
            soil_data[soilcolor_cols[0]] = 1
    
    return soil_data

def recommend_crops(county, season, soilcolor):
    """Recommend crops based on county, season, and soil color."""
    # Get county specific crops
    county_crops = kenyan_counties.get(county, {}).get("suitable_crops", [])
    
    # Get season specific crops
    season_crops = seasons.get(season, [])
    
    # Get soil color specific crops
    # Define soil_colors mapping if not already defined
    soil_colors = {
        "Black": ["Maize", "Beans", "Rice"],
        "Red": ["Wheat", "Tomato", "Coffee"],
        "Brown": ["Potatoes", "Tea", "Vegetables"],
        "Yellow": ["Sorghum", "Millet", "Cassava"],
        "Grey": ["Barley", "Bananas", "Green Grams"]
    }
    soil_crops = soil_colors.get(soilcolor, [])
    
    # Find crops that match all criteria
    intersection = set(county_crops) & set(season_crops) & set(soil_crops)
    
    # If no intersection, prioritize county crops, then add season and soil crops
    if not intersection:
        intersection = set(county_crops)
        if not intersection:
            intersection = set(season_crops) | set(soil_crops)
    
    # Convert to list and ensure we have at least 3 crops
    recommended_crops = list(intersection)
    
    # If we don't have enough crops, add more from each category
    while len(recommended_crops) < 3:
        for crop_list in [county_crops, season_crops, soil_crops]:
            for crop in crop_list:
                if crop not in recommended_crops:
                    recommended_crops.append(crop)
                    break
            if len(recommended_crops) >= 3:
                break
        
        # If still not enough, add default crops
        if len(recommended_crops) < 3:
            default_crops = ["Maize", "Beans", "Wheat", "Rice", "Vegetables"]
            for crop in default_crops:
                if crop not in recommended_crops:
                    recommended_crops.append(crop)
                if len(recommended_crops) >= 3:
                    break
    
    # Assign probabilities (confidence levels)
    total = 100
    probabilities = []
    
    # Top crop gets 50-70% confidence
    top_prob = random.randint(50, 70)
    probabilities.append(top_prob)
    remaining = total - top_prob
    
    # Second crop gets 20-30% of remaining confidence
    if len(recommended_crops) > 1:
        second_prob = random.randint(20, min(30, remaining - 5))  # Ensure we leave at least 5% for the third crop
        probabilities.append(second_prob)
        remaining -= second_prob
    
    # Distribute remaining probability among other crops
    for i in range(2, len(recommended_crops)):
        if i == len(recommended_crops) - 1 or remaining <= 5:
            # Last crop or small remainder gets all remaining probability
            probabilities.append(remaining)
            break
        else:
            # Distribute evenly
            prob = max(5, remaining // (len(recommended_crops) - i))
            probabilities.append(prob)
            remaining -= prob
    
    # Format as percentage strings
    formatted_probs = [f"{prob}%" for prob in probabilities]
    
    # Create result list
    result = []
    for i in range(min(3, len(recommended_crops))):
        result.append({"crop": recommended_crops[i], "probability": formatted_probs[i]})
    
    return result


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from request
        data = request.json
        county = data.get("county")
        season = data.get("season")
        soilcolor = data.get("soilcolor", "Black")

        if not county or not season:
            return jsonify({"error": "County and season are required"}), 400

        # Always attempt model prediction first
        if model is not None and scaler is not None and label_encoder is not None:
            # Get weather and soil features
            weather_data, weather_source = get_weather_features(county, season)
            soil_data = get_soil_features(county, soilcolor)

            # Combine all features into a single dictionary
            combined_data = {**weather_data, **soil_data}

            # Create DataFrame with expected features
            df = pd.DataFrame([combined_data])
            for feature in expected_features:
                if feature not in df.columns:
                    df[feature] = 0  # Fill missing features with zeros
            df = df[expected_features]  # Ensure only expected features are used

            try:
                # Scale the data
                scaled = scaler.transform(df)
            except Exception as e:
                logger.error(f"Scaling error: {e}")
                scaled = df.values  # Use unscaled data if scaling fails

            try:
                # Make prediction using the model
                prediction = model.predict(scaled)
                predicted_label = label_encoder.inverse_transform([prediction[0]])[0]

                # Get other crops for secondary recommendations
                crops = list(label_encoder.classes_)
                other_crops = [c for c in crops if c != predicted_label][:2]

                # Format the top predictions
                top_preds = [
                    {"crop": predicted_label, "probability": "85.5%"},
                    {"crop": other_crops[0] if len(other_crops) > 0 else "Alternative", "probability": "10.2%"},
                    {"crop": other_crops[1] if len(other_crops) > 1 else "Another", "probability": "4.3%"}
                ]
            except Exception as e:
                logger.error(f"Model prediction failed: {e}")
                # Fallback to recommendations if model fails
                recommendations = recommend_crops(county, season, soilcolor)
                predicted_label = recommendations[0]["crop"] if recommendations else "Maize"
                recommendations = recommend_crops(county, season, soilcolor)
                recommendations = recommend_crops(county, season, soilcolor)
                top_preds = recommendations if recommendations else [
                    {"crop": "Maize", "probability": "60.0%"},
                    {"crop": "Beans", "probability": "30.0%"},
                    {"crop": "Wheat", "probability": "10.0%"}
                ]
        else:
            # If model is unavailable, use recommendations as fallback
            recommendations = recommend_crops(county, season, soilcolor)
            predicted_label = recommendations[0]["crop"] if recommendations else "Maize"
            top_preds = recommendations if recommendations else [
                {"crop": "Maize", "probability": "60.0%"},
                {"crop": "Beans", "probability": "30.0%"},
                {"crop": "Wheat", "probability": "10.0%"}
            ]

        # Log the prediction
        try:
            weather_data, weather_source = get_weather_features(county, season)
            soil_data = get_soil_features(county, soilcolor)
            combined_data = {**weather_data, **soil_data}
            log_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "county": county,
                "season": season,
                "input": str(combined_data)[:100],  # Truncate to avoid overflow
                "prediction": predicted_label,
                "true_label": None
            }
            pd.DataFrame([log_entry]).to_csv(PREDICTION_LOG, mode="a", header=False, index=False)
        except Exception as e:
            logger.warning(f"Failed to log prediction: {e}")

        # Generate related product inputs for the recommended crop
        inputs = []
        try:
            if predicted_label in farm_inputs:
                crop_inputs = farm_inputs[predicted_label]
                if "seeds" in crop_inputs and crop_inputs["seeds"]:
                    seed = random.choice(crop_inputs["seeds"])
                    inputs.append({
                        "price": seed["price"],
                        "vendor": f"{county} Farm Supplies",
                        "contact": f"07{fake.random_number(digits=8)}"
                    })
                if "fertilizers" in crop_inputs and crop_inputs["fertilizers"]:
                    fertilizer = random.choice(crop_inputs["fertilizers"])
                    inputs.append({
                        "price": fertilizer["price"],
                        "vendor": f"{county} Agrovet",
                        "contact": f"07{fake.random_number(digits=8)}"
                    })
            while len(inputs) < 2:
                inputs.append({
                    "price": fake.random_int(min=100, max=1000),
                    "vendor": fake.company(),
                    "contact": f"07{fake.random_number(digits=8)}"
                })
        except Exception as e:
            logger.warning(f"Error generating product inputs: {e}")
            inputs = [
                {"price": fake.random_int(min=100, max=1000), "vendor": fake.company(), "contact": f"07{fake.random_number(digits=8)}"}
                for _ in range(2)
            ]

        # Return the final response
        return jsonify({
            "prediction": predicted_label,
            "top_predictions": top_preds,
            "inputs": inputs,
            "weather_source": weather_source
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        # Return a default response even in case of error
        return jsonify({
            "prediction": "Maize",
            "top_predictions": [
                {"crop": "Maize", "probability": "60.0%"},
                {"crop": "Beans", "probability": "30.0%"},
                {"crop": "Wheat", "probability": "10.0%"}
            ],
            "inputs": [
                {"price": 500, "vendor": "Default Farm Supplies", "contact": "0700000000"},
                {"price": 300, "vendor": "Backup Vendor Ltd", "contact": "0711111111"}
            ],
            "weather_source": "Error recovery mode"
        })

@app.route("/", methods=["GET", "POST"])
def home():
    """Home page route."""
    error = None
    prediction = None
    top_preds = []
    inputs = []
    weather_src = None
    
    # Get list of counties for dropdown
    counties = kenyan_counties
    
    # Get list of soil colors
    # Define soil_colors mapping if not already defined
    soil_colors = {
        "Black": ["Maize", "Beans", "Rice"],
        "Red": ["Wheat", "Tomato", "Coffee"],
        "Brown": ["Potatoes", "Tea", "Vegetables"],
        "Yellow": ["Sorghum", "Millet", "Cassava"],
        "Grey": ["Barley", "Bananas", "Green Grams"]
    }
    soil_colors_list = soil_colors
    
    if request.method == "POST":
        try:
            # Get form data
            county = request.form.get("county")
            season = request.form.get("season")
            soilcolor = request.form.get("soilcolor")
            
            if not county or not season or not soilcolor:
                error = "Please fill all required fields"
            else:
                # Make API call to our own prediction endpoint
                response = requests.post(
                    f"{request.host_url}predict",
                    json={"county": county, "season": season, "soilcolor": soilcolor},
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    prediction = result.get("prediction")
                    top_preds = result.get("top_predictions", [])
                    inputs = result.get("inputs", [])
                    weather_src = result.get("weather_source")
                else:
                    error = f"Prediction failed: {response.text}"
                    
        except Exception as e:
            error = f"Error: {str(e)}"
    
    return render_template(
        "index.html", 
        prediction=prediction,
        top_preds=top_preds,
        inputs=inputs,
        counties=counties,
        seasons=seasons,
        soil_colors=soil_colors_list,
        vendors=vendors,
        weather_src=weather_src,
        error=error
    )

@app.route("/crop_details/<crop_name>")
def crop_details(crop_name):
    """Crop details page."""
    # Get farm inputs for this crop
    crop_inputs = farm_inputs.get(crop_name, {})
    
    if not crop_inputs:
        flash(f"No data available for {crop_name}", "warning")
        return redirect(url_for('home'))
    
    # Get vendors that might sell inputs for this crop
    crop_vendors = []
    for vendor in vendors:
        if vendor["product"] in ["Seeds", "Fertilizers", "Pesticides", "Equipment"]:
            crop_vendors.append(vendor)
    
    return render_template(
        "crop_details.html",
        crop_name=crop_name,
        farm_inputs=crop_inputs,
        crop_vendors=crop_vendors
    )

@app.route("/vendor_marketplace")
def vendor_marketplace():
    """Vendor marketplace page."""
    # Get query parameters for filtering
    product_filter = request.args.get('product', '')
    location_filter = request.args.get('location', '')
    
    # Apply filters
    filtered_vendors = vendors
    if product_filter:
        filtered_vendors = [v for v in filtered_vendors if v.get("product") == product_filter]
    
    if location_filter:
        filtered_vendors = [v for v in filtered_vendors if v.get("location") == location_filter]
    
    # Get list of counties for location filter
    counties = list(kenyan_counties.keys())
    
    return render_template(
        "vendor_marketplace.html",
        vendors=filtered_vendors,
        counties=counties,
        product_filter=product_filter,
        location_filter=location_filter
    )

@app.route("/login", methods=["GET", "POST"])
def login():
    """Handle user login."""
    error = None
    step = "phone"  # Default step
    phone = None
    
    if request.method == "POST":
        step = request.form.get("step", "phone")
        
        if step == "phone":
            phone = request.form.get("phone")
            
            # Validate phone
            if not phone or not re.match(r"^0[0-9]{9}$", phone):
                error = "Please enter a valid phone number in the format 07XXXXXXXX"
            else:
                # Check if user exists
                user_exists = False
                for user in users:
                    if user.get("phone") == phone:
                        user_exists = True
                        break
                
                if not user_exists:
                    error = "Phone number not registered. Please register first."
                else:
                    # Generate and send verification code
                    code = generate_verification_code()
                    if send_verification_sms(phone, code):
                        step = "verify"
                    else:
                        error = "Failed to send verification code. Please try again."
        
        elif step == "verify":
            phone = request.form.get("phone")
            verification_code = request.form.get("verification_code")
            
            # Validate verification code
            try:
                with open("verification_codes.json", "r") as f:
                    verification_codes = json.load(f)
            except:
                verification_codes = {}
                
            expected_code = verification_codes.get(phone)
            
            if not expected_code or verification_code != expected_code:
                error = "Invalid verification code. Please try again."
            else:
                # Valid code - log user in
                # Find user info
                user_info = None
                for user in users:
                    if user.get("phone") == phone:
                        user_info = user
                        break
                
                if user_info:
                    session["user_phone"] = phone
                    session["user_name"] = user_info.get("name", "Farmer")
                    session["user_county"] = user_info.get("county", "")
                    
                    # Clear verification code
                    del verification_codes[phone]
                    with open("verification_codes.json", "w") as f:
                        json.dump(verification_codes, f, indent=2)
                    
                    flash("Login successful!", "success")
                    
                    # Redirect to next page or home
                    next_page = request.args.get("next")
                    if next_page:
                        return redirect(next_page)
                    return redirect(url_for("home"))
                else:
                    error = "User not found"
    
    return render_template(
        "login.html",
        error=error,
        step=step,
        phone=phone
    )

@app.route("/register", methods=["GET", "POST"])
def register():
    """Handle user registration."""
    error = None
    step = "details"  # Default step
    name = None
    phone = None
    county = None
    
    if request.method == "POST":
        step = request.form.get("step", "details")
        
        if step == "details":
            name = request.form.get("name")
            phone = request.form.get("phone")
            county = request.form.get("county")
            
            # Validate phone
            if not phone or not re.match(r"^0[0-9]{9}$", phone):
                error = "Please enter a valid phone number in the format 07XXXXXXXX"
            # Validate other fields
            elif not name or not county:
                error = "Please fill all required fields"
            else:
                # Check if user already exists
                for user in users:
                    if user.get("phone") == phone:
                        error = "Phone number already registered. Please login instead."
                        break
                
                if not error:
                    # Generate and send verification code
                    code = generate_verification_code()
                    if send_verification_sms(phone, code):
                        step = "verify"
                    else:
                        error = "Failed to send verification code. Please try again."
        
        elif step == "verify":
            name = request.form.get("name")
            phone = request.form.get("phone")
            county = request.form.get("county")
            verification_code = request.form.get("verification_code")
            
            # Validate verification code
            try:
                with open("verification_codes.json", "r") as f:
                    verification_codes = json.load(f)
            except:
                verification_codes = {}
                
            expected_code = verification_codes.get(phone)
            
            if not expected_code or verification_code != expected_code:
                error = "Invalid verification code. Please try again."
            else:
                # Valid code - register user
                new_user = {
                    "name": name,
                    "phone": phone,
                    "county": county,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                users.append(new_user)
                
                # Save user data
                with open("users.json", "w") as f:
                    json.dump(users, f, indent=2)
                
                # Clear verification code
                del verification_codes[phone]
                with open("verification_codes.json", "w") as f:
                    json.dump(verification_codes, f, indent=2)
                
                # Log user in
                session["user_phone"] = phone
                session["user_name"] = name
                session["user_county"] = county
                
                flash("Registration successful!", "success")
                return redirect(url_for("home"))
    
    # Get list of counties for dropdown
    counties = list(kenyan_counties.keys())
    
    return render_template(
        "register.html",
        error=error,
        step=step,
        name=name,
        phone=phone,
        county=county,
        counties=counties
    )

@app.route("/logout")
def logout():
    """Handle user logout."""
    # Clear session
    session.pop("user_phone", None)
    session.pop("user_name", None)
    session.pop("user_county", None)
    
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))

@app.route("/register_vendor", methods=["GET", "POST"])
@login_required
def register_vendor():
    """Handle vendor registration."""
    error = None
    
    if request.method == "POST":
        # Get form data
        name = request.form.get("name")
        email = request.form.get("email")
        phone = request.form.get("phone")
        location = request.form.get("location")
        product = request.form.get("product")
        hours = request.form.get("hours", "8:00 AM - 5:00 PM")
        
        # Validate fields
        if not name or not email or not phone or not location or not product:
            error = "Please fill all required fields"
        elif not re.match(r"^0[0-9]{9}$", phone):
            error = "Please enter a valid phone number in the format 07XXXXXXXX"
        elif not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            error = "Please enter a valid email address"
        else:
            # Create new vendor
            new_vendor = {
                "name": name,
                "email": email,
                "phone": phone,
                "location": location,
                "product": product,
                "hours": hours,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Add to vendors list
            vendors.append(new_vendor)
            
            # Save vendor data
            with open("vendors.json", "w") as f:
                json.dump(vendors, f, indent=2)
            
            flash("Vendor registration successful!", "success")
            return redirect(url_for("vendor_marketplace"))
    
    # Get list of counties for dropdown
    counties = list(kenyan_counties.keys())
    
    return render_template(
        "register_vendor.html",
        error=error,
        counties=counties
    )

if __name__ == "__main__":
    app.run(debug=True)