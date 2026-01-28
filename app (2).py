import streamlit as st
import pickle
import numpy as np
import requests
from datetime import datetime

# ================= CONFIG =================
st.set_page_config(page_title="Food Delivery Delay Predictor", layout="centered")

TOMTOM_API_KEY = "8HcuqBpg8YhwXacWnqWoA3vMJccr2P2x"
WEATHER_API_KEY = "8161c6aad9c5c3bfc6f2c1da6ad77b4c"

# ================= LOAD MODEL =================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# SINGLE LabelEncoder (Vehicle_Type)
with open("label_encoders.pkl", "rb") as f:
    vehicle_encoder = pickle.load(f)

# ================= HELPERS =================
def get_weather(lat, lon):
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}"
        )
        data = requests.get(url, timeout=5).json()
        return data["weather"][0]["main"]
    except:
        return "Clear"

def get_traffic(lat, lon):
    try:
        url = (
            f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
            f"?point={lat},{lon}&key={TOMTOM_API_KEY}"
        )
        data = requests.get(url, timeout=5).json()
        cs = data["flowSegmentData"]["currentSpeed"]
        fs = data["flowSegmentData"]["freeFlowSpeed"]
        ratio = cs / fs

        if ratio > 0.8:
            return 0  # Low
        elif ratio > 0.5:
            return 1  # Medium
        else:
            return 2  # High
    except:
        return 1  # Medium default

def encode_weather(weather):
    weather_map = {
        "Clear": 0,
        "Clouds": 1,
        "Rain": 2,
        "Mist": 3,
        "Haze": 3
    }
    return weather_map.get(weather, 0)

def encode_time_of_day():
    hour = datetime.now().hour
    if hour < 12:
        return 0  # Morning
    elif hour < 17:
        return 1  # Afternoon
    else:
        return 2  # Evening

# ================= UI =================
st.title("🚴 Food Delivery Delay Prediction")

distance = st.number_input("Distance (km)", min_value=0.1)
prep_time = st.number_input("Preparation Time (minutes)", min_value=1)
courier_exp = st.number_input("Courier Experience (years)", min_value=0)

lat = st.number_input("Latitude", value=18.5314)
lon = st.number_input("Longitude", value=73.8446)

vehicle = st.selectbox(
    "Vehicle Type",
    vehicle_encoder.classes_
)

# ================= PREDICTION =================
if st.button("Predict Delivery Time"):

    weather = get_weather(lat, lon)
    traffic = get_traffic(lat, lon)

    weather_encoded = encode_weather(weather)
    time_encoded = encode_time_of_day()
    vehicle_encoded = vehicle_encoder.transform([vehicle])[0]

    # FINAL FEATURE VECTOR (7 FEATURES)
    X = np.array([[
        distance,            # Distance_km
        traffic,             # Traffic_Level
        weather_encoded,     # Weather
        time_encoded,        # Time_of_Day
        vehicle_encoded,     # Vehicle_Type
        prep_time,           # Preparation_Time_min
        courier_exp          # Courier_Experience_yrs
    ]])

    prediction = model.predict(X)

    st.success(f"🕒 Estimated Delivery Time: {prediction[0]:.2f} minutes")
    st.caption(f"🌦 Weather: {weather} | 🚦 Traffic Level: {traffic}")
