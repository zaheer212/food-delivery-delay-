import streamlit as st
import pickle
import numpy as np
import requests

# ================= API KEYS =================
TOMTOM_API_KEY = "8HcuqBpg8YhwXacWnqWoA3vMJccr2P2x"
WEATHER_API_KEY = "8161c6aad9c5c3bfc6f2c1da6ad77b4c"

# ================= LOAD MODEL =================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# SINGLE LabelEncoder (Vehicle only)
with open("label_encoders.pkl", "rb") as f:
    vehicle_encoder = pickle.load(f)

# ================= WEATHER =================
def get_weather(lat, lon):
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}"
    )
    try:
        res = requests.get(url, timeout=5).json()
        return res["weather"][0]["main"]
    except:
        return "Clear"

# ================= TRAFFIC =================
def get_traffic(lat, lon):
    url = (
        f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
        f"?point={lat},{lon}&key={TOMTOM_API_KEY}"
    )
    try:
        data = requests.get(url, timeout=5).json()
        cs = data["flowSegmentData"]["currentSpeed"]
        fs = data["flowSegmentData"]["freeFlowSpeed"]
        ratio = cs / fs

        if ratio > 0.8:
            return 0   # Low
        elif ratio > 0.5:
            return 1   # Medium
        else:
            return 2   # High
    except:
        return 1

# ================= STREAMLIT UI =================
st.set_page_config(page_title="Food Delivery Delay Predictor")
st.title("🚴 Food Delivery Delay Predictor")

distance = st.number_input("Distance (km)", min_value=0.0)
prep_time = st.number_input("Preparation Time (minutes)", min_value=0)
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

    vehicle_encoded = vehicle_encoder.transform([vehicle])[0]

    # Feature order MUST match training
    X = np.array([[
        distance,
        prep_time,
        traffic,
        vehicle_encoded
    ]])

    prediction = model.predict(X)

    st.success(f"🕒 Estimated Delivery Time: {prediction[0]:.2f} minutes")
    st.caption(f"🌦 Weather: {weather}")
