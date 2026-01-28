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

with open("label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# ================= WEATHER =================
def get_weather(lat, lon):
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}"
    )

    try:
        response = requests.get(url, timeout=5)
        data = response.json()

        if response.status_code != 200:
            return "Unknown"

        if "weather" not in data:
            return "Unknown"

        return data["weather"][0]["main"]

    except Exception:
        return "Unknown"

# ================= TRAFFIC =================
def get_traffic(lat, lon):
    url = (
        f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
        f"?point={lat},{lon}&key={TOMTOM_API_KEY}"
    )

    try:
        response = requests.get(url, timeout=5)
        data = response.json()

        if "flowSegmentData" not in data:
            return "Unknown"

        current_speed = data["flowSegmentData"]["currentSpeed"]
        free_speed = data["flowSegmentData"]["freeFlowSpeed"]

        ratio = current_speed / free_speed

        if ratio > 0.8:
            return "Low"
        elif ratio > 0.5:
            return "Medium"
        else:
            return "High"

    except Exception:
        return "Unknown"

# ================= STREAMLIT UI =================
st.set_page_config(page_title="Food Delivery Delay Predictor")
st.title("🚚 Food Delivery Delay Predictor")

distance = st.number_input("Distance (km)", min_value=0.0)
prep_time = st.number_input("Preparation Time (minutes)", min_value=0)
lat = st.number_input("Latitude", value=18.5314)
lon = st.number_input("Longitude", value=73.8446)

vehicle = st.selectbox(
    "Vehicle Type", encoders["vehicle"].classes_
)

# ================= PREDICTION =================
if st.button("Predict Delivery Time"):

    weather = get_weather(lat, lon)
    traffic = get_traffic(lat, lon)

    # -------- SAFE FALLBACKS --------
    if weather not in encoders["weather"].classes_:
        weather = encoders["weather"].classes_[0]

    if traffic not in encoders["traffic"].classes_:
        traffic = encoders["traffic"].classes_[0]

    # -------- ENCODING (CORRECT WAY) --------
    weather_enc = encoders["weather"].transform([weather])[0]
    traffic_enc = encoders["traffic"].transform([traffic])[0]
    vehicle_enc = encoders["vehicle"].transform([vehicle])[0]

    # -------- FEATURE ORDER MUST MATCH TRAINING --------
    X = np.array([[
        distance,
        prep_time,
        weather_enc,
        traffic_enc,
        vehicle_enc
    ]])

    prediction = model.predict(X)

    st.success(f"🕒 Estimated Delivery Time: {prediction[0]:.2f} minutes")
    st.caption(f"🌦 Weather: {weather} | 🚦 Traffic: {traffic}")

