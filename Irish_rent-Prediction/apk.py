import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load artifacts
# -----------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("🏠 Irish Rent Prediction")

# -----------------------------
# User inputs (RAW)
# -----------------------------
area = st.number_input("Area (sq meters)", 20, 500, 60)
bedrooms = st.number_input("Bedrooms", 1, 10, 2)

period = st.selectbox(
    "Period",
    ["20202", "20211", "20212", "20221", "20222"]
)

county = st.selectbox(
    "County",
    ["Dublin", "Cork", "Galway", "Limerick"]
)

location = st.selectbox(
    "Location",
    ["City Centre", "Suburbs", "Rural"]
)

# -----------------------------
# Create raw DataFrame
# -----------------------------
raw_df = pd.DataFrame([{
    "area": area,
    "bedrooms": bedrooms,
    "Period": period,
    "county": county,
    "location": location
}])

# -----------------------------
# One-hot encode
# -----------------------------
encoded_df = pd.get_dummies(raw_df)

# -----------------------------
# FORCE training feature space
# -----------------------------
encoded_df = encoded_df.reindex(columns=columns, fill_value=0)

# 🚨 DEBUG CHECK (optional but recommended)
# st.write(encoded_df.columns)

# -----------------------------
# Scaling (NOW SAFE)
# -----------------------------
input_scaled = scaler.transform(encoded_df)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Rent 💰"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"Estimated Monthly Rent: € {int(prediction):,}")
