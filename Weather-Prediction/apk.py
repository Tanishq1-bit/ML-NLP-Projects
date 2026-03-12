import streamlit as st
import numpy as np
import joblib
from datetime import date

# Page Configuration
st.set_page_config(
    page_title="AI Weather Predictor",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Model
@st.cache_resource
def load_model():
    return joblib.load("weather_model.pkl")

model = load_model()

# Custom CSS (Glassmorphism + Animation)
st.markdown("""
<style>
    /* Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
    }

    /* Title Styling */
    .title {
        font-size: 55px;
        font-weight: bold;
        text-align: center;
        background: -webkit-linear-gradient(#00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Glass Card */
    .glass {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        height: 60px;
        border-radius: 15px;
        font-size: 20px;
        font-weight: bold;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white;
        border: none;
        transition: 0.3s;
    }

    .stButton>button:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #0072ff, #00c6ff);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0e1117 !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">🌦️ AI Weather Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown("### 🔍 Predict Weather Conditions using Machine Learning")

# Sidebar Inputs (Professional Layout)
st.sidebar.header("⚙️ Input Weather Parameters")

precipitation = st.sidebar.slider("🌧️ Precipitation", 0.0, 60.0, 2.0)
temp_max = st.sidebar.slider("🌡️ Max Temperature (°C)", -10.0, 40.0, 25.0)
temp_min = st.sidebar.slider("❄️ Min Temperature (°C)", -15.0, 30.0, 15.0)
wind = st.sidebar.slider("💨 Wind Speed", 0.0, 20.0, 3.0)

selected_date = st.sidebar.date_input("📅 Select Date", date.today())
year = selected_date.year
month = selected_date.month
day = selected_date.day

# Main Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("📊 Live Input Summary")
    st.write(f"**Precipitation:** {precipitation}")
    st.write(f"**Max Temp:** {temp_max} °C")
    st.write(f"**Min Temp:** {temp_min} °C")
    st.write(f"**Wind Speed:** {wind}")
    st.write(f"**Date:** {selected_date}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("🤖 Model Prediction")

    if st.button("🚀 Predict Weather Now"):
        input_data = np.array([[precipitation, temp_max, temp_min, wind, year, month, day]])

        prediction = model.predict(input_data)[0]

        # Try to get confidence (if model supports predict_proba)
        try:
            confidence = np.max(model.predict_proba(input_data)) * 100
        except:
            confidence = None

        # Weather Icons Logic
        weather_icons = {
            "rain": "☔",
            "sun": "🌞",
            "snow": "❄️",
            "drizzle": "🌦️",
            "fog": "🌫️",
            "cloudy": "☁️"
        }

        icon = weather_icons.get(prediction.lower(), "🌤️")

        st.markdown(f"## {icon} Predicted Weather: **{prediction.upper()}**")

        if confidence:
            st.progress(int(confidence))
            st.write(f"🎯 Confidence Score: **{confidence:.2f}%**")

        # Smart Suggestions (Real-world touch)
        st.markdown("### 🧠 AI Recommendation")
        if prediction.lower() == "rain":
            st.info("Carry an umbrella and wear waterproof clothing.")
        elif prediction.lower() == "sun":
            st.warning("Use sunscreen and stay hydrated.")
        elif prediction.lower() == "snow":
            st.error("Wear warm clothes. Cold weather expected.")
        elif prediction.lower() == "fog":
            st.warning("Drive carefully due to low visibility.")
        else:
            st.success("Weather looks stable. Have a great day!")

    st.markdown('</div>', unsafe_allow_html=True)

# Footer (Professional)
st.markdown("---")
st.markdown(
    "<center>💼 Built by Tanishq Jadhav | Machine Learning + Streamlit Project | 2026</center>",
    unsafe_allow_html=True
)
