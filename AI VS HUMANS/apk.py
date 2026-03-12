import streamlit as st
import pickle
import numpy as np

# Load model and vectorizer
model = pickle.load(open("ai_human_model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

# Page config
st.set_page_config(
    page_title="AI vs Human Text Detector",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.title {
    font-size:40px;
    font-weight:bold;
    text-align:center;
    color:#4CAF50;
}
.subtitle{
    text-align:center;
    font-size:18px;
    color:gray;
}
.result{
    font-size:28px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="title">🤖 AI vs Human Text Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Detect whether a text is written by AI or a Human using Machine Learning</p>', unsafe_allow_html=True)

st.divider()

# Layout columns
col1, col2 = st.columns([2,1])

with col1:
    user_input = st.text_area("✍ Enter Text Here", height=200)

    predict = st.button("🔍 Analyze Text")

with col2:
    st.info(
    """
    ### 📌 About Project
    This NLP model classifies whether text is **AI generated or Human written**.

    **Tech Used**
    - Python  
    - Scikit-learn  
    - TF-IDF  
    - Naive Bayes / Logistic Regression  
    - Streamlit
    """
    )

# Prediction
if predict:

    if user_input.strip() == "":
        st.warning("⚠ Please enter some text.")
    else:
        vector = vectorizer.transform([user_input])
        prediction = model.predict(vector)[0]

        try:
            prob = model.predict_proba(vector).max()
        except:
            prob = 0.90

        st.divider()

        if prediction == "AI":
            st.error(f"🤖 AI Generated Text")
        else:
            st.success(f"👨 Human Written Text")

        st.metric("Confidence Score", f"{round(prob*100,2)} %")

st.divider()

# Footer
st.markdown(
"""
---
### 🚀 Project by Tanishq Jadhav  
Machine Learning | NLP | AI Projects
"""
)