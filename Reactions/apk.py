import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords

# --- PAGE CONFIG ---
st.set_page_config(page_title="Reaction AI", page_icon="✨", layout="centered")

# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    try:
        model = pickle.load(open("Reaction_model.pkl", "rb"))
        vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
        # Ensure stopwords are available
        nltk.download('stopwords', quiet=True)
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'Reaction_model.pkl' and 'vectorizer.pkl' are in the same directory.")
        return None, None

model, vectorizer = load_assets()

# --- PREPROCESSING FUNCTIONS (From your Notebook) ---
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove Punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove Numbers
    text = "".join([i for i in text if not i.isdigit()])
    # Remove Stopwords
    words = text.split()
    cleaned = [i for i in words if i not in stop_words]
    return " ".join(cleaned)

# --- UI DESIGN ---
st.title("✨ Reaction AI")
st.markdown("#### Instant Social Media Sentiment Analysis")
st.info("This model classifies text into **Positive**, **Neutral**, or **Negative** sentiments based on your trained SVM/Logistic model.")

# Custom CSS for a "10/10" look
st.markdown("""
    <style>
    .stTextArea textarea {
        font-size: 1.2rem !important;
    }
    .sentiment-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 24px;
    }
    </style>
    """, unsafe_allow_html=True)

# Input Area
user_input = st.text_area("What's on your mind?", placeholder="e.g., highkey this is lit ✨", height=150)

if st.button("Analyze Sentiment", use_container_width=True):
    if user_input.strip() != "" and model is not None:
        # 1. Preprocess
        cleaned = clean_text(user_input)
        
        # 2. Vectorize
        vectorized_input = vectorizer.transform([cleaned])
        
        # 3. Predict
        prediction = model.predict(vectorized_input)[0]
        
        # Mapping (Adjust the indices 0, 1, 2 based on your unique_label order)
        # Based on your notebook: 0=Neutral, 1=Positive, 2=Negative (typically)
        # Check your 'label_numbers' dict in the notebook to confirm!
        labels = {0: "Neutral 😐", 1: "Positive 😁", 2: "Negative 💔"}
        result = labels.get(prediction, "Unknown")

        st.divider()
        
        # Visual Result
        if "Positive" in result:
            st.balloons()
            st.success(f"Result: {result}")
        elif "Negative" in result:
            st.error(f"Result: {result}")
        else:
            st.warning(f"Result: {result}")
            
        st.caption(f"**Cleaned text used for prediction:** {cleaned}")
    else:
        st.warning("Please enter some text first!")

# --- FOOTER ---
st.markdown("---")
st.markdown("<p style='text-align: center;'>Built with ❤️ using Streamlit & Scikit-Learn</p>", unsafe_allow_html=True)