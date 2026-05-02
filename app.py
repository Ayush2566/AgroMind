import streamlit as st
import pandas as pd
import joblib
import requests
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np

API_KEY = "88b5bdf1060ca72e1d3a7e200d2b1ded"

def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    response = requests.get(url)
    data = response.json()

    if data.get("cod") != 200:
        return None

    return {
        "temp": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "condition": data["weather"][0]["description"],
        "wind": data["wind"]["speed"]
    }

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AgroMind",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=Instrument+Serif:ital@0;1&display=swap');

/* ====== RESET & BASE ====== */
*, *::before, *::after { box-sizing: border-box; }

html, .stApp {
    background: #0a1a0f;
    color: #d4f5d4;
    font-family: 'Sora', sans-serif;
}

/* ====== BACKGROUND TEXTURE ====== */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 10% 0%, rgba(34,120,60,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 90% 100%, rgba(100,200,80,0.10) 0%, transparent 55%),
        repeating-linear-gradient(
            0deg,
            transparent,
            transparent 59px,
            rgba(60,120,60,0.035) 60px
        ),
        repeating-linear-gradient(
            90deg,
            transparent,
            transparent 59px,
            rgba(60,120,60,0.035) 60px
        );
    pointer-events: none;
    z-index: 0;
}

/* ====== SIDEBAR ====== */
section[data-testid="stSidebar"] {
    background: #0c1f10 !important;
    border-right: 1px solid rgba(100,200,80,0.15);
}

section[data-testid="stSidebar"] > div:first-child {
    padding-top: 2rem;
}

.sidebar-brand {
    text-align: center;
    padding: 1rem 1rem 2rem;
    border-bottom: 1px solid rgba(168,255,62,0.15);
    margin-bottom: 1.5rem;
}

.sidebar-brand-icon {
    font-size: 3rem;
    display: block;
    margin-bottom: 0.5rem;
    filter: drop-shadow(0 0 12px rgba(168,255,62,0.5));
}

.sidebar-brand-name {
    font-size: 1.6rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.5px;
    line-height: 1;
}

.sidebar-brand-tagline {
    font-size: 0.72rem;
    color: #6ab87a;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 0.3rem;
}

/* Sidebar radio */
div[data-testid="stRadio"] label {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.65rem 1rem;
    border-radius: 10px;
    margin-bottom: 0.25rem;
    font-size: 0.9rem;
    font-weight: 500;
    color: #ffffff !important;
    cursor: pointer;
    transition: all 0.2s;
    border: 1px solid transparent;
}

div[data-testid="stRadio"] label:hover {
    background: rgba(168,255,62,0.08);
    color: #ffffff !important;
    border-color: rgba(168,255,62,0.2);
}

div[data-testid="stRadio"] label p,
div[data-testid="stRadio"] label span,
div[data-testid="stRadio"] label div,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] label p,
section[data-testid="stSidebar"] label span,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #ffffff !important;
}

/* ====== HERO ====== */
.hero {
    position: relative;
    text-align: center;
    padding: 4rem 2rem 3rem;
    overflow: hidden;
}

.hero-eyebrow {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #6bff8f;
    background: rgba(107,255,143,0.1);
    border: 1px solid rgba(107,255,143,0.25);
    padding: 0.35rem 1rem;
    border-radius: 50px;
    margin-bottom: 1.5rem;
}

.hero-title {
    font-family: 'Instrument Serif', serif;
    font-size: clamp(3rem, 6vw, 5.5rem);
    font-weight: 400;
    font-style: italic;
    line-height: 1.05;
    color: #ffffff;
    margin: 0 0 0.2rem;
    text-shadow: 0 0 60px rgba(168,255,62,0.2);
}

.hero-title span {
    color: #a8ff3e;
    font-style: italic;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #7ab87a;
    max-width: 520px;
    margin: 1.2rem auto 0;
    line-height: 1.7;
    font-weight: 300;
}

/* ====== STAT PILLS ====== */
.stat-row {
    display: flex;
    justify-content: center;
    gap: 1rem;
    flex-wrap: wrap;
    margin: 2.5rem 0 1rem;
}

.stat-pill {
    background: rgba(20,50,25,0.8);
    border: 1px solid rgba(168,255,62,0.2);
    border-radius: 50px;
    padding: 0.6rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    backdrop-filter: blur(10px);
}

.stat-pill-icon { font-size: 1.2rem; }
.stat-pill-label { font-size: 0.72rem; color: #6ab87a; letter-spacing: 1px; text-transform: uppercase; }
.stat-pill-value { font-size: 1.1rem; font-weight: 700; color: #a8ff3e; }

/* ====== FEATURE CARDS ====== */
.features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1.2rem;
    margin: 2rem 0;
}

.feature-card {
    background: linear-gradient(145deg, #112a18, #0e2214);
    border: 1px solid rgba(168,255,62,0.12);
    border-radius: 18px;
    padding: 2rem 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all 0.35s cubic-bezier(0.34, 1.56, 0.64, 1);
    cursor: pointer;
}

.feature-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #a8ff3e, transparent);
    opacity: 0;
    transition: opacity 0.3s;
}

.feature-card:hover {
    transform: translateY(-6px);
    border-color: rgba(168,255,62,0.4);
    box-shadow: 0 20px 50px rgba(0,0,0,0.5), 0 0 30px rgba(168,255,62,0.08);
}

.feature-card:hover::before { opacity: 1; }

.feature-card-icon {
    font-size: 2.5rem;
    margin-bottom: 1rem;
    display: block;
    filter: drop-shadow(0 0 10px rgba(168,255,62,0.3));
}

.feature-card-title {
    font-weight: 700;
    font-size: 1rem;
    color: #d4f5d4;
    margin-bottom: 0.4rem;
}

.feature-card-desc {
    font-size: 0.82rem;
    color: #5a8a6a;
    line-height: 1.6;
}

.feature-card-badge {
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 0.2rem 0.7rem;
    border-radius: 50px;
    margin-top: 1rem;
    background: rgba(168,255,62,0.12);
    color: #a8ff3e;
    border: 1px solid rgba(168,255,62,0.25);
}

.feature-card-badge.soon {
    background: rgba(255,180,50,0.1);
    color: #ffb432;
    border-color: rgba(255,180,50,0.25);
}

/* ====== SECTION HEADERS ====== */
.section-header {
    margin: 2.5rem 0 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}

.section-header-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(168,255,62,0.3), transparent);
}

.section-header-text {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #6ab87a;
}

/* ====== METRIC CARDS ====== */
.metric-card {
    background: linear-gradient(145deg, #0f2918, #0c2014);
    border: 1px solid rgba(168,255,62,0.15);
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    text-align: center;
}

.metric-icon { font-size: 2rem; margin-bottom: 0.5rem; }
.metric-label { font-size: 0.72rem; letter-spacing: 2px; text-transform: uppercase; color: #5a8a6a; margin-bottom: 0.4rem; }
.metric-value { font-size: 2rem; font-weight: 800; color: #a8ff3e; line-height: 1; }
.metric-unit { font-size: 0.8rem; color: #6ab87a; margin-top: 0.2rem; }

/* ====== FORM SECTIONS ====== */
.form-section {
    background: linear-gradient(145deg, #0f2918, #0c2014);
    border: 1px solid rgba(168,255,62,0.15);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 1.5rem;
}

.form-section-title {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #6ab87a;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ====== INPUTS ====== */
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] select,
div[data-baseweb="select"] {
    background: #0a1f0f !important;
    border: 1px solid rgba(168,255,62,0.2) !important;
    border-radius: 10px !important;
    color: #d4f5d4 !important;
    font-family: 'Sora', sans-serif !important;
}

div[data-testid="stNumberInput"] input:focus {
    border-color: rgba(168,255,62,0.6) !important;
    box-shadow: 0 0 0 2px rgba(168,255,62,0.1) !important;
}

/* Labels */
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label {
    color: #88b888 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px;
}

/* ====== BUTTONS ====== */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #4ddb72, #a8ff3e) !important;
    color: #061008 !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 800 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.5px;
    padding: 0.8rem 2rem !important;
    transition: all 0.25s !important;
    box-shadow: 0 4px 20px rgba(107,255,143,0.2) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(107,255,143,0.35) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* ====== SUCCESS BOX ====== */
div[data-testid="stAlert"] {
    background: linear-gradient(135deg, rgba(30,80,40,0.9), rgba(20,60,30,0.9)) !important;
    border: 1px solid rgba(168,255,62,0.4) !important;
    border-radius: 14px !important;
    color: #a8ff3e !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
}

/* ====== RESULT CARD ====== */
.result-card {
    background: linear-gradient(135deg, #112a18, #163d26);
    border: 2px solid rgba(168,255,62,0.35);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    animation: pop 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
}

@keyframes pop {
    0% { transform: scale(0.9); opacity: 0; }
    100% { transform: scale(1); opacity: 1; }
}

.result-label {
    font-size: 0.72rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #6ab87a;
    margin-bottom: 0.5rem;
}

.result-value {
    font-family: 'Instrument Serif', serif;
    font-style: italic;
    font-size: 2.8rem;
    color: #a8ff3e;
    text-shadow: 0 0 30px rgba(168,255,62,0.3);
    margin-bottom: 0.5rem;
}

.result-icon { font-size: 3.5rem; margin-bottom: 1rem; }

/* ====== TIP BOX ====== */
.tip-box {
    background: rgba(20, 50, 30, 0.7);
    border-left: 3px solid #a8ff3e;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.5rem;
    margin: 1.5rem 0;
    font-size: 0.85rem;
    color: #88b888;
    line-height: 1.7;
}

.tip-box strong { color: #a8ff3e; }

/* ====== DIVIDER ====== */
hr {
    border: none !important;
    border-top: 1px solid rgba(168,255,62,0.12) !important;
    margin: 2rem 0 !important;
}

/* ====== FOOTER ====== */
.footer {
    text-align: center;
    padding: 2rem 1rem;
    font-size: 0.8rem;
    color: #3a6a4a;
    letter-spacing: 1px;
}

/* ====== SCROLLBAR ====== */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a1a0f; }
::-webkit-scrollbar-thumb { background: rgba(168,255,62,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(168,255,62,0.5); }

/* ====== SUBHEADER ====== */
h2, h3 {
    color: #d4f5d4 !important;
    font-family: 'Sora', sans-serif !important;
}

/* ====== COMING SOON BANNER ====== */
.coming-soon-banner {
    background: linear-gradient(135deg, rgba(255,180,50,0.08), rgba(255,140,30,0.05));
    border: 1px solid rgba(255,180,50,0.25);
    border-radius: 16px;
    padding: 3rem 2rem;
    text-align: center;
}

.coming-soon-icon { font-size: 4rem; margin-bottom: 1rem; filter: drop-shadow(0 0 20px rgba(255,180,50,0.4)); }
.coming-soon-title { font-size: 1.5rem; font-weight: 800; color: #ffb432; margin-bottom: 0.5rem; }
.coming-soon-desc { color: #7a6a4a; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_models():
    crop_model = joblib.load("ML models/crp_ML.pkl")
    fert_model = joblib.load("ML models/fertilizer_model.pkl")

    from tensorflow.keras.models import load_model
    disease_model = load_model("ML models/disease_model.h5")

    return crop_model, fert_model, disease_model


crop_model, fert_model, disease_model = load_models()
# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div class='sidebar-brand'>
        <span class='sidebar-brand-icon'>🌾</span>
        <div class='sidebar-brand-name'>AgroMind</div>
        <div class='sidebar-brand-tagline'>Smart Farming AI</div>
    </div>
    """, unsafe_allow_html=True)

    option = st.radio(
        "",
        [
            "🏠  Home",
            "🌾  Crop Prediction",
            "🧪  Fertilizer Guide",
            "🦠  Disease Detection",
            "🌦  Weather Intel"
        ],
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='padding: 1rem; background: rgba(168,255,62,0.06); border-radius: 12px; border: 1px solid rgba(168,255,62,0.12); font-size: 0.78rem; color: #5a8a6a; line-height: 1.7;'>
        💡 <strong style='color:#a8ff3e'>Tip:</strong> Enter your soil data accurately for best AI predictions. Use lab-tested NPK values when possible.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# HOME
# ============================================================
if option == "🏠  Home":

    # Hero
    st.markdown("""
    <div class='hero'>
        <div class='hero-eyebrow'>✦ AI-Powered Agriculture Platform</div>
        <div class='hero-title'>Farm Smarter,<br><span>Grow Better</span></div>
        <div class='hero-subtitle'>Precision recommendations powered by machine learning — from crop selection to soil nutrition.</div>
    </div>
    """, unsafe_allow_html=True)

    # Live Stats
    st.markdown("""
    <div class='stat-row'>
        <div class='stat-pill'>
            <span class='stat-pill-icon'>🌡</span>
            <div>
                <div class='stat-pill-label'>Temperature</div>
                <div class='stat-pill-value'>32°C</div>
            </div>
        </div>
        <div class='stat-pill'>
            <span class='stat-pill-icon'>💧</span>
            <div>
                <div class='stat-pill-label'>Soil Moisture</div>
                <div class='stat-pill-value'>45%</div>
            </div>
        </div>
        <div class='stat-pill'>
            <span class='stat-pill-icon'>🌫</span>
            <div>
                <div class='stat-pill-label'>Humidity</div>
                <div class='stat-pill-value'>60%</div>
            </div>
        </div>
        <div class='stat-pill'>
            <span class='stat-pill-icon'>☀️</span>
            <div>
                <div class='stat-pill-label'>UV Index</div>
                <div class='stat-pill-value'>High</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Divider
    st.markdown("""
    <div class='section-header'>
        <div class='section-header-line'></div>
        <div class='section-header-text'>What AgroMind Can Do</div>
        <div class='section-header-line' style='background: linear-gradient(270deg, rgba(168,255,62,0.3), transparent);'></div>
    </div>
    """, unsafe_allow_html=True)

    # Feature Cards
    st.markdown("""
    <div class='features-grid'>
        <div class='feature-card'>
            <span class='feature-card-icon'>🌾</span>
            <div class='feature-card-title'>Crop Prediction</div>
            <div class='feature-card-desc'>AI analyzes your soil NPK, pH, temperature & rainfall to recommend the ideal crop.</div>
            <span class='feature-card-badge'>✓ Available Now</span>
        </div>
        <div class='feature-card'>
            <span class='feature-card-icon'>🧪</span>
            <div class='feature-card-title'>Fertilizer Guide</div>
            <div class='feature-card-desc'>Get smart fertilizer recommendations based on soil nutrients and your chosen crop.</div>
            <span class='feature-card-badge'>✓ Available Now</span>
        </div>
        <div class='feature-card'>
            <span class='feature-card-icon'>🦠</span>
            <div class='feature-card-title'>Disease Detection</div>
            <div class='feature-card-desc'>Upload a leaf photo and get instant disease diagnosis with treatment plans.</div>
            <span class='feature-card-badge soon'>✓ Available Now</span>
        </div>
        <div class='feature-card'>
            <span class='feature-card-icon'>🌦</span>
            <div class='feature-card-title'>Weather Intel</div>
            <div class='feature-card-desc'>Hyperlocal 7-day forecasts with farming-specific alerts and sowing windows.</div>
            <span class='feature-card-badge soon'>✓ Available Now</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # How it works
    st.markdown("""
    <div class='section-header'>
        <div class='section-header-line'></div>
        <div class='section-header-text'>How It Works</div>
        <div class='section-header-line' style='background: linear-gradient(270deg, rgba(168,255,62,0.3), transparent);'></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, icon, step, title, desc in [
        (c1, "📋", "Step 1", "Enter Your Data", "Input your soil test values — Nitrogen, Phosphorus, Potassium, pH, and local climate readings."),
        (c2, "🤖", "Step 2", "AI Analyzes", "Our trained ML model instantly processes hundreds of data points and crop patterns."),
        (c3, "✅", "Step 3", "Get Recommendation", "Receive a clear, actionable crop or fertilizer recommendation tailored to your farm.")
    ]:
        with col:
            st.markdown(f"""
            <div class='metric-card' style='text-align:left; padding: 1.8rem;'>
                <div style='font-size:2rem; margin-bottom:0.8rem;'>{icon}</div>
                <div style='font-size:0.65rem; letter-spacing:2.5px; text-transform:uppercase; color:#6ab87a; margin-bottom:0.3rem;'>{step}</div>
                <div style='font-weight:700; color:#d4f5d4; font-size:1rem; margin-bottom:0.5rem;'>{title}</div>
                <div style='font-size:0.82rem; color:#5a7a5a; line-height:1.6;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# CROP PREDICTION
# ============================================================
elif option == "🌾  Crop Prediction":

    st.markdown("""
    <div style='margin-bottom: 2rem;'>
        <div class='hero-eyebrow'>🌾 Module 01</div>
        <h2 style='margin: 0.5rem 0 0.3rem; font-size: 2rem; font-family: Instrument Serif, serif; font-style: italic; font-weight: 400;'>Crop Recommendation</h2>
        <p style='color: #5a8a6a; font-size: 0.9rem; margin: 0;'>Enter your soil profile and climate data to get the best crop for your field.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='tip-box'>
        <strong>💡 How to fill this form:</strong> Use values from your latest soil test report.
        N, P, K values are in kg/ha. pH should be between 0–14 (ideal farmland: 5.5–7.5).
        Rainfall is in mm (annual average for your region).
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("<div class='form-section'>", unsafe_allow_html=True)
        st.markdown("<div class='form-section-title'>🧱 Soil Nutrients</div>", unsafe_allow_html=True)
        N = st.number_input("Nitrogen (N) — kg/ha", 0, 150, value=60, help="Nitrogen content in soil")
        P = st.number_input("Phosphorus (P) — kg/ha", 0, 150, value=45, help="Phosphorus content in soil")
        K = st.number_input("Potassium (K) — kg/ha", 0, 150, value=40, help="Potassium content in soil")
        ph = st.number_input("Soil pH", 0.0, 14.0, value=6.5, step=0.1, help="Acidity/Alkalinity of soil")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='form-section'>", unsafe_allow_html=True)
        st.markdown("<div class='form-section-title'>🌤 Climate Data</div>", unsafe_allow_html=True)
        temp = st.number_input("Temperature (°C)", 0.0, 50.0, value=28.0, step=0.5)
        humidity = st.number_input("Humidity (%)", 0, 100, value=65)
        rainfall = st.number_input("Annual Rainfall (mm)", 0.0, 300.0, value=120.0, step=5.0)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🌱  Predict Best Crop  →"):
        data = pd.DataFrame({
            'N': [N], 'P': [P], 'K': [K],
            'temperature': [temp],
            'humidity': [humidity],
            'ph': [ph],
            'rainfall': [rainfall]
        })
        result = crop_model.predict(data)[0]

        st.markdown(f"""
        <div class='result-card'>
            <div class='result-icon'>🌾</div>
            <div class='result-label'>AI Recommended Crop</div>
            <div class='result-value'>{result.title()}</div>
            <p style='color:#5a8a6a; font-size:0.85rem; margin-top:0.5rem;'>
                Based on your soil and climate profile, <strong style='color:#88c888'>{result.title()}</strong> is the most suitable crop for your field.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()

    st.markdown("""
    <div class='tip-box' style='margin-top: 1.5rem;'>
        <strong>📌 Note:</strong> This prediction is based on your input values and historical crop data.
        Always cross-check with your local agricultural extension officer for best results.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FERTILIZER
# ============================================================
elif option == "🧪  Fertilizer Guide":

    st.markdown("""
    <div style='margin-bottom: 2rem;'>
        <div class='hero-eyebrow'>🧪 Module 02</div>
        <h2 style='margin: 0.5rem 0 0.3rem; font-size: 2rem; font-family: Instrument Serif, serif; font-style: italic; font-weight: 400;'>Fertilizer Recommendation</h2>
        <p style='color: #5a8a6a; font-size: 0.9rem; margin: 0;'>Select your crop and enter soil conditions to find the ideal fertilizer blend.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='tip-box'>
        <strong>💡 Tip:</strong> Over-fertilizing is as harmful as under-fertilizing.
        Use this guide to apply only what your soil actually needs — saving cost and protecting soil health.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("<div class='form-section'>", unsafe_allow_html=True)
        st.markdown("<div class='form-section-title'>🧱 Soil Nutrients</div>", unsafe_allow_html=True)
        N = st.number_input("Nitrogen (N) — kg/ha", 0, 150, value=50)
        P = st.number_input("Phosphorus (P) — kg/ha", 0, 150, value=35)
        K = st.number_input("Potassium (K) — kg/ha", 0, 150, value=30)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='form-section'>", unsafe_allow_html=True)
        st.markdown("<div class='form-section-title'>🌤 Field Conditions</div>", unsafe_allow_html=True)
        temp = st.number_input("Temperature (°C)", 0.0, 50.0, value=28.0, step=0.5)
        humidity = st.number_input("Humidity (%)", 0, 100, value=60)
        moisture = st.number_input("Soil Moisture (%)", 0, 100, value=40)
        st.markdown("</div>", unsafe_allow_html=True)

    crop_name = st.selectbox(
        "🌾  Which crop are you growing?",
        ["Rice", "Wheat", "Maize", "Cotton","Sugarcane","Jowar","Ginger","Turmeric"],
        help="Select the crop currently planted in your field"
    )

    crop_map = {"Rice": 0, "Wheat": 1, "Maize": 2, "Cotton": 3,"Sugarcane": 4,"Jowar": 5,"Ginger": 6,"Turmeric": 7}

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🧪  Get Fertilizer Recommendation  →"):
        data = pd.DataFrame({
            'N': [N], 'P': [P], 'K': [K],
            'temperature': [temp],
            'crop': [crop_map[crop_name]]
        })
        result = fert_model.predict(data)[0]

        st.markdown(f"""
        <div class='result-card'>
            <div class='result-icon'>🧪</div>
            <div class='result-label'>Recommended Fertilizer for {crop_name}</div>
            <div class='result-value'>{result}</div>
            <p style='color:#5a8a6a; font-size:0.85rem; margin-top:0.5rem;'>
                Apply <strong style='color:#88c888'>{result}</strong> based on your current soil nutrient levels.
                Follow manufacturer dosage guidelines.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='tip-box' style='margin-top: 1.5rem;'>
        <strong>⚠️ Safety First:</strong> Always wear gloves when handling fertilizers.
        Apply during early morning or evening to avoid nutrient evaporation.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# COMING SOON PAGES
# ============================================================
elif option == "🦠  Disease Detection":

    st.markdown("""
    <div style='margin-bottom: 2rem;'>
        <div class='hero-eyebrow'>🦠 Module 03</div>
        <h2 style='margin: 0.5rem 0 0.3rem; font-size: 2rem; font-family: Instrument Serif, serif; font-style: italic; font-weight: 400;'>Disease Detection</h2>
        <p style='color: #5a8a6a; font-size: 0.9rem;'>Upload a leaf image to detect crop disease using AI.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📸 Upload Leaf Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:

        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Detect Disease"):

            # Preprocess
            img_resized = img.resize((128, 128))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = disease_model.predict(img_array)

            class_names = [
                'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
                'Blueberry___healthy','Cherry___Powdery_mildew','Cherry___healthy',
                'Corn___Cercospora_leaf_spot','Corn___Common_rust','Corn___healthy','Corn___Northern_Leaf_Blight',
                'Grape___Black_rot','Grape___Esca','Grape___healthy','Grape___Leaf_blight',
                'Orange___Haunglongbing','Peach___Bacterial_spot','Peach___healthy',
                'Pepper___Bacterial_spot','Pepper___healthy',
                'Potato___Early_blight','Potato___Late_blight','Potato___healthy',
                'Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew',
                'Strawberry___Leaf_scorch','Strawberry___healthy',
                'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight',
                'Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites',
                'Tomato___Target_Spot','Tomato___Yellow_Leaf_Curl','Tomato___Mosaic_virus','Tomato___healthy'
            ]

            result = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)

            st.markdown(f"""
            <div class='result-card'>
                <div class='result-icon'>🦠</div>
                <div class='result-label'>Detected Disease</div>
                <div class='result-value'>{result.replace("_", " ")}</div>
                <p style='color:#5a8a6a; font-size:0.9rem;'>
                    Confidence: <strong style='color:#88c888'>{confidence*100:.2f}%</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

elif option == "🌦  Weather Intel":

    st.markdown("""
    <div style='margin-bottom: 2rem;'>
        <div class='hero-eyebrow'>🌦 Module 04</div>
        <h2 style='margin: 0.5rem 0 0.3rem; font-size: 2rem; font-family: Instrument Serif, serif; font-style: italic; font-weight: 400;'>Weather Intelligence</h2>
        <p style='color: #5a8a6a; font-size: 0.9rem;'>Get real-time weather data for better farming decisions.</p>
    </div>
    """, unsafe_allow_html=True)

    city = st.text_input(
        "📍 Enter City Name",
        "Kolhapur",
        help="Enter your location for real-time weather insights"
    )

    if st.button("🌦 Get Weather Data"):

        weather = get_weather(city)   # ✅ CALL FUNCTION HERE

        if weather:

           st.markdown(f"""
           <div class='result-card'>
         <div class='result-icon'>🌤</div>
          <div class='result-label'>Live Weather in {city.title()}</div>
          <div class='result-value'>{weather['temp']}°C</div>

          <p style='color:#5a8a6a; font-size:0.9rem;'>
          Condition: <strong style='color:#88c888'>{weather['condition'].title()}</strong><br>
         Humidity: {weather['humidity']}%<br>
         Wind Speed: {weather['wind']} m/s
          </p>
          </div>
           """, unsafe_allow_html=True)  

        else:
            st.error("❌ Could not fetch weather. Check city name or API.")

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class='footer'>
    <hr style='margin-bottom: 1.5rem;'>
    🌱 AgroMind &nbsp;·&nbsp; Built with AI for Indian Farmers &nbsp;·&nbsp; 2025
</div>
""", unsafe_allow_html=True)

