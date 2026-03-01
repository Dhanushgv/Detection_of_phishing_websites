import streamlit as st
import pandas as pd
import joblib
import torch
import requests
from src.features import features_from_url
from src.dataset import prepare_sequences
from src.cnn_model_torch import CharCNN
import plotly.graph_objects as go

# ----------------- UI CONFIG -----------------
st.set_page_config(
    page_title="🛡 Phishing URL Detector",
    page_icon="🔍",
    layout="centered",
)

# ----------------- CUSTOM CSS -----------------
st.markdown("""
<style>
    /* Dark grey gradient background */
    .stApp {
        background: linear-gradient(135deg, #1f2937 0%, #111827 50%, #0f172a 100%);
    }
    
    /* Main container styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Title styling */
    h1 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        padding: 1rem 0;
    }
    
    /* Subtitle */
    h3 {
        color: #f0f0f0 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #374151;
        background: #1f2937;
        color: #f9fafb;
        padding: 0.75rem 1rem;
        font-size: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        background: #111827;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.3);
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #9ca3af;
    }
    
    /* Button styling with gradient */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 16px;
        font-weight: 600;
        width: 100%;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
    }
    
    /* Alert boxes with dark theme */
    .stAlert {
        background: rgba(31, 41, 55, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border-left: 4px solid;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        color: #f9fafb;
    }
    
    /* Success alert */
    div[data-baseweb="notification"][kind="success"] {
        background: rgba(16, 185, 129, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Info alert */
    div[data-baseweb="notification"][kind="info"] {
        background: rgba(59, 130, 246, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Warning alert */
    div[data-baseweb="notification"][kind="warning"] {
        background: rgba(245, 158, 11, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Error alert */
    div[data-baseweb="notification"][kind="error"] {
        background: rgba(239, 68, 68, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(31, 41, 55, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        font-weight: 600;
        color: white;
        border: 1px solid #374151;
    }
    
    .streamlit-expanderContent {
        background: rgba(17, 24, 39, 0.95);
        border-radius: 0 0 10px 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        border: 1px solid #374151;
        border-top: none;
        color: #f9fafb;
    }
    
    /* Plotly chart container */
    .js-plotly-plot {
        border-radius: 12px;
        background: rgba(31, 41, 55, 0.95);
        backdrop-filter: blur(10px);
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        border: 1px solid #374151;
    }
    
    /* Caption */
    .stCaption {
        color: white !important;
        text-align: center;
        font-size: 14px;
        opacity: 0.8;
    }
    
    /* Horizontal rule */
    hr {
        border-color: rgba(255, 255, 255, 0.3);
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡 Phishing URL Detector")
st.markdown("### Enter a website URL below to check if it's *legit or phishing* ⚠")


@st.cache_resource
def load_models():
    rf = joblib.load("models/rf_model.joblib")

    vocab_size = 94  # update this to your model vocab size
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cnn_model = CharCNN(vocab_size=vocab_size).to(device)
    cnn_model.load_state_dict(torch.load("models/cnn_best_torch.pt", map_location=device))
    cnn_model.eval()
    return rf, cnn_model, device

rf, cnn_model, device = load_models()

def check_url_reachability(url):
    """Check if the given URL is reachable."""
    try:
        response = requests.head(url, timeout=20, allow_redirects=True)
        if response.status_code < 400:
            return True
        else:
            return False
    except Exception:
        return False

url = st.text_input("🌐 Enter URL", "https://example.com")

if st.button("Predict") and url:
    st.info(" Checking reachability...")
    reachable = check_url_reachability(url)

    if not reachable:
        st.error(" The URL seems *unreachable*. Please check if it's correct or online.")
    else:
        st.success(" The URL is *reachable*. Proceeding with analysis...")

        # ----------------- MODEL PREDICTION -----------------
        rf_feat = pd.DataFrame([features_from_url(url)])
        rf_p = float(rf.predict_proba(rf_feat)[:, 1][0])

        X = prepare_sequences([url], max_len=200)
        X = torch.tensor(X, dtype=torch.long, device=device)
        with torch.no_grad():
            cnn_p = float(cnn_model(X).cpu().item())

        prob = (rf_p + cnn_p) / 2.0

        # ----------------- GAUGE METER -----------------
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Phishing Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 30], 'color': "green"},
                    {'range': [30, 60], 'color': "orange"},
                    {'range': [60, 100], 'color': "red"},
                ],
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        # ----------------- RESULT -----------------
        if prob < 0.3:
            st.success("✅ This website looks *Safe*.")
        elif prob < 0.6:
            st.warning("⚠ This website looks *Suspicious*. Be careful!")
        else:
            st.error("🚨 This website is likely *Phishing*! Avoid it.")

        # ----------------- EXTRA DETAILS -----------------
        with st.expander("🔎 More details"):
            st.write({
                "Final Probability": round(prob, 3),
                "RF Score": round(rf_p, 3),
                "CNN Score": round(cnn_p, 3),
                "Label": int(prob >= 0.5),
                "Reachable": reachable
            })

st.markdown("---")
st.caption("Its the probability")