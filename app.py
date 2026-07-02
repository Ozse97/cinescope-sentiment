"""
app.py — Sentiment Analysis Dashboard
=======================================
A cinematic-themed Streamlit UI that loads the trained Logistic Regression
pipeline and predicts sentiment (Positive / Negative) for any movie review.

Run
---
    streamlit run app.py
"""

import os
import pickle
import time
import streamlit as st
import numpy as np

# Required so pickle can resolve the TfidfVectorizer's custom preprocessor
# function (it was pickled by reference to this module).
from text_utils import clean_text  # noqa: F401

# ─────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CineScope · Sentiment AI",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CSS — dark cinematic theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Playfair+Display:wght@700;900&display=swap');

/* ── Reset / root ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0d0d0f;
    color: #e8e8e8;
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0d0d0f 0%, #12121a 60%, #0d0d0f 100%);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Main content wrapper ── */
.block-container {
    max-width: 780px;
    padding: 2.5rem 2rem 4rem;
}

/* ── Hero heading ── */
.hero-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 3.2rem;
    font-weight: 900;
    line-height: 1.1;
    background: linear-gradient(135deg, #e63946 0%, #f4a261 55%, #2ec4b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem;
    letter-spacing: -1px;
}
.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #555566;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    margin-bottom: 2.4rem;
}

/* ── Divider ── */
.divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #2ec4b6 40%, #e63946 70%, transparent);
    margin: 2rem 0;
    opacity: 0.45;
}

/* ── Text area ── */
textarea {
    background: #14141c !important;
    border: 1px solid #252530 !important;
    border-radius: 8px !important;
    color: #e8e8e8 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.88rem !important;
    line-height: 1.7 !important;
    transition: border-color 0.25s !important;
}
textarea:focus {
    border-color: #2ec4b6 !important;
    box-shadow: 0 0 0 2px rgba(46,196,182,0.12) !important;
}

/* ── Analyze button ── */
div.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #e63946 0%, #c1121f 100%);
    color: #fff;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    padding: 0.75rem 1.5rem;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s;
    box-shadow: 0 4px 20px rgba(230,57,70,0.35);
    margin-top: 0.6rem;
}
div.stButton > button:hover {
    opacity: 0.92;
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(230,57,70,0.45);
}
div.stButton > button:active {
    transform: translateY(0);
}

/* ── Result cards ── */
.result-positive {
    background: linear-gradient(135deg, #0a2926 0%, #0d1f1e 100%);
    border: 1px solid rgba(46,196,182,0.4);
    border-left: 4px solid #2ec4b6;
    border-radius: 12px;
    padding: 1.8rem 2rem;
    margin-top: 1.6rem;
    animation: slideUp 0.4s ease;
}
.result-negative {
    background: linear-gradient(135deg, #200a0c 0%, #1a0d0e 100%);
    border: 1px solid rgba(230,57,70,0.4);
    border-left: 4px solid #e63946;
    border-radius: 12px;
    padding: 1.8rem 2rem;
    margin-top: 1.6rem;
    animation: slideUp 0.4s ease;
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.verdict-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 0.3rem;
}
.verdict-text {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 2.2rem;
    font-weight: 900;
    line-height: 1.1;
    margin-bottom: 0.8rem;
}
.verdict-positive { color: #2ec4b6; }
.verdict-negative { color: #e63946; }

/* ── Confidence bar ── */
.conf-wrapper {
    margin-top: 1rem;
}
.conf-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #888;
    letter-spacing: 0.1em;
    margin-bottom: 0.35rem;
}
.conf-bar-bg {
    background: #1a1a24;
    border-radius: 100px;
    height: 8px;
    overflow: hidden;
}
.conf-bar-fill-pos {
    background: linear-gradient(90deg, #2ec4b6, #00f5d4);
    height: 8px;
    border-radius: 100px;
    transition: width 0.8s cubic-bezier(.22,.68,0,1.2);
}
.conf-bar-fill-neg {
    background: linear-gradient(90deg, #e63946, #ff6b6b);
    height: 8px;
    border-radius: 100px;
    transition: width 0.8s cubic-bezier(.22,.68,0,1.2);
}
.conf-pct {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    margin-top: 0.5rem;
}

/* ── Metric chips ── */
.chips-row {
    display: flex;
    gap: 0.75rem;
    margin-top: 1.1rem;
    flex-wrap: wrap;
}
.chip {
    background: #1a1a24;
    border: 1px solid #252530;
    border-radius: 100px;
    padding: 0.25rem 0.85rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #888;
}

/* ── History cards ── */
.hist-card {
    background: #13131a;
    border: 1px solid #1e1e28;
    border-radius: 8px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.6rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
}
.hist-pos { color: #2ec4b6; }
.hist-neg { color: #e63946; }

/* ── Sidebar info ── */
[data-testid="stSidebar"] {
    background: #0f0f17;
    border-right: 1px solid #1a1a24;
}
[data-testid="stSidebar"] * { font-family: 'Space Mono', monospace !important; }

/* ── Info boxes ── */
.info-box {
    background: #13131a;
    border: 1px solid #1e1e28;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #666;
    line-height: 1.7;
    margin-top: 1rem;
}
.info-box strong { color: #aaa; }

/* ── Examples section ── */
.example-btn-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #555;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "sentiment_model.pkl")

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

pipeline = load_model()

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "review_text" not in st.session_state:
    st.session_state.review_text = ""

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎬 CineScope AI")
    st.markdown("---")
    st.markdown("**Model**")
    st.markdown("Logistic Regression + TF-IDF")
    st.markdown("**Dataset**")
    st.markdown("IMDb Movie Reviews")
    st.markdown("**Features**")
    st.markdown("60,000 n-gram TF-IDF")
    st.markdown("---")

    if os.path.exists(os.path.join(os.path.dirname(__file__), "results", "accuracy.png")):
        st.markdown("**Accuracy Plot**")
        st.image("results/accuracy.png", use_container_width=True)

    if os.path.exists(os.path.join(os.path.dirname(__file__), "results", "confusion_matrix.png")):
        st.markdown("**Confusion Matrix**")
        st.image("results/confusion_matrix.png", use_container_width=True)

    st.markdown("---")
    st.markdown("<span style='font-size:0.7rem;color:#444;'>MIT License · 2025</span>",
                unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Hero header
# ─────────────────────────────────────────────
st.markdown('<div class="hero-title">CineScope</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">✦ AI-Powered Movie Review Sentiment Analysis</div>',
            unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

if pipeline is None:
    st.error("⚠️  Model file not found. Run `python model.py` first to train and save the model.")
    st.stop()

# ─────────────────────────────────────────────
# Example reviews
# ─────────────────────────────────────────────
EXAMPLES = {
    "👍 Glowing review":
        "An absolute triumph of cinema. The performances are electrifying, "
        "the direction masterful, and the story deeply moving. One of the finest "
        "films I have had the pleasure of watching this decade.",
    "👎 Scathing review":
        "A complete disaster from start to finish. The script is incoherent, "
        "the acting painfully wooden, and the pacing unbearably slow. "
        "I want my two hours back.",
    "😐 Mixed feelings":
        "Some breathtaking visuals and a strong lead performance, "
        "but the third act falls apart completely. A frustrating near-miss.",
}

st.markdown('<div class="example-btn-label">Quick examples</div>', unsafe_allow_html=True)
cols = st.columns(len(EXAMPLES))
for col, (label, text) in zip(cols, EXAMPLES.items()):
    if col.button(label, use_container_width=True):
        st.session_state.review_text = text

# ─────────────────────────────────────────────
# Text input
# ─────────────────────────────────────────────
review = st.text_area(
    label="Movie Review",
    value=st.session_state.review_text,
    placeholder="Paste or type a movie review here…",
    height=160,
    label_visibility="collapsed",
)

analyze_clicked = st.button("⚡  ANALYZE SENTIMENT", use_container_width=True)

# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────
if analyze_clicked:
    text = review.strip()
    if not text:
        st.warning("Please enter a review before analyzing.")
    elif len(text.split()) < 3:
        st.warning("Please enter at least a few words for a meaningful prediction.")
    else:
        with st.spinner("Analyzing…"):
            time.sleep(0.35)   # brief theatrical pause
            pred   = pipeline.predict([text])[0]
            proba  = pipeline.predict_proba([text])[0]
            conf   = float(max(proba)) * 100
            is_pos = pred == 1
            label  = "Positive" if is_pos else "Negative"
            emoji  = "✦" if is_pos else "✗"
            card_cls   = "result-positive" if is_pos else "result-negative"
            verdict_cls= "verdict-positive" if is_pos else "verdict-negative"
            bar_cls    = "conf-bar-fill-pos" if is_pos else "conf-bar-fill-neg"
            bar_color  = "#2ec4b6" if is_pos else "#e63946"

            word_count  = len(text.split())
            char_count  = len(text)
            pos_score   = float(proba[1]) * 100
            neg_score   = float(proba[0]) * 100

            st.markdown(f"""
<div class="{card_cls}">
  <div class="verdict-label">VERDICT</div>
  <div class="verdict-text {verdict_cls}">{emoji}  {label}</div>
  <div class="conf-wrapper">
    <div class="conf-label">CONFIDENCE</div>
    <div class="conf-bar-bg">
      <div class="{bar_cls}" style="width:{conf:.1f}%"></div>
    </div>
    <div class="conf-pct" style="color:{bar_color}">{conf:.1f}%</div>
  </div>
  <div class="chips-row">
    <span class="chip">📝 {word_count} words</span>
    <span class="chip">🟢 POS {pos_score:.1f}%</span>
    <span class="chip">🔴 NEG {neg_score:.1f}%</span>
    <span class="chip">🔤 {char_count} chars</span>
  </div>
</div>
""", unsafe_allow_html=True)

            # Store in history (most recent first, max 8)
            st.session_state.history.insert(0, {
                "text":  text[:70] + "…" if len(text) > 70 else text,
                "label": label,
                "conf":  conf,
                "is_pos": is_pos,
            })
            st.session_state.history = st.session_state.history[:8]

# ─────────────────────────────────────────────
# Analysis history
# ─────────────────────────────────────────────
if st.session_state.history:
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(
        '<div class="verdict-label" style="margin-bottom:0.8rem">RECENT ANALYSES</div>',
        unsafe_allow_html=True
    )
    for item in st.session_state.history:
        color_cls = "hist-pos" if item["is_pos"] else "hist-neg"
        icon      = "✦" if item["is_pos"] else "✗"
        st.markdown(f"""
<div class="hist-card">
  <span style="color:#555;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
               padding-right:1rem">{item['text']}</span>
  <span class="{color_cls}">{icon} {item['label']}&nbsp;&nbsp;{item['conf']:.0f}%</span>
</div>""", unsafe_allow_html=True)

    if st.button("Clear history", key="clear"):
        st.session_state.history = []
        st.rerun()

# ─────────────────────────────────────────────
# Footer info
# ─────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
<strong>How it works:</strong><br>
The model transforms your review text into a high-dimensional TF-IDF vector
(up to 60k unigram + bigram features) and feeds it through a calibrated
Logistic Regression classifier trained on 50,000 IMDb movie reviews.<br><br>
<strong>Stack:</strong> Python · Scikit-learn · Streamlit · TF-IDF · Logistic Regression
</div>
""", unsafe_allow_html=True)
