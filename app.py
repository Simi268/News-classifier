# app.py

import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# ==============================
# Page Config
# ==============================

st.set_page_config(
    page_title="AI News Classifier",
    page_icon="📰",
    layout="wide"
)

# ==============================
# Load Model (cached)
# ==============================

@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("Simik26/news-classifier")
    model = BertForSequenceClassification.from_pretrained("Simik26/news-classifier")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ==============================
# Prediction Function
# ==============================

def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    pred_id = torch.argmax(probs).item()

    return {
        "label": model.config.id2label[pred_id],
        "confidence": probs[pred_id].item(),
        "probs": probs.tolist()
    }

# ==============================
# Custom CSS (SaaS UI)
# ==============================

st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.main {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}
.card {
    background: #111827;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 0 15px rgba(0,0,0,0.4);
}
.stButton > button {
    background: linear-gradient(to right, #6366f1, #3b82f6);
    color: white;
    border-radius: 8px;
    height: 3em;
    font-weight: 600;
}
textarea {
    background-color: #020617 !important;
    color: white !important;
}
.stProgress > div > div {
    background-color: #6366f1;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# Sidebar
# ==============================

with st.sidebar:
    st.markdown("## 🚀 AI News Classifier")
    st.write("Classify news using fine-tuned BERT model.")
    st.markdown("---")
    st.write("### 📊 Features")
    st.write("- Real-time predictions")
    st.write("- Confidence scores")
    st.write("- Top 5 categories")
    st.markdown("---")
    st.caption("Built using BERT + Streamlit")

# ==============================
# Header
# ==============================

st.markdown("""
<div style="
    padding: 30px 20px;
    border-radius: 14px;
    background: linear-gradient(135deg, #1e293b, #020617);
    box-shadow: 0 0 25px rgba(0,0,0,0.6);
    margin-bottom: 20px;
">
    <div style="font-size: 34px; font-weight: 700; color: white;">
        📰 AI News Classifier
    </div>
    <div style="font-size: 16px; color: #94a3b8; margin-top: 8px;">
        Turn raw news into structured insights instantly
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ==============================
# Layout
# ==============================

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    user_input = st.text_area(
        "Enter your news text",
        placeholder="e.g. Global markets fall amid rising inflation fears...",
        height=150
    )

    predict_btn = st.button("⚡ Analyze News")

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.markdown("### ✨ Try Examples")

    examples = [
        "Stock markets crash amid global uncertainty",
        "New AI model beats humans in language tasks",
        "Government announces new economic reforms",
        "Olympics 2026 preparations underway"
    ]

    for ex in examples:
        if st.button(ex):
            user_input = ex

    st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# Prediction Output
# ==============================

if predict_btn:

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            result = predict(user_input)

        label = result["label"]
        confidence = result["confidence"]
        probs = result["probs"]

        st.markdown("---")

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.success(f"🏷️ Prediction: {label}")
        st.progress(confidence)
        st.write(f"Confidence: **{confidence * 100:.2f}%**")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### 🔍 Top Predictions")

        labels = list(model.config.id2label.values())

        top_indices = sorted(
            range(len(probs)),
            key=lambda i: probs[i],
            reverse=True
        )[:5]

        for i in top_indices:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.write(f"**{labels[i]}**")
            st.progress(probs[i])
            st.write(f"{probs[i]*100:.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# Footer
# ==============================

st.markdown("---")
st.caption("© 2026 AI News Classifier  🚀")