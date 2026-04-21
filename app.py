# app.py

import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# ==============================
# Config
# ==============================

MODEL_PATH = "saved_models/bert-base-uncased"
MAX_LEN = 64

# ==============================
# Page Styling (🔥 makes it look pro)
# ==============================

st.set_page_config(page_title="News Classifier", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    .big-font {
        font-size: 26px !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================
# Load model
# ==============================

@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ==============================
# Prediction
# ==============================

def predict(text):
    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**encoding)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    pred_id = torch.argmax(probs).item()

    label = model.config.id2label[pred_id]

    return label, probs


# ==============================
# Header
# ==============================

st.markdown("<div class='big-font'>📰 AI News Classifier</div>", unsafe_allow_html=True)
st.caption("Classify news into categories using BERT")

st.markdown("---")

# ==============================
# Layout
# ==============================

col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area(
        "✍️ Enter news text",
        placeholder="Type a news headline or description...",
        height=180
    )

    predict_btn = st.button("🚀 Predict")

with col2:
    st.markdown("### 📌 Examples")
    examples = [
        "Stock markets crash amid global uncertainty",
        "New AI model beats humans in language tasks",
        "Government announces new economic reforms",
        "Olympics 2026 preparations underway"
    ]

    for ex in examples:
        if st.button(ex):
            user_input = ex


# ==============================
# Prediction Output
# ==============================

if predict_btn:

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        label, probs = predict(user_input)

        st.markdown("---")

        # Main result
        st.success(f"🏷️ **Prediction: {label}**")

        # Confidence bar
        confidence = torch.max(probs).item()
        st.progress(confidence)
        st.write(f"Confidence: **{confidence:.2f}**")

        # Top predictions
        st.markdown("### 🔍 Top Predictions")

        labels = model.config.id2label

        prob_list = probs.tolist()
        top_indices = sorted(range(len(prob_list)), key=lambda i: prob_list[i], reverse=True)[:5]

        for i in top_indices:
            st.write(f"**{labels[i]}** — {prob_list[i]:.3f}")
            st.progress(prob_list[i])

# ==============================
# Footer
# ==============================

st.markdown("---")
st.caption("Built using BERT + Streamlit")