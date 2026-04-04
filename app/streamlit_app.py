import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import sys, os
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import ModelDevelopment
from inference import BasicInference

st.set_page_config(page_title="Face Mask Detector | IWMI",
                   page_icon="😷", layout="wide")

# Load Model 
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = ModelDevelopment(num_classes=2)
    model.load_state_dict(torch.load('models/best_model.pth',
                                      map_location=device))
    model.to(device)   # ← ADD THIS LINE — moves model weights to GPU
    model.eval()
    inferencer = BasicInference(model, device, img_size=128,
                                classes=['with_mask', 'without_mask'])
    return inferencer, device

# Sidebar 
with st.sidebar:
    st.title("Model Info")
    st.markdown("**Architecture:** Custom CNN")
    st.markdown("**Framework:** PyTorch")
    st.markdown("""
    **Layers:**
    - 4 × Conv Block (Conv → BN → ReLU → MaxPool → Dropout)
    - 2 × Fully Connected
    - Total parameters: ~2.1M
    """)
    st.markdown("**Input Size:** 128 × 128 × 3")
    st.markdown("**Classes:** with_mask, without_mask")
    st.markdown("---")
    st.markdown("**IWMI Data Science Assessment**")

# Main UI 
st.title("Face Mask Detector")
st.markdown("Upload an image to detect whether people are wearing face masks.")

uploaded_file = st.file_uploader("Choose an image...",
                                  type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    inferencer, device = load_model()

    # Save temp file for OpenCV
    temp_path = "temp_upload.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(temp_path, use_column_width=True)

    with col2:
        st.subheader("🔍 Detection Result")
        with st.spinner("Detecting faces and classifying..."):
            annotated, results = inferencer.detect_images(temp_path)

        if len(results) == 0:
            st.warning("No faces detected in the image.")
        else:
            st.image(annotated, use_column_width=True)

    # Results Table 
    if results:
        st.markdown("---")
        st.subheader("Prediction Details")

        for i, r in enumerate(results):
            st.markdown(f"**Face {i+1}:** `{r['class']}` — "
                        f"Confidence: `{r['confidence']:.1f}%`")

            # Bar chart of class probabilities
            fig, ax = plt.subplots(figsize=(6, 2.5))
            classes = ['with_mask', 'without_mask']
            colors  = ['#2ecc71', '#e74c3c']
            bars = ax.barh(classes, r['probabilities'] * 100, color=colors)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Confidence (%)")
            ax.set_title(f"Face {i+1} — Top Class Probabilities")
            for bar, val in zip(bars, r['probabilities'] * 100):
                ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                        f"{val:.1f}%", va='center')
            st.pyplot(fig)
            plt.close()

    os.remove(temp_path)