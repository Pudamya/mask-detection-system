import streamlit as st
import torch
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import ModelDevelopment
from inference import BasicInference

# Page config
st.set_page_config(
    page_title="Face Mask Detection System | IWMI Assessment",
    page_icon="FM",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
    .main {
        padding-top: 1.2rem;
    }

    .hero-card {
        padding: 1.4rem 1.6rem;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(20,24,35,0.95), rgba(10,14,24,0.95));
        margin-bottom: 1rem;
    }

    .metric-card {
        padding: 1rem 1.2rem;
        border-radius: 16px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.06);
        text-align: center;
    }

    .section-card {
        padding: 1rem 1.2rem;
        border-radius: 16px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 1rem;
    }

    .small-label {
        font-size: 0.9rem;
        opacity: 0.75;
        margin-bottom: 0.2rem;
    }

    .big-value {
        font-size: 1.4rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModelDevelopment(num_classes=2)

    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    inferencer = BasicInference(
        model=model,
        device=device,
        img_size=128,
        classes=['with_mask', 'without_mask']
    )
    return inferencer, device


@st.cache_data
def load_metrics():
    metrics_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def render_sidebar_metrics(metrics):
    st.markdown("---")
    st.markdown("**Achieved Metrics**")
    if metrics:
        st.markdown(f"**Accuracy:** {metrics.get('accuracy', 0.0) * 100:.2f}%")
        st.markdown(f"**Precision:** {metrics.get('precision_weighted', 0.0):.4f}")
        st.markdown(f"**Recall:** {metrics.get('recall_weighted', 0.0):.4f}")
        st.markdown(f"**F1 Score:** {metrics.get('f1_weighted', 0.0):.4f}")
    else:
        st.caption("Run training + evaluation to generate metrics.json")


def get_top3_display(probabilities):
    probs = np.array(probabilities, dtype=float)
    with_mask_score = float(probs[0] * 100)
    without_mask_score = float(probs[1] * 100)
    uncertainty_score = float((100.0 - abs(with_mask_score - without_mask_score)) / 2.0)

    labels = ["With Mask", "Without Mask", "Uncertainty Score"]
    values = [with_mask_score, without_mask_score, uncertainty_score]

    order = np.argsort(values)[::-1]
    ordered_labels = [labels[i] for i in order]
    ordered_values = [values[i] for i in order]

    return ordered_labels, ordered_values


metrics = load_metrics()

# Sidebar
with st.sidebar:
    st.title("System Overview")
    st.markdown("**Architecture:** CustomAttentionCNN")
    st.markdown("**Framework:** PyTorch")
    st.markdown("**Input Size:** 128 x 128 x 3")
    st.markdown("**Classes:** with_mask, without_mask")

    st.markdown("---")
    st.markdown("**Backbone Design**")
    st.markdown("""
    - Block 1: 32 channels
    - Block 2: 64 channels
    - Block 3: 128 channels
    - Block 4: 256 channels
    - SE attention block
    - Global average pooling head
    """)

    st.markdown("---")
    st.markdown("**Training Setup**")
    st.markdown("""
    - Adam optimizer
    - ReduceLROnPlateau scheduler
    - Batch normalization
    - Dropout regularization
    - Label smoothing
    """)

    render_sidebar_metrics(metrics)

    st.markdown("---")
    st.caption("IWMI Data Science Intern Assessment")


# Hero
st.markdown("""
<div class="hero-card">
    <h1 style="margin-bottom:0.4rem;">Face Mask Detection System</h1>
    <p style="margin-bottom:0.2rem;">
        Custom PyTorch CNN for static image face-mask classification with face detection,
        confidence scoring, and interactive visual analysis.
    </p>
    <p style="opacity:0.75; margin-top:0.5rem;">
        Built as an IWMI Data Science Intern assessment submission.
    </p>
</div>
""", unsafe_allow_html=True)

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.markdown("""
    <div class="metric-card">
        <div class="small-label">Model Type</div>
        <div class="big-value">Custom CNN</div>
    </div>
    """, unsafe_allow_html=True)

with metric_col2:
    st.markdown("""
    <div class="metric-card">
        <div class="small-label">Input Size</div>
        <div class="big-value">128 x 128</div>
    </div>
    """, unsafe_allow_html=True)

with metric_col3:
    st.markdown("""
    <div class="metric-card">
        <div class="small-label">Classes</div>
        <div class="big-value">2</div>
    </div>
    """, unsafe_allow_html=True)

with metric_col4:
    st.markdown("""
    <div class="metric-card">
        <div class="small-label">Inference Mode</div>
        <div class="big-value">Static Image</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

tab1, tab2, tab3 = st.tabs(["Live Detection", "Performance", "System Notes"])

with tab1:
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear frontal face image for best results"
    )

    if uploaded_file is not None:
        inferencer, device = load_model()

        temp_path = "temp_upload.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            st.image(temp_path, width="stretch")

        with col2:
            st.subheader("Detection Result")
            with st.spinner("Detecting faces and classifying..."):
                try:
                    annotated, results = inferencer.detect_images(temp_path)
                    st.image(annotated, width="stretch")

                    used_fallback = any(r.get('used_full_image_fallback', False) for r in results)
                    if used_fallback:
                        st.info("No reliable face box was detected, so the system used full-image classification as a fallback.")

                except Exception as e:
                    st.error(f"Error during detection: {e}")
                    results = []

        if results:
            st.markdown("---")
            st.subheader("Prediction Details")

            for i, r in enumerate(results):
                predicted_label = r.get('class', 'unknown')
                raw_label = r.get('raw_class', predicted_label)
                confidence = float(r.get('confidence', 0.0))
                probabilities = r.get('probabilities', None)
                blur_detected = r.get('blur_detected', False)
                blur_score = float(r.get('blur_score', 0.0))
                blur_text = "Yes" if blur_detected else "No"
                source_mode = "Full-image fallback" if r.get("used_full_image_fallback", False) else "Detected face crop"

                with st.container():
                    if predicted_label == 'with_mask':
                        st.success(f"Face {i+1}: **WITH MASK** - Confidence: `{confidence:.1f}%`")
                    elif predicted_label == 'without_mask':
                        st.error(f"Face {i+1}: **WITHOUT MASK** - Confidence: `{confidence:.1f}%`")
                    else:
                        st.warning(f"Face {i+1}: **UNCERTAIN** - Confidence: `{confidence:.1f}%`")

                    st.markdown(f"""
                    <div class="section-card">
                        <strong>Prediction Source:</strong> {source_mode}<br>
                        <strong>Raw Prediction:</strong> {raw_label}<br>
                        <strong>Displayed Label:</strong> {predicted_label}<br>
                        <strong>Confidence:</strong> {confidence:.2f}%<br>
                        <strong>Blur Detected:</strong> {blur_text}<br>
                        <strong>Blur Score:</strong> {blur_score:.2f}
                    </div>
                    """, unsafe_allow_html=True)

                    if probabilities is None:
                        probs = np.array([0.5, 0.5], dtype=float)
                    else:
                        probs = np.array(probabilities, dtype=float)

                    top3_labels, top3_values = get_top3_display(probs)

                    fig, ax = plt.subplots(figsize=(7, 2.6))
                    fig.patch.set_facecolor('#0e1117')
                    ax.set_facecolor('#0e1117')

                    bar_colors = ['#2ecc71', '#e74c3c', '#f1c40f']

                    bars = ax.barh(
                        top3_labels,
                        top3_values,
                        color=bar_colors[:len(top3_labels)],
                        height=0.5,
                        edgecolor='none'
                    )

                    for bar, val in zip(bars, top3_values):
                        ax.text(
                            min(val + 1.5, 95),
                            bar.get_y() + bar.get_height() / 2,
                            f"{val:.1f}%",
                            va='center',
                            ha='left',
                            color='white',
                            fontsize=11,
                            fontweight='bold'
                        )

                    ax.set_xlim(0, 100)
                    ax.set_xlabel("Confidence (%)", color='white')
                    ax.set_title(f"Face {i+1} Top 3 Prediction View", color='white', fontsize=12)
                    ax.tick_params(colors='white')
                    ax.spines['bottom'].set_color('#444')
                    ax.spines['left'].set_color('#444')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.xaxis.label.set_color('white')

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                    st.caption("Top 3 view includes the two class probabilities plus an uncertainty score because this is a binary classifier.")

        elif uploaded_file is not None:
            st.markdown("---")
            st.warning("No face or usable fallback prediction could be produced. Try a clearer, front-facing image with better lighting.")

        if os.path.exists(temp_path):
            os.remove(temp_path)

    else:
        st.info("Upload an image above to get started.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### How it works")
            st.markdown(
                "1. Upload a `.jpg` or `.png` image\n"
                "2. Haar Cascade detects faces\n"
                "3. CNN classifies each face\n"
                "4. Results shown with confidence"
            )
        with col2:
            st.markdown("#### Best Results With")
            st.markdown(
                "- Clear frontal face photos\n"
                "- Good lighting\n"
                "- Face clearly visible\n"
                "- Single or multiple people"
            )
        with col3:
            st.markdown("#### Limitations")
            st.markdown(
                "- Side profile faces may be missed\n"
                "- Very small faces may not detect\n"
                "- Heavy occlusion may confuse model"
            )

with tab2:
    st.subheader("Training and Evaluation Artifacts")

    metrics_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'metrics.json')
    train_curve_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'training_curves.png')
    confusion_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'confusion_matrix.png')

    if metrics:
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0.0) * 100:.2f}%")
        with mc2:
            st.metric("Precision", f"{metrics.get('precision_weighted', 0.0):.4f}")
        with mc3:
            st.metric("Recall", f"{metrics.get('recall_weighted', 0.0):.4f}")
        with mc4:
            st.metric("F1 Score", f"{metrics.get('f1_weighted', 0.0):.4f}")
    else:
        st.info("metrics.json not found yet. Run training and evaluation first.")

    c1, c2 = st.columns(2)

    with c1:
        if os.path.exists(train_curve_path):
            st.image(train_curve_path, caption="Training Curves", width="stretch")
        else:
            st.info("Training curves image not found yet.")

    with c2:
        if os.path.exists(confusion_path):
            st.image(confusion_path, caption="Confusion Matrix", width="stretch")
        else:
            st.info("Confusion matrix image not found yet.")

with tab3:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Strengths")
        st.markdown("""
        - Uses a custom CNN designed from scratch
        - Handles single and multiple face detections
        - Shows confidence-aware predictions
        - Includes preprocessing and regularization for better generalization
        """)

        st.markdown("### Success Cases")
        st.markdown("""
        - Clear frontal face images
        - Medium-to-large faces
        - Good lighting conditions
        - Standard medical and cloth masks
        """)

    with col_b:
        st.markdown("### Known Limitations")
        st.markdown("""
        - Side-profile faces may be harder to detect
        - Very small faces may be skipped
        - Heavy blur and extreme occlusion can reduce confidence
        - Haar cascade face detection can miss some hard cases
        """)

        st.markdown("### Failure Cases")
        st.markdown("""
        - Severe motion blur
        - Very dark images
        - Strongly angled faces
        - Images where the face detector misses the face box
        """)

    st.markdown("### Recommended Input Conditions")
    st.markdown("""
    - Clear frontal face images
    - Good lighting conditions
    - Medium to high image quality
    - Minimal motion blur
    """)

    st.markdown("### Evaluation Note")
    st.markdown("""
    This system is optimized for strong binary mask classification performance on the provided dataset.
    Reported evaluation metrics are measured on a held-out test split, but real-world performance can still vary
    for side profiles, very small faces, extreme blur, or severe occlusion.
    """)