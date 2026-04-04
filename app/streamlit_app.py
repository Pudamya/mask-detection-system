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
body {
    background: radial-gradient(circle at top, #0f172a, #020617);
}

.main {
    padding-top: 1.2rem;
}

.hero-card {
    padding: 1.6rem;
    border-radius: 20px;
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1rem;
}

.metric-card {
    padding: 1rem;
    border-radius: 16px;
    background: linear-gradient(145deg, rgba(30,41,59,0.7), rgba(15,23,42,0.7));
    border: 1px solid rgba(255,255,255,0.05);
    text-align: center;
    transition: 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
}

.section-card {
    padding: 1rem;
    border-radius: 16px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 1rem;
}

.result-card {
    padding: 1.2rem 1.3rem;
    border-radius: 18px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 0.8rem;
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

.status-pill {
    display: inline-block;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 0.35rem;
}

img:hover {
    transform: scale(1.02);
    transition: 0.3s ease;
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
    certainty_level = float(max(with_mask_score, without_mask_score))

    labels = ["With Mask", "Without Mask", "Certainty Level"]
    values = [with_mask_score, without_mask_score, certainty_level]

    return labels, values


def get_result_theme(predicted_label):
    if predicted_label == "with_mask":
        return {
            "title": "Mask Detected",
            "color": "#22c55e",
            "bg": "rgba(34,197,94,0.12)",
            "border": "rgba(34,197,94,0.35)",
            "pill_bg": "rgba(34,197,94,0.18)",
            "pill_text": "#86efac",
            "insight": "The face appears to be covered with a mask."
        }
    if predicted_label == "without_mask":
        return {
            "title": "No Mask Detected",
            "color": "#ef4444",
            "bg": "rgba(239,68,68,0.12)",
            "border": "rgba(239,68,68,0.35)",
            "pill_bg": "rgba(239,68,68,0.18)",
            "pill_text": "#fca5a5",
            "insight": "The face appears to be visible without a mask."
        }
    return {
        "title": "Uncertain Result",
        "color": "#eab308",
        "bg": "rgba(234,179,8,0.12)",
        "border": "rgba(234,179,8,0.35)",
        "pill_bg": "rgba(234,179,8,0.18)",
        "pill_text": "#fde68a",
        "insight": "The system detected a face-like region, but confidence is moderate."
    }


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
    st.markdown("### Model Info")
    st.markdown("""
    - Custom CNN
    - Input: 128×128
    - Classes: 2
    """)

    st.markdown("### Performance")
    if metrics:
        st.markdown(f"- Accuracy: {metrics.get('accuracy', 0.0) * 100:.2f}%")
        st.markdown(f"- F1 Score: {metrics.get('f1_weighted', 0.0):.4f}")
    else:
        st.markdown("- Accuracy: Not available yet")
        st.markdown("- F1 Score: Not available yet")

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
            with st.spinner("Analyzing image..."):
                try:
                    annotated, results = inferencer.detect_images(temp_path)
                    st.image(annotated, width="stretch")

                except Exception as e:
                    st.error(f"Error during detection: {e}")
                    results = []

        if results:
            st.markdown("---")
            st.subheader("Result Summary")

            for i, r in enumerate(results):
                predicted_label = r.get('class', 'unknown')
                raw_label = r.get('raw_class', predicted_label)
                confidence = float(r.get('confidence', 0.0))
                probabilities = r.get('probabilities', None)
                blur_detected = r.get('blur_detected', False)
                blur_score = float(r.get('blur_score', 0.0))
                blur_text = "Yes" if blur_detected else "No"
                analysis_mode = "Adaptive image analysis" if r.get("used_full_image_fallback", False) else "Face crop analysis"

                theme = get_result_theme(predicted_label)

                st.markdown(f"""
                <div class="result-card" style="
                    background:{theme['bg']};
                    border:1px solid {theme['border']};
                ">
                    <div style="font-size:1.35rem; font-weight:700; color:{theme['color']};">
                        Face {i+1}: {theme['title']}
                    </div>
                    <div style="margin-top:0.45rem; font-size:1rem;">
                        Confidence: <strong>{confidence:.1f}%</strong>
                    </div>
                    <div class="status-pill" style="
                        background:{theme['pill_bg']};
                        color:{theme['pill_text']};
                    ">
                    </div>
                    
                </div>
                """, unsafe_allow_html=True)

                if probabilities is None:
                    probs = np.array([0.5, 0.5], dtype=float)
                else:
                    probs = np.array(probabilities, dtype=float)

                top3_labels, top3_values = get_top3_display(probs)

                st.markdown("#### Confidence Analysis")
                fig, ax = plt.subplots(figsize=(7, 2.7))
                fig.patch.set_facecolor('#0e1117')
                ax.set_facecolor('#0e1117')

                bar_colors = ["#22c55e", "#ef4444", "#38bdf8"]

                bars = ax.barh(
                    top3_labels,
                    top3_values,
                    color=bar_colors,
                    height=0.4,
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
                ax.set_title(f"Face {i+1} Confidence Analysis", color='white', fontsize=12)
                ax.tick_params(colors='white')
                ax.spines['bottom'].set_color('#444')
                ax.spines['left'].set_color('#444')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.xaxis.label.set_color('white')

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.caption("This view shows both class probabilities and a certainty level derived from the strongest class score.")

                with st.expander(f"View technical details for Face {i+1}"):
                    st.markdown(f"""
                    <div class="section-card">
                        <strong>Analysis Mode:</strong> {analysis_mode}<br>
                        <strong>Model Output:</strong> {raw_label}<br>
                        <strong>Final Decision:</strong> {predicted_label}<br>
                        <strong>Confidence:</strong> {confidence:.2f}%<br>
                        <strong>Blur Detected:</strong> {blur_text}<br>
                        <strong>Blur Score:</strong> {blur_score:.2f}
                    </div>
                    """, unsafe_allow_html=True)

        elif uploaded_file is not None:
            st.markdown("---")
            st.error("No human face was detected in the uploaded image. Please upload a clear image containing a visible human face.")

        if os.path.exists(temp_path):
            os.remove(temp_path)

    else:
        st.info("Upload an image above to get started.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### How it works")
            st.markdown(
                "1. Upload a `.jpg` or `.png` image\n"
                "2. Detect and analyze visible face regions\n"
                "3. Run custom CNN classification\n"
                "4. Display the final decision with confidence"
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
                "- Heavy occlusion may affect confidence"
            )

with tab2:
    st.subheader("Training and Evaluation Artifacts")

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
        - Face detection can still miss some hard cases
        """)

        st.markdown("### Failure Cases")
        st.markdown("""
        - Severe motion blur
        - Very dark images
        - Strongly angled faces
        - Images where clear facial structure is not visible
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