import streamlit as st
import torch
import os
import sys
import matplotlib.pyplot as plt


sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import ModelDevelopment
from inference import BasicInference

#Config
st.set_page_config(
    page_title="Face Mask Detection System | IWMI Assessment",
    page_icon="FM",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Load Model 
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = ModelDevelopment(num_classes=2)
    model.load_state_dict(torch.load(
        os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pth'),
        map_location=device
    ))
    model.to(device)
    model.eval()
    inferencer = BasicInference(
        model, device, img_size=128,
        classes=['with_mask', 'without_mask']
    )
    return inferencer, device

# Sidebar
with st.sidebar:
    st.image("https://www.iwmi.cgiar.org/wp-content/uploads/2020/09/IWMI-logo.png", width=180)
    st.title("Model Info")

    st.markdown("**Architecture:** Custom CNN")
    st.markdown("**Framework:** PyTorch")
    st.markdown("**Input Size:** 128 × 128 × 3")
    st.markdown("**Classes:** `with_mask` · `without_mask`")

    st.markdown("---")
    st.markdown("**Network Layers:**")
    st.markdown("""
    - Conv Block 1 → 32 filters
    - Conv Block 2 → 64 filters
    - Conv Block 3 → 128 filters
    - Conv Block 4 → 256 filters
    - FC Layer → 512 units
    - FC Layer → 128 units
    - Output → 2 classes
    """)
    st.markdown("**Total Parameters:** ~8.8M")
    st.markdown("**Optimizer:** Adam + ReduceLROnPlateau")
    st.markdown("**Regularization:** BatchNorm + Dropout")

    st.markdown("---")

    # Show accuracy if results file exists
    results_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'confusion_matrix.png')
    if os.path.exists(results_path):
        st.markdown("**Confusion Matrix:**")
        st.image(results_path)

    st.markdown("---")
    st.markdown("*IWMI Data Science Assessment*")

# Main Title 
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

# File Upload 
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=['jpg', 'jpeg', 'png'],
    help="Upload a clear frontal face image for best results"
)

if uploaded_file is not None:
    inferencer, device = load_model()

    # Save temp file
    temp_path = "temp_upload.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(temp_path, use_column_width=True)

    with col2:
        st.subheader("Detection Result")
        with st.spinner("Detecting faces and classifying..."):
            try:
                annotated, results = inferencer.detect_images(temp_path)
                st.image(annotated, use_column_width=True)
            except Exception as e:
                st.error(f"Error during detection: {e}")
                results = []

    # Prediction Details
    if results:
        st.markdown("---")
        st.subheader("Prediction Details")

        for i, r in enumerate(results):
            with st.container():
                # Status badge
                if r['class'] == 'with_mask':
                    st.success(f"Face {i+1}: **WITH MASK** — Confidence: `{r['confidence']:.1f}%`")
                else:
                    st.error(f"Face {i+1}: **WITHOUT MASK** — Confidence: `{r['confidence']:.1f}%`")

                # Clean bar chart
                fig, ax = plt.subplots(figsize=(7, 2.2))
                fig.patch.set_facecolor('#0e1117')
                ax.set_facecolor('#0e1117')

                classes     = ['With Mask', 'Without Mask']
                probs       = r['probabilities'] * 100
                bar_colors  = ['#2ecc71', '#e74c3c']

                bars = ax.barh(classes, probs, color=bar_colors,
                               height=0.5, edgecolor='none')

                # Value labels on bars
                for bar, val in zip(bars, probs):
                    ax.text(
                        min(val + 1.5, 95),
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%",
                        va='center', ha='left',
                        color='white', fontsize=11, fontweight='bold'
                    )

                ax.set_xlim(0, 100)
                ax.set_xlabel("Confidence (%)", color='white')
                ax.set_title(f"Face {i+1} — Class Probabilities",
                             color='white', fontsize=12)
                ax.tick_params(colors='white')
                ax.spines['bottom'].set_color('#444')
                ax.spines['left'].set_color('#444')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.xaxis.label.set_color('white')

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    elif len(results) == 0 and uploaded_file is not None:
        st.markdown("---")
        st.warning("No faces were detected in this image. Try a clearer frontal face photo.")

    # Cleanup temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)

else:
    # Show placeholder when no image uploaded
    st.info("Upload an image above to get started.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### How it works")
        st.markdown("1. Upload a `.jpg` or `.png` image\n2. Haar Cascade detects faces\n3. CNN classifies each face\n4. Results shown with confidence")
    with col2:
        st.markdown("#### Best Results With")
        st.markdown("- Clear frontal face photos\n- Good lighting\n- Face clearly visible\n- Single or multiple people")
    with col3:
        st.markdown("#### Limitations")
        st.markdown("- Side profile faces may be missed\n- Very small faces may not detect\n- Heavy occlusion may confuse model")