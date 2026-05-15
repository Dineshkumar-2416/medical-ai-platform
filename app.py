"""
Streamlit dashboard — Medical AI Chest X-ray Analysis Platform

Run with:
    streamlit run app.py
"""
import io
import base64
import time

import requests
import numpy as np
from PIL import Image
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
API_URL     = "http://localhost:8000"
PAGE_TITLE  = "Chest X-ray AI Analysis"

st.set_page_config(
    page_title = PAGE_TITLE,
    page_icon  = "🫁",
    layout     = "wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .prediction-box {
        padding: 20px; border-radius: 12px;
        text-align: center; font-size: 28px; font-weight: bold;
        margin: 10px 0;
    }
    .pneumonia { background: #ff4b4b22; border: 2px solid #ff4b4b; color: #ff4b4b; }
    .normal    { background: #21c35422; border: 2px solid #21c354; color: #21c354; }
    .warning   { background: #ffa50022; border: 2px solid #ffa500; color: #ffa500; }
    .metric-label { font-size: 13px; color: #888; margin-bottom: 2px; }
    .metric-value { font-size: 22px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────
def check_api_health() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def call_predict(image_bytes: bytes, filename: str) -> dict | None:
    try:
        files    = {"file": (filename, image_bytes, "image/jpeg")}
        response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
        return response.json()
    except Exception as e:
        st.error(f"API call failed: {e}")
        return None


def b64_to_image(b64_str: str) -> Image.Image:
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes))


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫁 Medical AI Platform")
    st.markdown("**Chest X-ray Pneumonia Detection**")
    st.divider()

    health = check_api_health()
    if health:
        st.success("API Server: Online")
        st.caption(f"Device: {health['device'].upper()}")
        st.caption(f"Model loaded: {'Yes' if health['model_loaded'] else 'No'}")
        st.caption(f"Uptime: {int(health['uptime_sec'])}s")
    else:
        st.error("API Server: Offline")
        st.warning("Start the server:\n```\nuvicorn src.api.main:app --port 8000\n```")

    st.divider()
    st.markdown("**Model Performance**")
    st.metric("Test AUC",         "0.9620")
    st.metric("Sensitivity",      "99.49%")
    st.metric("Specificity",      "64.10%")
    st.metric("Training Epochs",  "18 (best)")
    st.divider()

    st.markdown("**About GradCAM**")
    st.caption(
        "GradCAM highlights which regions of the X-ray "
        "caused the model's prediction. "
        "Red/yellow = high activation. Blue = low activation."
    )
    st.divider()
    st.caption("⚠️ For research purposes only. Not a substitute for clinical diagnosis.")


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🫁 Chest X-ray Pneumonia Detection")
st.markdown("Upload a chest X-ray image to get an AI-assisted diagnosis with explainability.")
st.divider()

uploaded_file = st.file_uploader(
    "Upload Chest X-ray",
    type=["jpg", "jpeg", "png"],
    help="JPEG or PNG, max 10MB",
)

if uploaded_file is not None:
    image_bytes = uploaded_file.read()

    # Show analyze button
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        analyze = st.button("Analyze X-ray", type="primary", use_container_width=True)

    if analyze:
        if not health:
            st.error("Cannot analyze — API server is offline.")
            st.stop()

        with st.spinner("Running inference on GPU..."):
            t0     = time.time()
            result = call_predict(image_bytes, uploaded_file.name)
            elapsed = time.time() - t0

        if result is None:
            st.error("Inference failed. Check the API server logs.")
            st.stop()

        # ── Layout: 3 columns ─────────────────────────────────────────
        st.markdown(f"*Analysis completed in {elapsed:.2f}s*")
        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Original X-ray")
            original = Image.open(io.BytesIO(image_bytes))
            st.image(original, use_container_width=True)

        with col2:
            st.markdown("#### GradCAM Heatmap")
            overlay_img = b64_to_image(result["gradcam_image"])
            st.image(overlay_img, use_container_width=True)
            st.caption("Red = regions that drove the prediction")

        with col3:
            st.markdown("#### Diagnosis")

            pred = result["prediction"]
            conf = result["confidence"]

            # Prediction box
            css_class = "pneumonia" if pred == "PNEUMONIA" else "normal"
            icon      = "🔴" if pred == "PNEUMONIA" else "🟢"
            st.markdown(
                f'<div class="prediction-box {css_class}">{icon} {pred}</div>',
                unsafe_allow_html=True,
            )

            # Low confidence warning
            if result["low_confidence"]:
                st.markdown(
                    f'<div class="prediction-box warning">⚠️ LOW CONFIDENCE</div>',
                    unsafe_allow_html=True,
                )
                st.warning(result["message"])

            st.markdown("---")

            # Confidence bar
            st.markdown("**Confidence**")
            st.progress(conf, text=f"{conf*100:.1f}%")

            st.markdown("---")

            # Probability breakdown
            st.markdown("**Class Probabilities**")
            p_normal    = result["prob_normal"]
            p_pneumonia = result["prob_pneumonia"]

            st.markdown(f"🟢 Normal:    **{p_normal*100:.2f}%**")
            st.progress(p_normal)

            st.markdown(f"🔴 Pneumonia: **{p_pneumonia*100:.2f}%**")
            st.progress(p_pneumonia)

            st.markdown("---")

            # Clinical note
            if pred == "PNEUMONIA":
                st.error(
                    "**Finding:** Radiographic opacity consistent with pneumonia detected. "
                    "Clinical correlation and follow-up recommended."
                )
            else:
                st.success(
                    "**Finding:** No acute cardiopulmonary process detected. "
                    "Lungs appear clear."
                )

    else:
        # Show preview while waiting for analyze click
        st.image(Image.open(io.BytesIO(image_bytes)), width=300, caption="Uploaded X-ray")

else:
    # Empty state
    st.info("Upload a chest X-ray JPEG or PNG to begin analysis.")
    st.markdown("**Sample images available in:** `data/raw/chest_xray/test/`")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **What this model detects:**
        - ✅ Normal (healthy) lungs
        - ✅ Bacterial pneumonia
        - ✅ Viral pneumonia
        """)
    with col2:
        st.markdown("""
        **What you get back:**
        - Prediction + confidence score
        - GradCAM explainability heatmap
        - Low-confidence warnings
        """)
