"""
Professional Streamlit interface for the lung segmentation API.
"""

import base64
import io

import numpy as np
import pydicom
import requests
import streamlit as st
from PIL import Image

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="PulmoVision | Lung Segmentation",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

    :root {
        --navy: #102a43;
        --navy-deep: #0b1f33;
        --teal: #0f8b8d;
        --teal-soft: #e6f6f5;
        --slate: #52606d;
        --line: #d9e2ec;
        --surface: #ffffff;
        --background: #f5f8fa;
    }

    .stApp { background: var(--background); color: var(--navy); font-family: 'DM Sans', sans-serif; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: var(--navy-deep); border-right: 0; }
    [data-testid="stSidebar"] * { color: #e6eef5 !important; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.16); }
    h1, h2, h3 { font-family: 'Manrope', sans-serif; color: var(--navy); letter-spacing: -.03em; }
    [data-testid="stSidebar"] h2 { color: white !important; }
    .hero { padding: 2.2rem 0 1.6rem; border-bottom: 1px solid var(--line); margin-bottom: 1.8rem; }
    .eyebrow { color: var(--teal); font-size: .78rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; margin-bottom: .55rem; }
    .hero h1 { font-size: clamp(2rem, 4vw, 3.2rem); margin: 0 0 .55rem; }
    .hero p { color: var(--slate); font-size: 1.04rem; margin: 0; max-width: 720px; }
    .section-label { color: var(--slate); font-size: .8rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; margin: 0 0 .65rem; }
    .status { padding: .85rem 1rem; border-radius: .55rem; background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.14); font-size: .9rem; }
    .upload-card, [data-testid="stImage"], [data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--line); border-radius: .7rem; }
    [data-testid="stMetric"] { padding: 1rem 1.1rem; }
    [data-testid="stMetricLabel"] { color: var(--slate); }
    [data-testid="stMetricValue"] { color: var(--navy); font-family: 'Manrope', sans-serif; }
    .stButton > button[kind="primary"] { background: var(--teal); border: 0; border-radius: .5rem; font-weight: 700; padding: .65rem 1.3rem; color: #000000; }
    .stButton > button[kind="primary"]:hover { background: #0b7072; color: #000000; }
    .stButton > button[kind="primary"] p { color: #000000; }
    [data-testid="stFileUploader"] { background: var(--surface); border: 1px dashed #9fb3c8; border-radius: .7rem; padding: .35rem; }
    .footer-note { color: var(--slate); font-size: .82rem; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--line); }

    /* Keeps the two result columns aligned at the same vertical level,
       even though one has a tab bar and the other doesn't. */
    .column-header-spacer { height: 2.75rem; display: flex; align-items: center; }
    </style>
    """,
    unsafe_allow_html=True,
)


def dicom_to_png_bytes(file_bytes):
    """Convert a DICOM file (.dcm) to 8-bit grayscale PNG bytes, so the rest of the
    pipeline (preview + API call) can treat it exactly like a normal PNG/JPEG."""
    ds = pydicom.dcmread(io.BytesIO(file_bytes))
    img = ds.pixel_array.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255.0
    img_uint8 = img.astype(np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(img_uint8).save(buffer, format="PNG")
    return buffer.getvalue()


with st.sidebar:
    st.markdown("## PulmoVision")
    st.caption("AI-powered chest imaging analysis platform")
    st.divider()
    st.markdown("### About")
    st.write(
        "Upload a chest X-ray to obtain a clear segmentation of the "
        "right and left lungs "
    )
    st.markdown("### Service status")
    api_available = False
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        if health.get("model_loaded", False):
            api_available = True
            st.markdown(
                f'<div class="status">● API operational<br><small>Device: {health.get("device", "?")}</small></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="status">● Model not loaded<br><small>{health.get("load_error", "Unknown error")}</small></div>',
                unsafe_allow_html=True,
            )
    except requests.exceptions.RequestException:
        st.markdown('<div class="status">● API unavailable<br><small>Please check that the server is running.</small></div>', unsafe_allow_html=True)

st.markdown(
    '<div class="hero"><div class="eyebrow">AI-assisted medical analysis</div>'
    '<h1>Lung Segmentation</h1>'
    '<p>Automatic Detection and Delineation of Lungs in Chest X-Rays.</p></div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">1 · Upload an image and run the analysis</div>', unsafe_allow_html=True)

upload_col, action_col = st.columns([3, 1], gap="large", vertical_alignment="top")
with upload_col:
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray in PNG, JPEG, or DICOM format",
        type=["png", "jpg", "jpeg", "dcm"],
        label_visibility="visible",
    )
with action_col:
    st.markdown('<div style="height: 1.9rem;"></div>', unsafe_allow_html=True)
    launch_segmentation = st.button(
        "Run segmentation",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None or not api_available,
    )
    if uploaded_file is None:
        st.caption("Upload an image to enable analysis.")
    elif not api_available:
        st.caption("The API is currently unavailable.")

if uploaded_file is not None:
    is_dicom = uploaded_file.name.lower().endswith(".dcm")

    # Convert once, up front -- both the preview and the API call reuse these PNG bytes.
    if is_dicom:
        try:
            image_bytes = dicom_to_png_bytes(uploaded_file.getvalue())
            image_filename = uploaded_file.name.rsplit(".", 1)[0] + ".png"
            image_mimetype = "image/png"
        except Exception as e:
            st.error(f"Unable to read this DICOM file: {e}")
            st.stop()
    else:
        image_bytes = uploaded_file.getvalue()
        image_filename = uploaded_file.name
        image_mimetype = uploaded_file.type

    col_input, col_output = st.columns(2, gap="large")

    with col_input:
        st.markdown('<div class="section-label">Source image</div>', unsafe_allow_html=True)
        # Empty spacer matching the height of the tab bar on the right column,
        # so both images start at exactly the same vertical position.
        st.markdown('<div class="column-header-spacer"></div>', unsafe_allow_html=True)
        st.image(image_bytes, use_container_width=True)
        if is_dicom:
            st.caption("DICOM file converted to grayscale for preview and analysis.")

    if launch_segmentation:
        with st.spinner("Analyzing image…"):
            try:
                files = {"file": (image_filename, image_bytes, image_mimetype)}
                response = requests.post(
                    f"{API_URL}/predict",
                    files=files,
                    timeout=30,
                )
                response.raise_for_status()
                result = response.json()
            except requests.exceptions.HTTPError as e:
                detail = ""
                try:
                    detail = response.json().get("detail", "")
                except ValueError:
                    pass
                st.error(f"API error ({response.status_code}): {detail or e}")
                st.stop()
            except requests.exceptions.RequestException as e:
                st.error(f"Error calling the API: {e}")
                st.stop()

        overlay_bytes = base64.b64decode(result["overlay_png_base64"])
        mask_bytes = base64.b64decode(result["mask_png_base64"])

        with col_output:
            st.markdown('<div class="section-label">Analysis result</div>', unsafe_allow_html=True)
            tab_overlay, tab_mask = st.tabs(["Overlay", "Segmented mask"])
            with tab_overlay:
                st.image(Image.open(io.BytesIO(overlay_bytes)), use_container_width=True)
                st.caption("Red: right lung · Green: left lung")
            with tab_mask:
                st.image(Image.open(io.BytesIO(mask_bytes)), use_container_width=True)

        st.markdown('<div class="section-label">Confidence indicators</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Average confidence", f"{result['mean_confidence'] * 100:.1f} %")

        areas = result.get("class_areas", {})
        right = areas.get("Right lung", {})
        left = areas.get("Left lung", {})
        c2.metric("Right lung", f"{right.get('ratio', 0) * 100:.1f} %")
        c3.metric("Left lung", f"{left.get('ratio', 0) * 100:.1f} %")

        if right.get("pixels", 0) == 0 or left.get("pixels", 0) == 0:
            st.warning("One of the two lungs was not detected — check the image quality and try again.")

        with st.expander("Show technical data (JSON)"):
            st.json(result)
    else:
        with col_output:
            st.markdown('<div class="section-label">Analysis result</div>', unsafe_allow_html=True)
            st.markdown('<div class="column-header-spacer"></div>', unsafe_allow_html=True)
            st.info("Run the segmentation to see the result here.")
else:
    st.info("Upload an image to begin the analysis.")

st.markdown('<div class="footer-note">Analysis support tool — results must be interpreted by a healthcare professional.</div>', unsafe_allow_html=True)