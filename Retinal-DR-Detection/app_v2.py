from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError
from model import load_model, predict_image

# RetinaAI - navigation-fixed Streamlit application.
# Keep this file in the same folder as model.py and the models/ directory.

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
HISTORY_FILE = DATA_DIR / "history.json"
HISTORY_IMAGES_DIR = DATA_DIR / "history_images"
SAMPLE_IMAGES_DIR = BASE_DIR / "sample_images"
MODEL_FILE = BASE_DIR / "models" / "best_vit_retinal_improved.pth"

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
DISPLAY_NAMES = {
    "No DR": "No DR",
    "Mild": "Mild DR",
    "Moderate": "Moderate DR",
    "Severe": "Severe DR",
    "Proliferative DR": "Proliferative DR",
}
PAGES = ["Home", "Upload & Predict", "History", "About"]
NAV_LABELS = {
    "Home": "⌂   Home",
    "Upload & Predict": "⇧   Upload & Predict",
    "History": "◷   History",
    "About": "ⓘ   About",
}
MAX_IMAGE_BYTES = 10 * 1024 * 1024

st.set_page_config(
    page_title="RetinaAI | Retinal Disease Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
      --bg:#06162c; --sidebar:#071b36; --panel:#0b2342;
      --line:#1b426d; --blue:#1688ff; --cyan:#51d7ff;
      --text:#f0f6ff; --muted:#a9c0de;
    }
    html, body, [data-testid="stAppViewContainer"], .stApp {
      background:var(--bg) !important; color:var(--text) !important;
    }
    [data-testid="stHeader"], [data-testid="stToolbar"],
    [data-testid="stStatusWidget"], [data-testid="stDecoration"] {
      display:none !important;
    }

    /* Keep the desktop navigation panel visible and give it a stable width. */
    section[data-testid="stSidebar"] {
      display:block !important; visibility:visible !important;
      transform:translateX(0) !important; width:250px !important;
      min-width:250px !important; max-width:250px !important;
      background:var(--sidebar) !important;
      border-right:1px solid #133458 !important;
    }
    section[data-testid="stSidebar"] > div {
      width:250px !important; min-width:250px !important;
      background:var(--sidebar) !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
      background:var(--sidebar) !important;
    }
    /* Hide collapse controls so the navigation cannot accidentally disappear. */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"] {
      display:none !important;
    }

    .block-container {max-width:1500px;padding-top:.8rem;padding-bottom:2rem;}
    h1,h2,h3,h4 {color:var(--text)!important;letter-spacing:-.02em;}
    p,li,label {color:#d8e7fa;}
    [data-testid="stCaptionContainer"] p {color:var(--muted)!important;}
    .ra-topbar {display:flex;align-items:center;justify-content:space-between;
      padding:.1rem .1rem .7rem;margin-bottom:1rem;border-bottom:1px solid #15385f;}
    .ra-brand {font-size:1.35rem;font-weight:850;color:white;display:flex;gap:.65rem;align-items:center;}
    .ra-eye {background:#0878ed;border:1px solid #62b5ff;border-radius:50%;
      width:2rem;height:2rem;display:inline-flex;align-items:center;justify-content:center;}
    .ra-user {color:#d5e4f6;font-size:.9rem;}
    .hero {display:flex;align-items:center;justify-content:space-between;gap:2rem;
      min-height:260px;padding:1.8rem;border-radius:15px;
      background:linear-gradient(115deg,#0a2344,#081d39 56%,#0d2b51);
      border:1px solid #17385e;margin-bottom:1.15rem;}
    .hero-copy {flex:1;min-width:0;}
    .eyebrow {color:var(--cyan);font-weight:750;font-size:.85rem;margin-bottom:.55rem;}
    .hero-title {font-weight:850;font-size:2.2rem;line-height:1.13;margin:0 0 .75rem;color:#f3f7ff;}
    .hero-title span {color:#55aaff;}
    .hero-desc {color:#c0d3ed;font-size:.95rem;line-height:1.55;max-width:620px;}
    .hero-eye {width:220px;min-width:180px;height:220px;border-radius:50%;object-fit:cover;
      border:3px solid #193e66;box-shadow:0 0 0 12px #1c6fc11a,0 0 34px #1f84ff22;}
    .feature-card,.panel {background:var(--panel);border:1px solid #173a62;border-radius:11px;padding:1rem;}
    .feature-card {height:100%;min-height:125px;}
    .feature-icon {color:#56aaff;font-size:1.4rem;margin-bottom:.4rem;}
    .feature-title {font-weight:800;color:#f1f6ff;margin-bottom:.35rem;}
    .feature-copy,.muted,.status-note {font-size:.82rem;line-height:1.45;color:#afc5e1;}
    .section-title {font-size:1.15rem;font-weight:800;margin:.3rem 0 .8rem;color:#f1f6ff;}
    .pill {display:inline-block;padding:.4rem .8rem;border-radius:999px;
      background:#ffd166;color:#1b2433;font-weight:850;}
    .prob-row {display:grid;grid-template-columns:135px 1fr 52px;gap:10px;
      align-items:center;margin:.62rem 0;color:#dbe8fb;font-size:.8rem;}
    .prob-track {height:9px;border-radius:99px;background:#1b385d;overflow:hidden;}
    .prob-fill {height:100%;border-radius:99px;background:#1688ff;}
    .prob-fill.selected {background:#ffca55;}
    .prob-value {text-align:right;color:#eef5ff;}
    .drop-hint {text-align:center;background:#0a2040;border:1px dashed #245783;
      padding:1.3rem;border-radius:10px;}
    div.stButton>button {border-radius:7px;border:1px solid #235383;
      background:#0c2e55;color:#eef6ff;font-weight:700;}
    div.stButton>button:hover {background:#104078;border-color:#1688ff;color:white;}
    div.stButton>button[kind="primary"] {background:#1688ff;border:0;color:white;font-weight:800;}
    [data-testid="stFileUploaderDropzone"] {background:#0a2040;border:1px dashed #255685;border-radius:10px;}
    [data-testid="stProgressBar"]>div>div {background:#1688ff;}
    [data-testid="stImage"] img {border:1px solid #1b426d;border-radius:9px;}
    hr {border-color:#1a3b60;}
    footer {visibility:hidden;}

    /* Streamlit radio is the actual navigation control, not decorative text. */
    [data-testid="stSidebar"] div[data-testid="stRadio"]>label {display:none !important;}
    [data-testid="stSidebar"] div[role="radiogroup"] {gap:.45rem;}
    [data-testid="stSidebar"] div[role="radiogroup"] label {
      display:flex;align-items:center;padding:.62rem .72rem;
      border:1px solid transparent;border-left:3px solid transparent;
      border-radius:7px;background:transparent;min-height:2.8rem;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
      background:#0a2546;border-color:#163a60;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
      background:#0d2d52;border-color:#174574;border-left-color:#2696ff;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label p {
      color:#d4e2f3;font-size:.92rem;font-weight:600;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
      color:#fff;font-weight:800;
    }
    @media(max-width:850px) {
      .hero {display:block;padding:1.3rem;}
      .hero-eye {width:160px;height:160px;margin-top:1rem;}
      .hero-title {font-size:1.7rem;}
      .prob-row {grid-template-columns:100px 1fr 48px;gap:6px;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()


def ensure_storage():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def read_history():
    ensure_storage()
    if not HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def write_history(records):
    ensure_storage()
    HISTORY_FILE.write_text(json.dumps(records, indent=2), encoding="utf-8")


def image_to_rgb(image_bytes):
    with Image.open(BytesIO(image_bytes)) as source:
        source.load()
        return source.convert("RGB")


def image_hash(image_bytes):
    return hashlib.sha256(image_bytes).hexdigest()[:20]


def save_history_record(image, prediction, filename):
    ensure_storage()
    record_id = uuid.uuid4().hex[:12]
    image_name = f"{record_id}.jpg"
    image.save(HISTORY_IMAGES_DIR / image_name, format="JPEG", quality=91)
    records = read_history()
    records.insert(0, {
        "id": record_id,
        "timestamp": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S"),
        "original_name": filename,
        "image_file": image_name,
        "prediction": str(prediction["prediction"]),
        "confidence": float(prediction["confidence"]),
        "probabilities": {str(k): float(v) for k, v in prediction["probabilities"].items()},
    })
    old_records = records[100:]
    write_history(records[:100])
    for old in old_records:
        old_path = HISTORY_IMAGES_DIR / str(old.get("image_file", ""))
        if old_path.is_file():
            try:
                old_path.unlink()
            except OSError:
                pass


def show_probabilities(probabilities, predicted_class=None):
    st.markdown("<div class='section-title'>Class Probabilities</div>", unsafe_allow_html=True)
    for class_name in CLASS_NAMES:
        value = max(0.0, min(1.0, float(probabilities.get(class_name, 0.0))))
        selected = " selected" if class_name == predicted_class else ""
        label = DISPLAY_NAMES.get(class_name, class_name)
        st.markdown(
            f"<div class='prob-row'><div>{label}</div>"
            f"<div class='prob-track'><div class='prob-fill{selected}' style='width:{value*100:.2f}%'></div></div>"
            f"<div class='prob-value'>{value*100:.1f}%</div></div>",
            unsafe_allow_html=True,
        )


def render_topbar():
    st.markdown(
        "<div class='ra-topbar'><div class='ra-brand'><span class='ra-eye'>👁</span> RetinaAI</div>"
        "<div class='ra-user'>◉ &nbsp; Student</div></div>",
        unsafe_allow_html=True,
    )


def render_sidebar():
    # Session state keeps the selected page across Streamlit reruns.
    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = "Home"

    with st.sidebar:
        st.markdown(
            "<div style='padding:.5rem .25rem 1.1rem;font-size:1.05rem;"
            "font-weight:800;color:#55cfff'>RETINAAI <span style='color:#8ba9ca;"
            "font-size:.72rem;font-weight:500'> · NAVIGATION</span></div>",
            unsafe_allow_html=True,
        )
        selected_page = st.radio(
            "Navigation",
            options=PAGES,
            format_func=lambda page: NAV_LABELS[page],
            key="nav_page",
            label_visibility="collapsed",
        )
        st.markdown(
            "<div style='height:1rem'></div><hr>"
            "<div class='status-note'>AI Retinal Screening<br>Vision Transformer · 5 classes</div>",
            unsafe_allow_html=True,
        )
    return selected_page


def go_to_upload():
    st.session_state["nav_page"] = "Upload & Predict"


def render_home():
    sample = SAMPLE_IMAGES_DIR / "Mild_DR.png"
    hero_image = ""
    if sample.is_file():
        import base64
        try:
            encoded = base64.b64encode(sample.read_bytes()).decode("ascii")
            hero_image = f"<img class='hero-eye' src='data:image/png;base64,{encoded}' alt='Retinal fundus image'>"
        except OSError:
            pass
    if not hero_image:
        hero_image = "<div class='hero-eye' style='display:flex;align-items:center;justify-content:center;font-size:5rem;color:#55aaff'>◉</div>"

    st.markdown(
        "<div class='hero'><div class='hero-copy'><div class='eyebrow'>AI for Better Vision</div>"
        "<div class='hero-title'>Retinal Disease<br>Detection using<br><span>Vision Transformer</span></div>"
        "<div class='hero-desc'>Our AI-assisted research system classifies retinal fundus images "
        "into five diabetic retinopathy severity categories. It provides model predictions to "
        "support academic evaluation and learning.</div></div>" + hero_image + "</div>",
        unsafe_allow_html=True,
    )
    st.button("Get Started  →", key="home_get_started", type="primary", on_click=go_to_upload)
    st.write("")
    cards = [
        ("⚙", "AI Classification", "Powered by your trained Vision Transformer (ViT)."),
        ("✣", "Five Disease Classes", "No DR, Mild, Moderate, Severe and Proliferative DR."),
        ("◴", "Fast Analysis", "Run model inference when an image is submitted."),
        ("♥", "Prediction History", "Review previous predictions saved on this device."),
    ]
    columns = st.columns(4, gap="small")
    for column, (icon, title, body) in zip(columns, cards):
        with column:
            st.markdown(
                f"<div class='feature-card'><div class='feature-icon'>{icon}</div>"
                f"<div class='feature-title'>{title}</div><div class='feature-copy'>{body}</div></div>",
                unsafe_allow_html=True,
            )
    st.write("")
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.markdown("<div class='section-title'>How it works</div>", unsafe_allow_html=True)
        st.markdown("1. Upload a retinal fundus image.\n2. The image is preprocessed.\n"
                    "3. The trained ViT predicts one of five classes.\n4. Review probabilities and history.")
    with right:
        st.markdown("<div class='section-title'>Supported classes</div>", unsafe_allow_html=True)
        st.markdown("<div class='panel'>No DR &nbsp; • &nbsp; Mild DR &nbsp; • &nbsp; Moderate DR"
                    "<br><br>Severe DR &nbsp; • &nbsp; Proliferative DR</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='panel status-note'>Research prototype only. This model is not clinically "
        "validated and is not a substitute for professional medical advice.</div>",
        unsafe_allow_html=True,
    )


def show_result_card(result, image, filename):
    prediction = str(result.get("prediction", "Unknown"))
    display_prediction = DISPLAY_NAMES.get(prediction, prediction)
    confidence = max(0.0, min(1.0, float(result.get("confidence", 0.0))))
    st.markdown("<div class='section-title'>Prediction Results</div>", unsafe_allow_html=True)
    left, right = st.columns([1, 1.05], gap="small")
    with left:
        st.image(image, use_container_width=True)
        st.caption(f"Uploaded image · {filename}")
    with right:
        st.markdown(
            f"<div class='panel'><div class='muted'>Predicted Class</div>"
            f"<div style='margin:.55rem 0 .75rem'><span class='pill'>{display_prediction}</span></div>"
            f"<div class='muted'>Confidence Score</div>"
            f"<div style='font-size:1.7rem;font-weight:850;color:#f5f9ff'>{confidence*100:.2f}%</div>"
            "<div class='status-note' style='margin-top:.65rem'>This is a model prediction, not a confirmed diagnosis.</div></div>",
            unsafe_allow_html=True,
        )
    st.markdown("<div class='panel' style='margin-top:.7rem'>", unsafe_allow_html=True)
    show_probabilities(result.get("probabilities", {}), prediction)
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("Model probabilities may be incorrect. Consult a qualified eye-care professional for medical advice.")


def render_upload():
    st.markdown("<div class='section-title'>Upload Retinal Image</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drag and drop your retinal image here",
        type=["jpg", "jpeg", "png"],
        key="retina_file_uploader",
        help="Supported formats: JPG, JPEG, PNG. Maximum size: 10 MB.",
    )
    st.caption("Supported formats: JPG, JPEG, PNG · Maximum size: 10 MB")
    if uploaded is None:
        st.markdown(
            "<div class='drop-hint'><div style='font-size:1.8rem;color:#52aaff'>⇧</div>"
            "<b>No image selected yet</b><div class='status-note'>Upload a retinal fundus image to begin.</div></div>",
            unsafe_allow_html=True,
        )
        return

    image_bytes = uploaded.getvalue()
    if not image_bytes:
        st.warning("The selected file is empty.")
        return
    if len(image_bytes) > MAX_IMAGE_BYTES:
        st.error("This file is larger than the 10 MB limit. Please choose a smaller image.")
        return
    try:
        image = image_to_rgb(image_bytes)
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("This file could not be opened. Please upload a valid JPG or PNG image.")
        return

    fingerprint = image_hash(image_bytes)
    st.divider()
    if st.button("Analyze Image  →", type="primary", key=f"analyse_{fingerprint}", use_container_width=True):
        try:
            with st.spinner("Loading the Vision Transformer and analyzing the retinal image..."):
                model = get_model()
                result = predict_image(model, image)
                save_history_record(image, result, uploaded.name)
            st.session_state["last_analysis"] = {
                "fingerprint": fingerprint, "filename": uploaded.name, "result": result
            }
            st.success("Analysis completed.")
            st.rerun()
        except FileNotFoundError as error:
            st.error(f"Model checkpoint not found. Check: {MODEL_FILE}")
            st.code(str(error))
        except Exception as error:
            st.error("The image could not be analyzed. Check the terminal for details.")
            st.exception(error)

    last = st.session_state.get("last_analysis")
    if last and last.get("fingerprint") == fingerprint:
        show_result_card(last["result"], image, uploaded.name)
    else:
        left, right = st.columns([1, 1.25], gap="small")
        with left:
            st.markdown("<div class='section-title'>Uploaded Image</div>", unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.caption(uploaded.name)
        with right:
            st.markdown("<div class='section-title'>Prediction Results</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='panel' style='min-height:180px;text-align:center;padding-top:3rem'>"
                "<div style='font-size:2rem;color:#1688ff'>◌</div><b>Ready to Analyze</b>"
                "<div class='status-note'>Click Analyze Image to run your trained model.</div></div>",
                unsafe_allow_html=True,
            )


def render_history():
    st.markdown("<div class='section-title'>Prediction History</div>", unsafe_allow_html=True)
    records = read_history()
    if not records:
        st.info("Your prediction history will appear here after you analyze an image.")
        return
    st.caption("Entries are saved locally in this project's data folder.")
    header = st.columns([1.5, .85, 1.7, 1.0, .75])
    for col, label in zip(header, ["Date & Time", "Image", "Predicted Class", "Confidence", "Action"]):
        with col:
            st.markdown(f"<div class='feature-title'>{label}</div>", unsafe_allow_html=True)
    for record in records:
        columns = st.columns([1.5, .85, 1.7, 1.0, .75], vertical_alignment="center")
        with columns[0]:
            st.caption(str(record.get("timestamp", "")))
        with columns[1]:
            image_path = HISTORY_IMAGES_DIR / str(record.get("image_file", ""))
            if image_path.is_file():
                st.image(str(image_path), width=64)
            else:
                st.caption("N/A")
        with columns[2]:
            prediction = str(record.get("prediction", "Unknown"))
            st.markdown(f"**{DISPLAY_NAMES.get(prediction, prediction)}**")
            st.caption(str(record.get("original_name", "")))
        with columns[3]:
            st.markdown(f"**{float(record.get('confidence', 0.0))*100:.1f}%**")
        with columns[4]:
            if st.button("View", key=f"view_{record.get('id', uuid.uuid4().hex)}", use_container_width=True):
                st.session_state["view_history_id"] = record.get("id")
                st.rerun()
        st.markdown("<hr style='margin:.2rem 0 .65rem'>", unsafe_allow_html=True)

    selected_id = st.session_state.get("view_history_id")
    selected = next((item for item in records if item.get("id") == selected_id), None)
    if selected:
        st.markdown("### Selected prediction")
        left, right = st.columns([1, 1.2], gap="large")
        with left:
            image_path = HISTORY_IMAGES_DIR / str(selected.get("image_file", ""))
            if image_path.is_file():
                st.image(str(image_path), use_container_width=True)
        with right:
            prediction = str(selected.get("prediction", "Unknown"))
            st.markdown(f"<span class='pill'>{DISPLAY_NAMES.get(prediction, prediction)}</span>", unsafe_allow_html=True)
            st.metric("Model confidence", f"{float(selected.get('confidence', 0.0))*100:.2f}%")
            show_probabilities(selected.get("probabilities", {}), prediction)


def render_about():
    st.markdown("<div class='section-title'>About This Project</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='panel'><b>RetinaAI — Vision Transformer (ViT)</b><p class='status-note'>"
        "This academic project uses a trained Vision Transformer to classify retinal fundus images "
        "into five diabetic retinopathy severity categories: No DR, Mild DR, Moderate DR, Severe DR "
        "and Proliferative DR.</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown("### Technical Workflow")
    st.markdown(
        "- **Input:** JPG, JPEG or PNG retinal fundus image.\n"
        "- **Preprocessing:** RGB conversion, resize and normalization as configured in model.py.\n"
        "- **Model:** Vision Transformer with five output classes.\n"
        "- **Checkpoint:** `models/best_vit_retinal_improved.pth`.\n"
        "- **Output:** predicted class and class probabilities.\n"
        "- **History:** prediction metadata and image previews saved locally."
    )
    st.markdown(
        "<div class='panel status-note'>Research prototype only. Predictions may be wrong and "
        "must not be used for medical decisions. Consult a qualified eye-care professional.</div>",
        unsafe_allow_html=True,
    )


# Render the selected page. Changing the radio selection triggers a Streamlit rerun
# and this dispatcher displays the matching page.
page = render_sidebar()
render_topbar()

if page == "Home":
    render_home()
elif page == "Upload & Predict":
    render_upload()
elif page == "History":
    render_history()
elif page == "About":
    render_about()
