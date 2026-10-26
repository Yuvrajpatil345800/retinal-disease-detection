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

# RetinaAI UI/UX version. This file is intentionally separate from app.py.
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
PAGE_OPTIONS = ["Home", "Upload & Predict", "History", "About"]
NAV_LABELS = {
    "Home": "⌂  Home",
    "Upload & Predict": "⇧  Upload & Predict",
    "History": "◷  History",
    "About": "ⓘ  About",
}
MAX_IMAGE_BYTES = 10 * 1024 * 1024

st.set_page_config(
    page_title="RetinaAI | Retinal Disease Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark navy / cyan design matching the six-page reference supplied by the user.
st.markdown(
    """
    <style>
      :root { --ra-bg:#06162c; --ra-sidebar:#071b36; --ra-panel:#0b2342;
              --ra-panel2:#0c2a4e; --ra-line:#1b426d; --ra-blue:#1688ff;
              --ra-cyan:#51d7ff; --ra-text:#f0f6ff; --ra-muted:#a9c0de; }
      .stApp { background:var(--ra-bg); color:var(--ra-text); }
      [data-testid="stHeader"] { background:rgba(6,22,44,.98); }
      [data-testid="stSidebar"] { background:var(--ra-sidebar); border-right:1px solid #133458; }
      [data-testid="stSidebar"] > div:first-child { padding-top:1rem; }
      .block-container { max-width:1500px; padding-top:1.15rem; padding-bottom:2rem; }
      h1,h2,h3,h4 { color:var(--ra-text)!important; letter-spacing:-.02em; }
      p,li,label { color:#d8e7fa; }
      [data-testid="stCaptionContainer"] p, .muted { color:var(--ra-muted)!important; }
      .topbar { display:flex; align-items:center; justify-content:space-between; gap:1rem;
        border-bottom:1px solid #19385b; padding:.15rem .1rem .9rem; margin-bottom:1.25rem; }
      .brand { display:flex; align-items:center; gap:.55rem; font-size:1.04rem; font-weight:800; color:white; }
      .brand-eye { display:flex; align-items:center; justify-content:center; width:1.75rem; height:1.75rem;
        color:#fff; background:linear-gradient(145deg,#258dff,#0756d9); border:1px solid #62b5ff;
        border-radius:50%; box-shadow:0 0 14px rgba(25,132,255,.3); }
      .brand-eye svg { width:1.2rem; height:1.2rem; display:block; }
      .student-tag { display:flex; align-items:center; justify-content:flex-end; gap:.45rem;
        color:#d5e4f6; font-size:.84rem; text-align:right; white-space:nowrap; }
      .profile-avatar { display:inline-flex; align-items:center; justify-content:center; width:1.45rem; height:1.45rem;
        border-radius:50%; background:#1b4d80; border:1px solid #5a9ee1; color:#fff; font-size:.72rem; font-weight:750; }
      .hero { display:flex; align-items:center; gap:2rem; min-height:255px; padding:1.75rem 1.8rem;
        border-radius:15px; background:linear-gradient(115deg,#0a2344 0%,#081d39 56%,#0d2b51 100%);
        border:1px solid #17385e; margin-bottom:1.15rem; }
      .hero-copy { flex:1; min-width:0; }
      .eyebrow { color:var(--ra-cyan); font-weight:750; font-size:.8rem; margin-bottom:.55rem; }
      .hero-title { font-weight:850; font-size:2.15rem; line-height:1.13; margin:0 0 .75rem; color:#f3f7ff; }
      .hero-title span { color:#55aaff; }
      .hero-desc { color:#c0d3ed; font-size:.94rem; line-height:1.55; max-width:610px; }
      .hero-image { width:230px; min-width:185px; aspect-ratio:1; border-radius:50%; object-fit:cover;
        border:3px solid #193e66; box-shadow:0 0 0 12px rgba(28,111,193,.07),0 0 34px rgba(31,132,255,.13); }
      .panel { background:var(--ra-panel); border:1px solid #173a62; border-radius:12px; padding:1rem 1.05rem; }
      .feature-card { height:100%; min-height:126px; background:#0b2342; border:1px solid #173b63;
        border-radius:10px; padding:1rem .95rem; }
      .feature-icon { color:#56aaff; font-size:1.35rem; margin-bottom:.42rem; }
      .feature-title { font-weight:800; color:#f1f6ff; margin-bottom:.32rem; }
      .feature-copy { font-size:.78rem; line-height:1.4; color:#afc5e1; }
      .section-title { font-size:1.05rem; font-weight:800; margin:.25rem 0 .8rem; }
      .pill { display:inline-block; padding:.4rem .78rem; border-radius:999px; background:#ffd166;
        color:#1b2433; font-weight:850; font-size:.96rem; }
      .kpi { border:1px solid #1b436d; background:#0b2749; border-radius:10px; padding:.9rem 1rem; min-height:94px; }
      .kpi-label { font-size:.78rem; color:#abc2de; }
      .kpi-value { color:white; font-size:1.55rem; line-height:1.35; font-weight:850; margin-top:.15rem; }
      .kpi-sub { color:#87a8cf; font-size:.72rem; margin-top:.1rem; }
      .prob-row { display:grid; grid-template-columns:135px 1fr 52px; gap:10px; align-items:center;
        margin:.62rem 0; color:#dbe8fb; font-size:.78rem; }
      .prob-track { height:9px; border-radius:99px; background:#1b385d; overflow:hidden; }
      .prob-fill { height:100%; border-radius:99px; background:#1688ff; }
      .prob-fill.selected { background:#ffca55; }
      .prob-value { text-align:right; white-space:nowrap; color:#eef5ff; }
      .drop-hint { text-align:center; background:#0a2040; border:1px dashed #245783; padding:1.2rem; border-radius:10px; }
      .analysis-panel { min-height:360px; display:flex; flex-direction:column; align-items:center;
        justify-content:center; text-align:center; padding:2.2rem; border-radius:12px;
        background:linear-gradient(145deg,#0a2344,#0b2a50); border:1px solid #1b4f83; }
      .analysis-spinner { width:54px; height:54px; border:4px solid #174776; border-top-color:#1688ff;
        border-right-color:#51c8ff; border-radius:50%; animation:ra-spin .85s linear infinite; margin-bottom:1.1rem; }
      @keyframes ra-spin { to { transform:rotate(360deg); } }
      .analysis-title { color:#f3f7ff; font-size:1.25rem; font-weight:800; margin-bottom:.45rem; }
      .analysis-copy { color:#b8cce5; font-size:.9rem; margin-bottom:1.3rem; }
      .analysis-track { width:min(100%,560px); height:9px; border-radius:99px; background:#183a61; overflow:hidden; }
      .analysis-fill { height:100%; border-radius:99px; background:linear-gradient(90deg,#1688ff,#51c8ff); transition:width .25s ease; }
      .analysis-step { color:#9fb8d7; font-size:.78rem; margin-top:.8rem; }
      [data-testid="stImage"] img { border:1px solid #1b426d; border-radius:9px; }
      .sample-caption { text-align:center; font-size:.77rem; color:#d2e0f3; padding-top:.35rem; }
      .table-head { color:#c1d7f1; background:#0b2a4e; font-size:.76rem; font-weight:750;
        padding:.65rem .5rem; border-bottom:1px solid #234a73; }
      .history-row { background:#0a2140; border-bottom:1px solid #173a61; border-radius:6px; padding:.3rem .15rem; }
      .status-note { color:#a7c0dd; font-size:.82rem; }
      div.stButton > button { border-radius:7px; border:1px solid #235383; background:#0c2e55; color:#eef6ff; font-weight:700; }
      div.stButton > button:hover { background:#104078; border-color:#1688ff; color:white; }
      div.stButton > button[kind="primary"] { background:#1688ff; border:0; color:white; font-weight:800; }
      div.stButton > button[kind="primary"]:hover { background:#3ba0ff; color:white; }
      [data-testid="stFileUploaderDropzone"] { background:#0a2040; border:1px dashed #255685; border-radius:10px; }
      [data-testid="stFileUploaderDropzoneInstructions"] { color:#d3e4fa; }
      [data-testid="stProgressBar"] > div > div { background:#1688ff; }
      [data-testid="stSidebar"] .block-container { padding-top:1rem; }
      [data-testid="stSidebar"] div[data-testid="stRadio"] > label { display:none; }
      [data-testid="stSidebar"] div[role="radiogroup"] { gap:.38rem; }
      [data-testid="stSidebar"] div[role="radiogroup"] label { padding:.52rem .7rem; border:1px solid transparent;
        border-left:3px solid transparent; border-radius:6px; background:transparent; min-height:2.45rem;
        transition:background .16s ease,border-color .16s ease; }
      [data-testid="stSidebar"] div[role="radiogroup"] label:hover { background:#0a2546; border-color:#163a60; }
      [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) { background:#0d2d52; border-color:#174574; border-left-color:#2696ff; }
      [data-testid="stSidebar"] div[role="radiogroup"] label p { color:#d4e2f3; font-size:.9rem; font-weight:550; }
      [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p { color:#fff; font-weight:700; }
      [data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child { margin-right:.42rem; }
      hr { border-color:#1a3b60; }
      footer { visibility:hidden; }
      @media(max-width:850px) {
        .hero { display:block; padding:1.3rem; }
        .hero-image { width:165px; margin-top:1rem; }
        .hero-title { font-size:1.7rem; }
        .prob-row { grid-template-columns:100px 1fr 48px; gap:6px; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_model():
    """Load the trained model from model.py once per Streamlit process."""
    return load_model()


def ensure_storage() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def read_history() -> list[dict]:
    ensure_storage()
    if not HISTORY_FILE.exists():
        return []
    try:
        contents = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return [record for record in contents if isinstance(record, dict)] if isinstance(contents, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def write_history(records: list[dict]) -> None:
    ensure_storage()
    HISTORY_FILE.write_text(json.dumps(records, indent=2), encoding="utf-8")


def image_to_rgb(image_bytes: bytes) -> Image.Image:
    with Image.open(BytesIO(image_bytes)) as source:
        source.load()
        return source.convert("RGB")


def image_hash(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()[:20]


def save_history_record(image: Image.Image, prediction: dict, filename: str) -> None:
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
    # Keep only the latest 100 records to avoid unlimited local file growth.
    old_records = records[100:]
    write_history(records[:100])
    for old in old_records:
        old_path = HISTORY_IMAGES_DIR / str(old.get("image_file", ""))
        if old_path.is_file():
            try:
                old_path.unlink()
            except OSError:
                pass


def show_probabilities(probabilities: dict, predicted_class: str | None = None) -> None:
    st.markdown("<div class='section-title'>Class Probabilities</div>", unsafe_allow_html=True)
    for class_name in CLASS_NAMES:
        value = float(probabilities.get(class_name, 0.0))
        value = max(0.0, min(1.0, value))
        fill_class = "prob-fill selected" if class_name == predicted_class else "prob-fill"
        label = DISPLAY_NAMES.get(class_name, class_name)
        st.markdown(
            f"<div class='prob-row'><div>{label}</div>"
            f"<div class='prob-track'><div class='{fill_class}' style='width:{value * 100:.2f}%'></div></div>"
            f"<div class='prob-value'>{value * 100:.1f}%</div></div>",
            unsafe_allow_html=True,
        )


def find_sample_images() -> list[Path]:
    if not SAMPLE_IMAGES_DIR.is_dir():
        return []
    # The only confirmed real sample is the image supplied by the user.
    paths = [SAMPLE_IMAGES_DIR / "Mild_DR.png"]
    paths = [path for path in paths if path.is_file()]
    order = ["no dr", "mild", "moderate", "severe", "proliferative"]
    def sort_key(path: Path):
        stem = path.stem.lower().replace("_", " ").replace("-", " ")
        for i, key in enumerate(order):
            if key in stem or (i == 0 and "no dr" in stem):
                return (0, i, path.name.lower())
        return (1, len(order), path.name.lower())
    return sorted(paths, key=sort_key)[:5]


def sample_class(path: Path) -> str:
    stem = path.stem.lower().replace("_", " ").replace("-", " ")
    if "proliferative" in stem:
        return "Proliferative DR"
    if "moderate" in stem:
        return "Moderate DR"
    if "severe" in stem:
        return "Severe DR"
    if "mild" in stem:
        return "Mild DR"
    if "no dr" in stem or "no retinopathy" in stem or "no dr" in stem.replace("_", " "):
        return "No DR"
    return "Sample image"


def image_data_uri(path: Path | None) -> str | None:
    if not path or not path.is_file():
        return None
    try:
        import base64
        raw = path.read_bytes()
        suffix = path.suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
    except OSError:
        return None


def render_topbar() -> None:
    left, right = st.columns([7, 1.25])
    with left:
        st.markdown(
            "<div class='topbar'><div class='brand'><span class='brand-eye'>"
            "<svg viewBox='0 0 24 24' aria-hidden='true'><path d='M2 12s3.6-6 10-6 10 6 10 6-3.6 6-10 6S2 12 2 12Z' fill='none' stroke='white' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'/><circle cx='12' cy='12' r='3.2' fill='white'/><circle cx='12' cy='12' r='1.35' fill='#1478ee'/></svg>"
            "</span> RetinaAI</div></div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("<div class='student-tag' style='padding:.28rem 0 .7rem'><span class='profile-avatar'>R</span> Rahul</div>", unsafe_allow_html=True)


def render_sidebar() -> str:
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = "Home"
    with st.sidebar:
        st.markdown("<div style='height:.15rem'></div>", unsafe_allow_html=True)
        page = st.radio(
            "Navigation",
            PAGE_OPTIONS,
            format_func=lambda value: NAV_LABELS[value],
            key="nav_page",
            label_visibility="collapsed",
        )
        st.markdown("<div style='height:1rem'></div><div class='status-note'>AI Retinal Screening</div>", unsafe_allow_html=True)
    return page


def go_to_upload() -> None:
    st.session_state.nav_page = "Upload & Predict"


def render_home() -> None:
    samples = find_sample_images()
    hero_uri = image_data_uri(samples[0]) if samples else None
    image_html = (
        f"<img class='hero-image' src='{hero_uri}' alt='Retinal fundus image'/>"
        if hero_uri else
        "<div class='hero-image' style='display:flex;align-items:center;justify-content:center;color:#58aaff;font-size:4rem'>◉</div>"
    )
    st.markdown(
        "<div class='hero'><div class='hero-copy'>"
        "<div class='eyebrow'>AI for Better Vision</div>"
        "<div class='hero-title'>Retinal Disease<br>Detection using<br><span>Vision Transformer</span></div>"
        "<div class='hero-desc'>Our AI-assisted research system classifies retinal fundus images into five diabetic retinopathy severity categories. It provides model predictions to support academic evaluation and learning.</div>"
        "</div>" + image_html + "</div>",
        unsafe_allow_html=True,
    )
    st.button("Get Started  →", key="home_get_started", type="primary", on_click=go_to_upload)
    st.write("")
    cols = st.columns(4, gap="small")
    cards = [
        ("⚙", "AI Classification", "Powered by your trained Vision Transformer (ViT) checkpoint."),
        ("✣", "Five Disease Classes", "No DR, Mild, Moderate, Severe and Proliferative DR."),
        ("◴", "Fast Analysis", "Run model inference when an image is submitted."),
        ("♥", "Prediction History", "Review previous predictions saved on this device."),
    ]
    for col, (icon, title, body) in zip(cols, cards):
        with col:
            st.markdown(
                f"<div class='feature-card'><div class='feature-icon'>{icon}</div>"
                f"<div class='feature-title'>{title}</div><div class='feature-copy'>{body}</div></div>",
                unsafe_allow_html=True,
            )
    st.write("")
    lower_left, lower_right = st.columns([1.25, 1], gap="large")
    with lower_left:
        st.markdown("<div class='section-title'>How it works</div>", unsafe_allow_html=True)
        st.markdown(
            "1. Upload one retinal fundus image.\n2. The image is resized and normalized.\n"
            "3. The trained ViT predicts one of five classes.\n4. Review class probabilities and saved history."
        )
    with lower_right:
        st.markdown("<div class='section-title'>Supported classes</div>", unsafe_allow_html=True)
        st.markdown("<div class='panel'>No DR &nbsp; • &nbsp; Mild DR &nbsp; • &nbsp; Moderate DR<br><br>Severe DR &nbsp; • &nbsp; Proliferative DR</div>", unsafe_allow_html=True)
    st.markdown("<div class='panel status-note'>Research prototype only. This model is not clinically validated and is not a substitute for professional medical advice.</div>", unsafe_allow_html=True)


def render_sample_gallery() -> None:
    samples = find_sample_images()
    # Hide the optional sample gallery when no sample_images folder exists.
    # Users can still upload any retinal image directly; no setup message is shown.
    if not samples:
        return
    st.markdown("<div class='section-title'>Sample Images</div>", unsafe_allow_html=True)
    cols = st.columns(5, gap="small")
    for idx, (col, sample) in enumerate(zip(cols, samples)):
        with col:
            try:
                st.image(str(sample), use_container_width=True)
            except Exception:
                st.caption("Preview unavailable")
            st.markdown(f"<div class='sample-caption'>{sample_class(sample)}</div>", unsafe_allow_html=True)
            if st.button("Use image", key=f"use_sample_{sample.name}", use_container_width=True):
                st.session_state.active_sample_path = str(sample)
                st.session_state.pop("last_analysis", None)
                st.rerun()


def show_result_card(result: dict, image: Image.Image, filename: str) -> None:
    prediction = str(result.get("prediction", "Unknown"))
    display_prediction = DISPLAY_NAMES.get(prediction, prediction)
    confidence = max(0.0, min(1.0, float(result.get("confidence", 0.0))))
    st.markdown("<div class='section-title' style='font-size:1.3rem'>Prediction Results</div>", unsafe_allow_html=True)
    left, right = st.columns([1, 1.05], gap="small")
    with left:
        st.image(image, use_container_width=True)
        st.caption(f"Uploaded Image · {filename}")
    with right:
        st.markdown(
            f"<div class='panel'><div class='muted'>Predicted Class</div><div style='margin:.55rem 0 .75rem'><span class='pill'>{display_prediction}</span></div>"
            f"<div class='muted'>Confidence Score</div><div style='font-size:1.7rem;font-weight:850;color:#f5f9ff'>{confidence * 100:.2f}%</div>"
            "<div class='status-note' style='margin-top:.65rem;border:1px solid #204b79;border-radius:7px;padding:.6rem'>This prediction is produced by the trained Vision Transformer model. For medical advice, consult an eye-care professional.</div></div>",
            unsafe_allow_html=True,
        )
    st.markdown("<div class='panel' style='margin-top:.7rem'>", unsafe_allow_html=True)
    show_probabilities(result.get("probabilities", {}), prediction)
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("Model probabilities are not a confirmed medical diagnosis. High confidence does not guarantee a correct prediction.")


def analyse_current_image(image_bytes: bytes, filename: str, image: Image.Image) -> None:
    fingerprint = image_hash(image_bytes)
    if st.button("Analyze Image  →", type="primary", key=f"analyse_{fingerprint}", use_container_width=True):
        panel = st.empty()

        def show_progress(message: str, percent: int) -> None:
            panel.markdown(
                "<div class='analysis-panel'><div class='analysis-spinner'></div>"
                "<div class='analysis-title'>Analyzing Your Image...</div>"
                "<div class='analysis-copy'>Please wait while the Vision Transformer processes the retinal image<br>and predicts the disease severity.</div>"
                f"<div class='analysis-track'><div class='analysis-fill' style='width:{percent}%'></div></div>"
                f"<div class='analysis-step'>{message}</div></div>",
                unsafe_allow_html=True,
            )

        try:
            show_progress("Preparing retinal image…", 15)
            show_progress("Loading the trained Vision Transformer…", 35)
            model = get_model()
            show_progress("Running five-class inference…", 70)
            result = predict_image(model, image)
            show_progress("Preparing your results…", 92)
            save_history_record(image, result, filename)
            st.session_state.last_analysis = {
                "fingerprint": fingerprint,
                "filename": filename,
                "result": result,
            }
            st.rerun()
        except FileNotFoundError as error:
            panel.empty()
            st.error("Check that `models/best_vit_retinal_improved.pth` exists and that model.py points to it.")
            st.code(str(error))
        except Exception as error:
            panel.empty()
            st.error("The uploaded image could not be analyzed. Check the VS Code terminal for details.")
            st.exception(error)


def render_upload() -> None:
    st.markdown("<div class='section-title' style='font-size:1.3rem'>Upload Retinal Image</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drag and drop your retinal image here",
        type=["jpg", "jpeg", "png"],
        key="retina_file_uploader",
        help="Supported formats: JPG, JPEG, PNG. Maximum application size: 10 MB.",
    )
    st.caption("Supported formats: JPG, JPEG, PNG  ·  Maximum size: 10 MB")

    if uploaded is not None:
        image_bytes = uploaded.getvalue()
        filename = uploaded.name
    else:
        image_bytes = b""
        filename = ""

    if not image_bytes:
        st.markdown("<div class='drop-hint'><div style='font-size:1.8rem;color:#52aaff'>⇧</div><b>No image selected yet</b><div class='status-note' style='margin-top:.3rem'>Upload a retinal fundus image to begin.</div></div>", unsafe_allow_html=True)
        return
    if len(image_bytes) > MAX_IMAGE_BYTES:
        st.error("This file is larger than the 10 MB limit. Please choose a smaller JPG or PNG image.")
        return
    try:
        image = image_to_rgb(image_bytes)
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("This file could not be opened as an image. Please upload a valid JPG or PNG file.")
        return

    fingerprint = image_hash(image_bytes)
    st.divider()
    analyse_current_image(image_bytes, filename, image)
    last = st.session_state.get("last_analysis")
    if last and last.get("fingerprint") == fingerprint:
        show_result_card(last["result"], image, filename)
    else:
        left, right = st.columns([1, 1.25], gap="small")
        with left:
            st.markdown("<div class='section-title'>Uploaded Image</div>", unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.caption(filename)
        with right:
            st.markdown("<div class='section-title'>Prediction Results</div>", unsafe_allow_html=True)
            st.markdown("<div class='panel' style='min-height:180px;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center'><div style='font-size:2rem;color:#1688ff'>◌</div><b>Ready to Analyze</b><div class='status-note' style='margin-top:.4rem'>Click Analyze Image to run your trained model.</div></div>", unsafe_allow_html=True)


def render_history() -> None:
    st.markdown("<div class='section-title' style='font-size:1.35rem'>Prediction History</div>", unsafe_allow_html=True)
    records = read_history()
    if not records:
        st.info("Your prediction history will appear here after you analyze an image.")
        return
    st.caption("Entries are saved locally in the project's data folder on this computer. They are not synchronized across devices by this version.")
    header = st.columns([1.5, .85, 1.7, 1.0, .75])
    for col, label in zip(header, ["Date & Time", "Image", "Predicted Class", "Confidence", "Action"]):
        with col:
            st.markdown(f"<div class='table-head'>{label}</div>", unsafe_allow_html=True)
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
            predicted = str(record.get("prediction", "Unknown"))
            st.markdown(f"**{DISPLAY_NAMES.get(predicted, predicted)}**")
            st.caption(str(record.get("original_name", "")))
        with columns[3]:
            st.markdown(f"**{float(record.get('confidence', 0.0)) * 100:.1f}%**")
        with columns[4]:
            if st.button("View", key=f"view_{record.get('id', uuid.uuid4().hex)}", use_container_width=True):
                st.session_state.view_history_id = record.get("id")
                st.rerun()
        st.markdown("<hr style='margin:.2rem 0 .65rem'>", unsafe_allow_html=True)

    selected_id = st.session_state.get("view_history_id")
    selected = next((r for r in records if r.get("id") == selected_id), None)
    if selected:
        st.markdown("### Selected prediction")
        cols = st.columns([1, 1.2], gap="large")
        with cols[0]:
            image_path = HISTORY_IMAGES_DIR / str(selected.get("image_file", ""))
            if image_path.is_file():
                st.image(str(image_path), use_container_width=True)
        with cols[1]:
            pred = str(selected.get("prediction", "Unknown"))
            st.markdown(f"<span class='pill'>{DISPLAY_NAMES.get(pred, pred)}</span>", unsafe_allow_html=True)
            st.metric("Model confidence", f"{float(selected.get('confidence', 0.0)) * 100:.2f}%")
            show_probabilities(selected.get("probabilities", {}), pred)


def render_about() -> None:
    st.markdown("<div class='section-title' style='font-size:1.35rem'>About This Project</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='panel'><b>RetinaAI — Vision Transformer (ViT)</b><p class='status-note' style='margin-top:.5rem'>This academic project uses a trained Vision Transformer to classify retinal fundus images into five diabetic retinopathy severity categories: No DR, Mild DR, Moderate DR, Severe DR and Proliferative DR.</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown("### Evaluation")
    st.markdown("<div class='panel status-note'>Verified evaluation metrics are not available in this application.</div>", unsafe_allow_html=True)
    st.markdown("### Technical Workflow")
    st.markdown(
        "- **Input:** JPG, JPEG or PNG retinal fundus image.\n"
        "- **Preprocessing:** RGB conversion, resize to 224 × 224, tensor conversion and ImageNet normalization.\n"
        "- **Model:** `vit_base_patch16_224` with five output classes.\n"
        "- **Checkpoint:** `models/best_vit_retinal_improved.pth`, loaded through `model.py`.\n"
        "- **Output:** predicted class and five softmax probabilities.\n"
        "- **History:** prediction metadata and image previews saved locally in the project `data` folder."
    )
    st.markdown("<div class='panel status-note'>Research prototype only. Predictions may be wrong and must not be used for medical decisions. Consult a qualified eye-care professional.</div>", unsafe_allow_html=True)


page = render_sidebar()
render_topbar()
if page == "Home":
    render_home()
elif page == "Upload & Predict":
    render_upload()
elif page == "History":
    render_history()
else:
    render_about()
