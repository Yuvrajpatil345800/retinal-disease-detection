import streamlit as st
from PIL import Image, UnidentifiedImageError
from model import load_model, predict_image

st.set_page_config(page_title="AI Retinal DR Detection", page_icon="👁️", layout="wide")
st.title("👁️ AI-Based Diabetic Retinopathy Detection")
st.caption("Vision Transformer (ViT) · Research Prototype")
st.warning("For academic demonstration only. This model is not validated for clinical diagnosis. Predictions may be incorrect.")


@st.cache_resource
def get_model():
    return load_model()


uploaded_file = st.file_uploader(
    "Upload a retinal fundus image",
    type=["jpg", "jpeg", "png"],
    help="Upload one retinal fundus photograph, not a comparison or collage.",
)

if uploaded_file is None:
    st.info("Upload a retinal image to begin.")
else:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        left, right = st.columns([1, 1.35], gap="large")
        with left:
            st.subheader("Uploaded Retinal Image")
            st.image(image, width=350)
            analyze = st.button("Analyze Image", type="primary", use_container_width=True)
        if analyze:
            with st.spinner("Loading model and analyzing image..."):
                result = predict_image(get_model(), image)
            with right:
                st.subheader("Prediction Result")
                st.success(f"Predicted class: {result['prediction']}")
                st.metric("Model confidence", f"{result['confidence'] * 100:.2f}%")
                st.subheader("Class Probabilities")
                for class_name, probability in result["probabilities"].items():
                    percentage = max(0.0, min(1.0, float(probability)))
                    st.write(f"**{class_name} · {percentage * 100:.2f}%**")
                    st.progress(max(0.02, percentage))
                st.caption("Probabilities are model outputs, not a confirmed medical diagnosis. High confidence does not guarantee a correct result.")
    except (UnidentifiedImageError, OSError):
        st.error("This image could not be opened. Please upload a valid JPG or PNG.")
    except Exception as error:
        st.error("Analysis failed. Check the VS Code terminal for details.")
        st.exception(error)
