import streamlit as st
import numpy as np
import cv2
from PIL import Image

from websiteBackend import load_models, run_pipeline

st.set_page_config(layout="wide")

st.title("Mammography AI Hybrid CAD System")

# Load models once
@st.cache_resource
def get_models():
    return load_models()

detector, hybrid_model = get_models()

# Inputs
uploaded_file = st.file_uploader(
    "Upload Mammogram Image",
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)

col1, col2, col3 = st.columns(3)
age = col1.number_input("Age", 10, 100, 50)
birads = col2.number_input("BIRADS", 1, 6, 3)
density = col3.number_input("Density", 1, 4, 2)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    if image.mode != "RGB":
        image = image.convert("RGB")

    img = np.array(image)

    if st.button("Run Hybrid Analysis"):

        enhanced, gradcam, edge, prob, uncertainty, score = run_pipeline(
            detector, hybrid_model, img, age, birads, density
        )

        st.subheader("Results")

        st.write(f"**Cancer Probability:** {prob*100:.2f}%")
        st.write(f"**Uncertainty:** ± {uncertainty*100:.2f}%")
        st.write(f"**Detection Score:** {score}")

        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        col1.image(enhanced, caption="Detection")
        col2.image(gradcam, caption="Grad-CAM")
        col3.image(enhanced, caption="CLAHE Enhanced")
        col4.image(edge, caption="Edge Map")
