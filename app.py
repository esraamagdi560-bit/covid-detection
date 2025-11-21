import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model(r"F:\AI course\covid detection\covid_19_model.h5")  # saved model (params + architecture)

class_map = {0: 'Covid', 1: 'Normal', 2: 'viral Pneumonia'}

x_resize = 224
y_resize = 224
dims = 3

## functions for preprocessing and prediction
def preprocess(image, x_resize, y_resize):
    # convert PIL image to numpy array
    img_array = np.array(image)
    # Resize the image
    img_array = cv2.resize(img_array, (x_resize, y_resize))
    #convert to grayscale
    if img_array.ndim == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2GRAY)

    # Normalize the image
    img_array = img_array.astype('float32') / 255.0

    # reshape image to match model input shape
    new_image = img_array.reshape(1, x_resize, y_resize, dims)

    return new_image

def predict(image):
    pred = model.predict(image)
    pred_label = np.argmax(pred, axis=1)
    pred_class = class_map[pred_label[0]]
    return pred_class

## --- Styling & layout enhancements  ---
PAGE_STYLE = """
<style>
/* Page background & font */
body {background-color: #0f1724; color: #e6eef8;}
.stApp {background-color: #0f1724}
h1, .st-ag {color:#e6eef8}

/* Title styling */
header {text-align:center}
.title {font-family: "Segoe UI", Roboto, sans-serif; font-size:34px; color:#e6eef8; margin-bottom:0.2rem}
.subtitle {color:#9fb2d0; margin-top:0; margin-bottom:1.2rem}

/* Card for uploader and preview */
.card {background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
       border-radius:12px; padding:16px; box-shadow: 0 4px 18px rgba(2,6,23,0.6);}

/* Center uploaded image */
.uploaded-image {display:block; margin-left:auto; margin-right:auto; border-radius:8px; max-width:100%;}

/* Styled result box */
.result {background:#05233b; padding:12px; border-radius:8px; color:#bfe6ff; font-weight:600}

/* Make Streamlit buttons a bit nicer (best-effort) */
button[kind] { }

/* Small screens adjustments */
@media (max-width: 640px) {
  .title {font-size:24px}
}
</style>
"""

st.markdown(PAGE_STYLE, unsafe_allow_html=True)

st.markdown("""
<div header>
  <div class='title'>Covid-19 Detection from Chest X‑Ray Images</div>
  <div class='subtitle'>Upload a chest X‑ray image to detect Covid‑19 — model inference runs locally.</div>
</div>
""", unsafe_allow_html=True)

with st.container():
    left, right = st.columns([1, 1])

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload an X‑ray image", type=["jpg", "jpeg", "png"])
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div style='color:#cfeeff; font-size:15px;'><strong>Instructions</strong> — prefer frontal chest X‑rays (PNG/JPEG). Click Predict after upload.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Preview & predict area below
    st.markdown("<br>", unsafe_allow_html=True)

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.image(img, caption='Uploaded Image')
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button('Predict'):
        if img:
            processed_image = np.array(img)  # preview only; actual preprocess() called below
            # Call existing preprocess + predict
            from_types_back = True
            processed_image = preprocess(img, x_resize, y_resize)
            prediction = predict(processed_image)
            st.markdown(f"<div class='result'>Prediction: {prediction}</div>", unsafe_allow_html=True)
        else:
            st.write("Please upload an image to get started.")


