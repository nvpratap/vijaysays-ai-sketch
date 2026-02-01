import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="VijaySays AI Sketch Tool", layout="centered")
st.set_option('server.maxUploadSize', 5)

st.markdown(
    "<h1 style='text-align:center;'>🖌️ Image to Sketch AI</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Powered by <b>VijaySays</b></p>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"]
)

style = st.selectbox(
    "Choose Sketch Style",
    ["Pencil Sketch", "Charcoal Sketch", "Edge Outline"]
)

if uploaded_file:
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()), dtype=np.uint8
    )
    image = cv2.imdecode(file_bytes, 1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if style == "Pencil Sketch":
        inverted = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(inverted, (21, 21), 0)
        inverted_blur = cv2.bitwise_not(blur)
        sketch = cv2.divide(gray, inverted_blur, scale=256.0)

    elif style == "Charcoal Sketch":
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        sketch = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 9, 2
        )

    else:  # Edge Outline
        sketch = cv2.Canny(gray, 50, 150)

    st.image(sketch, caption=f"{style} Output", channels="GRAY")

    st.download_button(
        label="⬇️ Download Sketch",
        data=cv2.imencode(".png", sketch)[1].tobytes(),
        file_name="vijaysays_sketch.png",
        mime="image/png"
    )
