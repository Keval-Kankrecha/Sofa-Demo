import requests
import streamlit as st
import numpy as np
from PIL import Image
from rembg import remove
import cv2
import io

st.set_page_config(page_title="🛋️ Sofa Recolor Tool", layout="wide")

st.markdown("""
    <style>
    .css-1v0mbdj {padding-top: 2rem;}
    </style>
""", unsafe_allow_html=True)

st.title("🛋️ Sofa Recolor Tool")
st.markdown("### Recolor your sofa using AI! Upload or use the sample image and choose a new color.")

# --- Load default image ---
DEFAULT_IMAGE_URL = "https://i.ibb.co/8L5LX827/sofa-img.jpg"

@st.cache_data(show_spinner=False)
def load_default_image():
    return Image.open(io.BytesIO(requests.get(DEFAULT_IMAGE_URL).content)).convert("RGBA")

# --- Upload image or use default ---
col1, col2 = st.columns([1, 2])

with col1:
    image = load_default_image()

    st.image(image, caption="Original Sofa", use_container_width=True)
    st.markdown("""
    **🛋️ Sofa Details:**
    - **Type**: L Shape  
    - **Description**: Modern sectional sofa with plush cushioning and elegant curved design. Perfect for contemporary living spaces with comfortable seating for multiple people.
    """)

# --- Color swatches ---
color_presets = {
    "Light Gray": (229, 228, 226),
    "Cream": (255, 229, 180),
    "Dark Gray": (169, 169, 169),
    "Royal Blue": (70, 105, 225),
    "Light Yellow": (255, 253, 208),
    "Burgundy": (136, 51, 51),
    "Purple": (121, 103, 153),
    "Coral": (255, 127, 80),
    "Brown": (168, 115, 44),
    "Tan": (196, 164, 132),
    "Blue": (0, 0, 255),
    "Sky Blue": (135, 206, 235),
    "Silver": (192, 194, 201),
    "Salmon": (250, 128, 114),
    "Dark Brown": (36, 14, 1),
    "Chocolate": (96, 62, 33),
}

# --- Image processing functions ---
def isolate_sofa(image_np):
    """Remove background from RGBA image using rembg"""
    pil_image = Image.fromarray(image_np)
    output = remove(pil_image)
    return np.array(output)

def apply_color_cv(image_np, target_rgb):
    """Apply grayscale shading of target_rgb to image with alpha"""
    image = image_np.astype(np.float32) / 255.0
    alpha = image[:, :, 3:4]

    rgb_image = image[:, :, :3]
    bgr_image = cv2.cvtColor((rgb_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = np.expand_dims(gray, axis=2)

    target = np.array(target_rgb).astype(np.float32) / 255.0
    recolored_rgb = gray * target

    recolored = np.concatenate((recolored_rgb, alpha), axis=2)
    recolored = (recolored * 255).astype(np.uint8)
    return recolored

def image_np_to_pil(img_np):
    return Image.fromarray(img_np)

with col2:
    st.markdown("### 🎨 Choose a Color")
    selected_color = st.selectbox("Color", list(color_presets.keys()))
    rgb = color_presets[selected_color]

    # Color preview
    st.markdown(f"#### Preview: {selected_color}")
    st.color_picker(" ", value='#%02x%02x%02x' % rgb, label_visibility="collapsed", disabled=True)

    if st.button("✨ Recolor Sofa"):
        with st.spinner("Recoloring your sofa..."):
            image_np = np.array(image)
            sofa_only = isolate_sofa(image_np)
            recolored = apply_color_cv(sofa_only, rgb)
            st.image(recolored, caption=f"Sofa Recolored - {selected_color}", use_container_width=True)

            # Optional: Download button
            output_img = image_np_to_pil(recolored)
            buf = io.BytesIO()
            output_img.save(buf, format="PNG")
            byte_im = buf.getvalue()

            st.download_button(
                label="💾 Download Recolored Sofa",
                data=byte_im,
                file_name=f"sofa_{selected_color.replace(' ', '_').lower()}.png",
                mime="image/png"
            )
