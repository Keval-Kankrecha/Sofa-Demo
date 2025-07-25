import numpy as np
import cv2
from PIL import Image
from rembg import remove
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import io
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def home():
    return render_template('index.html')  # Serve HTML UI

def load_image_from_bytes(file_bytes):
    image = Image.open(io.BytesIO(file_bytes)).convert("RGBA")
    return np.array(image)

def isolate_sofa(image_np):
    pil_image = Image.fromarray(image_np)
    output = remove(pil_image)
    return np.array(output)

def apply_color_cv(image_np, target_rgb):
    # Normalize image and target color
    image = image_np.astype(np.float32) / 255.0
    alpha = image[:, :, 3:4]

    # Convert RGB to Grayscale using OpenCV
    rgb_image = image[:, :, :3]
    bgr_image = cv2.cvtColor((rgb_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = np.expand_dims(gray, axis=2)

    # Apply target color
    target = np.array(target_rgb).astype(np.float32) / 255.0
    recolored_rgb = gray * target

    # Combine with alpha channel
    recolored = np.concatenate((recolored_rgb, alpha), axis=2)
    recolored = (recolored * 255).astype(np.uint8)
    return recolored

def image_to_bytes(image_np):
    pil_img = Image.fromarray(image_np)
    byte_io = io.BytesIO()
    pil_img.save(byte_io, format='PNG')
    byte_io.seek(0)
    return byte_io

@app.route('/recolor', methods=['POST'])
def recolor_endpoint():
    if 'image' not in request.files or 'color' not in request.form:
        return jsonify({'error': 'Image file and color (comma-separated RGB) required.'}), 400

    file = request.files['image']
    color_str = request.form['color']  # e.g., "255,0,0"
    try:
        target_color = tuple(map(int, color_str.split(',')))
        image_np = load_image_from_bytes(file.read())
        sofa_only = isolate_sofa(image_np)
        recolored = apply_color_cv(sofa_only, target_color)
        return send_file(image_to_bytes(recolored), mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
