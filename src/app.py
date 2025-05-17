import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load trained model
model = tf.keras.models.load_model("plant_disease_mobilenetv2.h5")

# Load class labels
def get_class_labels(path="dataset/train"):
    return sorted(os.listdir(path))

class_labels = get_class_labels()

# Preprocess uploaded image
def preprocess_image(image, target_size=(128, 128)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# Page setup
st.set_page_config(page_title="🌿 Leaf Disease Detection", layout="wide")

# CSS for improved UI: moderate image size and subtle animations for results
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

        body, .main {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #d0f0c0, #a8dda8);
            color: #1b3a1a;
            min-height: 100vh;
            margin: 0;
            padding: 2rem 4rem;
        }

        h1 {
            font-size: 3rem;
            font-weight: 700;
            color: #2e5724;
            text-align: center;
            letter-spacing: 0.06em;
            user-select: none;
            margin-bottom: 0.5rem;
            text-shadow: 0 2px 3px rgba(162, 224, 166, 0.7);
        }

        p {
            font-size: 1.3rem;
            color: #b7fba7;
            text-align: center;
            margin-top: 0;
            margin-bottom: 2.5rem;
            font-weight: 600;
            line-height: 1.5;
        }

        h2 {
            font-size: 2rem;
            font-weight: 700;
            color: #3a6f34;
            margin-top: 3rem;
            margin-bottom: 1rem;
            border-bottom: 3px solid #7dc87d;
            padding-bottom: 0.3rem;
            user-select: none;
        }

        /* Adapt uploader */
        .stFileUploader > label {
            font-weight: 700 !important;
            font-size: 1.3rem !important;
            color: #2e5724 !important;
            user-select: none;
        }

        .stFileUploader>div>div>button {
            background: #6bbb50 !important;
            color: white !important;
            font-weight: 700;
            padding: 0.7em 1.6em;
            border-radius: 16px;
            box-shadow: 0 6px 16px rgba(107, 187, 80, 0.75);
            transition: background 0.3s ease, box-shadow 0.3s ease;
            cursor: pointer;
        }
        .stFileUploader>div>div>button:hover {
            background: #4d7d32 !important;
            box-shadow: 0 10px 25px rgba(77, 125, 50, 0.85);
        }

        /* Limit image size */
        .stImage > img {
            border-radius: 28px;
            max-width: auto !important;
            max-height: 400px !important;
            width: auto !important;
            height: auto !important;
            box-shadow: 0 15px 44px rgba(46, 125, 23, 0.9);
            display: block;
            margin-left: auto;
            margin-right: auto;
            user-select: none;
            animation: fadeInImage 1s ease forwards;
            opacity: 0;
        }

        @keyframes fadeInImage {
            to {opacity: 1;}
        }

        /* Fade-in animation for results */
        .animated-fade-in {
            animation: fadeInUp 1s ease forwards;
            opacity: 0;
            transform: translateY(20px);
        }
        @keyframes fadeInUp {
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* Prediction texts fully opaque and bold with fade-in */
        .stSuccess {
            background-color: #def3d3 !important;
            border-left: 8px solid #4caf50 !important;
            border-radius: 16px;
            padding: 1.8em 2.2em;
            color: #1b3a1a !important;
            font-weight: 900;
            font-size: 1.5rem;
            box-shadow: 0 0 24px #70c470 !important;
            user-select: text;
            opacity: 1 !important;
            animation: fadeInUp 1.2s ease forwards;
        }

        .stError {
            background-color: #ffdada !important;
            border-left: 8px solid #d32f2f !important;
            border-radius: 16px;
            padding: 1.8em 2.2em;
            color: #6b1b1b !important;
            font-weight: 900;
            font-size: 1.4rem;
            box-shadow: 0 0 24px #d15151 !important;
            user-select: text;
            opacity: 1 !important;
            animation: fadeInUp 1.2s ease forwards;
        }

        .stInfo {
            background-color: #d5e9f7 !important;
            border-left: 8px solid #3178c6 !important;
            border-radius: 16px;
            padding: 1.8em 2.2em;
            color: #1f3f6a !important;
            font-weight: 900;
            font-size: 1.4rem;
            user-select: text;
            opacity: 1 !important;
            animation: fadeInUp 1.2s ease forwards;
        }

        /* Animate bar charts */
        .stBarChart {
            border-radius: 28px;
            background-color: #e6f4e8;
            padding: 1.8rem 2rem;
            margin-top: 2.2rem;
            margin-bottom: 3rem;
            box-shadow: 0 14px 42px rgba(46, 125, 23, 0.32);
            opacity: 0;
            animation: fadeInUp 1.5s ease forwards;
        }

        .stBarChart > div > div > svg text {
            fill: #2e5724 !important;
            font-weight: 800 !important;
            opacity: 1 !important;
        }

        section[data-testid="stHorizontalBlock"] {
            margin-top: 2rem;
            margin-bottom: 3rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div>
        <h1>🌿 Plant Leaf Disease Classifier</h1>
        <p>Upload or drag & drop a leaf image to identify the disease .</p>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=False)

    try:
        input_tensor = preprocess_image(image)
        prediction = model.predict(input_tensor)
        predicted_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        predicted_label = class_labels[predicted_index]

        st.markdown("### ✅ Prediction Result")
        st.success(f"**{predicted_label}** ({confidence:.2f}%)")

        st.markdown("### 📊 Prediction Confidence")
        confidence_dict = {class_labels[i]: float(prediction[0][i]) for i in range(len(class_labels))}
        st.bar_chart(confidence_dict)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
else:
    st.info("⬆️ Upload or drag and drop an image to begin.")

st.write("Made with ❤️ by khubchand")
st.markdown("""
    <footer>
        <p style="text-align: center; font-size: 0.9rem; color: #2e5724;">
            &copy; 2025 Plant Leaf Disease Classifier. All rights reserved.
        </p>
    </footer>
""", unsafe_allow_html=True)
