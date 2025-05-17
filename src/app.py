import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Load model
model = tf.keras.models.load_model("plant_disease_mobilenetv2.h5")

# Automatically load class labels from training directory
def get_class_labels(path="dataset/train"):
    return sorted(os.listdir(path))

class_labels = get_class_labels()

# Image preprocessing
def preprocess_image(image, target_size=(128, 128)):
    img = image.resize(target_size)
    img = np.array(img.convert("RGB")) / 255.0
    return img.reshape(1, *target_size, 3)

# Streamlit UI
st.set_page_config(page_title="🌿 Plant Leaf Disease Detector", layout="centered")
st.title("🌿 Plant Leaf Disease Detection ")
st.write("Upload a plant leaf image to predict the disease.")

uploaded_file = st.file_uploader("📷 Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    try:
        input_tensor = preprocess_image(image)
        prediction = model.predict(input_tensor)
        predicted_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        predicted_label = class_labels[predicted_index]

        st.success(f"✅ Prediction: **{predicted_label}** ({confidence:.2f}%)")

        # Optional: Show confidence bar chart
        st.subheader("📊 Confidence per Class")
        confidence_dict = {class_labels[i]: float(prediction[0][i]) for i in range(len(class_labels))}
        st.bar_chart(confidence_dict)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
st.write("Made with ❤️ by khubchand")

