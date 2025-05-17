import tensorflow as tf
import numpy as np
import cv2
import os
import sys

# Load model
model = tf.keras.models.load_model("plant_disease_mobilenetv2.h5")

# Automatically load class labels
def get_class_labels(path="dataset/train"):
    return sorted(os.listdir(path))

class_labels = get_class_labels()

# Preprocess image to match training setup
def preprocess_image(img_path, target_size=(128, 128)):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"❌ Could not read image: {img_path}")
    img = cv2.resize(img, target_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Predict class
def predict_disease(img_path):
    processed = preprocess_image(img_path)
    prediction = model.predict(processed)
    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    predicted_label = class_labels[predicted_index]
    print(f"✅ Predicted Disease: {predicted_label} ({confidence:.2f}%)")
    return predicted_label, confidence

# Entry point
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.isfile(image_path):
        print(f"❌ File not found: {image_path}")
        sys.exit(1)

    predict_disease(image_path)
