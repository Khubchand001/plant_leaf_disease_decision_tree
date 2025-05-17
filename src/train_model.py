import os
import cv2
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from joblib import dump
from tqdm import tqdm

def load_images_and_labels(dataset_path, image_size=(64, 64)):
    X, y = [], []
    class_names = os.listdir(dataset_path)

    for label in tqdm(class_names, desc="Loading classes"):
        folder_path = os.path.join(dataset_path, label)
        if not os.path.isdir(folder_path):
            continue

        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            if not os.path.isfile(img_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, image_size)
            gray = np.mean(img, axis=-1) / 255.0
            X.append(gray.reshape(-1))
            y.append(label)

    return np.array(X), np.array(y)

# Load training and validation sets
X_train, y_train = load_images_and_labels("dataset/train")
X_val, y_val = load_images_and_labels("dataset/val")

# Encode labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded = le.transform(y_val)

# Train model
clf = DecisionTreeClassifier(criterion='entropy', max_depth=10)
clf.fit(X_train, y_train_encoded)

# Evaluate
y_pred = clf.predict(X_val)
acc = accuracy_score(y_val_encoded, y_pred)
print(f"✅ Validation Accuracy: {acc * 100:.2f}%")

# Save model and label encoder
dump(clf, "decision_tree_model.joblib")
dump(le, "label_encoder.joblib")
