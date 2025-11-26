import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import json
import numpy as np
import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
MODEL_PATH = os.path.join(BASE_DIR, "model", "plant_disease_model.h5")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "test")
CLASS_INDICES_PATH = os.path.join(BASE_DIR, "ml", "class_indices.json")

# --- Load model and class labels ---
print("📦 Loading model and class indices...")
model = load_model(MODEL_PATH)
with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}

# --- Data generator for test images ---
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_gen = test_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# --- Predict on test set ---
print("🔍 Evaluating model...")
Y_pred = model.predict(test_gen)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_gen.classes

# --- Metrics ---
print("\n✅ Evaluation Results:")
print(classification_report(y_true, y_pred, target_names=list(idx_to_class.values())))

cm = confusion_matrix(y_true, y_pred)
print("\n🧩 Confusion Matrix:\n", cm)
