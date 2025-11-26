import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import json
import os

# ============================================================
# 📁 1️⃣ Paths and Parameters
# ============================================================
BASE_DIR = "dataset"
train_dir = os.path.join(BASE_DIR, "train")
val_dir = os.path.join(BASE_DIR, "val")

IMG_SIZE = (128, 128)
BATCH = 32
EPOCHS = 25

# ============================================================
# 🌿 2️⃣ Data Augmentation
# ============================================================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="categorical"
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="categorical"
)

num_classes = len(train_gen.class_indices)

# Save class indices for later reference
with open("ml/class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f, indent=4)

# ============================================================
# 🧠 3️⃣ Model Creation — MobileNetV2 (Transfer Learning)
# ============================================================
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Freeze pretrained layers

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])

# ============================================================
# ⚙️ 4️⃣ Compile Model
# ============================================================
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ============================================================
# 🚀 5️⃣ Callbacks
# ============================================================
checkpoint = ModelCheckpoint("ml/best_model.keras", monitor="val_accuracy", save_best_only=True, mode="max")
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-5)

# ============================================================
# 📈 6️⃣ Train Model
# ============================================================
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# ============================================================
# 💾 7️⃣ Save Final Model
# ============================================================
os.makedirs("model", exist_ok=True)
model.save("model/plant_disease_model.h5")

print("\n✅ Training completed successfully! Model saved at 'model/plant_disease_model.h5'")
