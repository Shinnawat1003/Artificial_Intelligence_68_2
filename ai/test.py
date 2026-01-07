import tensorflow as tf
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(BASE_DIR, "test")

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

print("Current file location:", BASE_DIR)
print("Looking for test folder at:", TEST_DIR)
print("Test folder exists:", os.path.exists(TEST_DIR))

# โหลดโมเดล
model = tf.keras.models.load_model("object_classifier.h5")

# โหลด test dataset (grayscale ให้ตรงกับ train)
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    shuffle=False
)

# Normalize
test_ds = test_ds.map(lambda x, y: (x / 255.0, y))

# ===== คำนวณ accuracy เอง =====
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

accuracy = np.mean(y_true == y_pred)

print("Test accuracy:", accuracy)
