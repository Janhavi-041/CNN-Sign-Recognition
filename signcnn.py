#.\asl_cnn_env\Scripts\Activate.ps1
"""
Sign Language Recognition using CNN
Dataset: dataset2 (ASL Kaggle Dataset)
Clean version (Pylance-safe)
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt

# ===================CONFIG=========================================
class Config:
    DATA_DIR = Path("dataset2")

    IMG_SIZE = (64, 64)
    BATCH_SIZE = 32
    EPOCHS = 15
    LR = 0.001


print("\n==============================")
print("ASL CNN TRAINING")
print("==============================")

# ============================================================
# LOAD DATA (FIXED FOR YOUR STRUCTURE)
# ============================================================
def load_data():
    print("\n[INFO] Loading dataset2 (A-Z + 0-9)...")

    ds = tf.keras.utils.image_dataset_from_directory(
        Config.DATA_DIR,
        label_mode="categorical",
        image_size=Config.IMG_SIZE,
        batch_size=Config.BATCH_SIZE,
        color_mode="grayscale",
        shuffle=True
    )

    class_names = ds.class_names

    print(f"[INFO] Classes found: {len(class_names)}")
    print(class_names)

    # Split dataset manually
    train_size = int(0.8 * len(ds))
    val_size = int(0.1 * len(ds))

    train_ds = ds.take(train_size)
    temp = ds.skip(train_size)

    val_ds = temp.take(val_size)
    test_ds = temp.skip(val_size)

    return train_ds, val_ds, test_ds, class_names


# ==================NORMALIZE==========================================
def normalize(ds):
    return ds.map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y))


# ==================Model==========================================
def build_model(num_classes):

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(64,64,1)),

        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(Config.LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ===================TRAIN=========================================
def train(model, train_ds, val_ds):

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=Config.EPOCHS
    )

    model.save("asl_model.keras")
    print("[INFO] Model saved as asl_model.keras") 

    return history


# ==================MAIN==========================================
def main():

    train_ds, val_ds, test_ds, class_names = load_data()

    train_ds = normalize(train_ds)
    val_ds = normalize(val_ds)
    test_ds = normalize(test_ds)

    model = build_model(len(class_names))

    history = train(model, train_ds, val_ds)

    print("\n[INFO] Training complete!")

    model.evaluate(test_ds)

    return model



if __name__ == "__main__":
    main()