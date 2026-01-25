import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split

from ela import convert_to_ela_image
from preprocess import preprocess_image
from model import build_model

X = []
y = []

# Load dataset
for label, folder in enumerate(["authentic", "tampered"]):
    folder_path = os.path.join("dataset", folder)

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        try:
            ela_img = convert_to_ela_image(img_path)
            img_array = img_to_array(ela_img)
            img_array = preprocess_image(img_array)

            X.append(img_array)
            y.append(label)
        except Exception as e:
            print(f"Skipping {img_name}: {e}")

X = np.array(X)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build and train model
model = build_model()
model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# Save trained model
model.save("forgery_model.h5")
print("✅ Model training complete. Saved as forgery_model.h5")
