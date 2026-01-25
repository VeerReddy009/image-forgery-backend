import cv2
import numpy as np

def preprocess_image(image, size=(128, 128)):
    image = np.array(image)
    image = cv2.resize(image, size)
    image = image.astype("float32") / 255.0
    return image
