# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# import numpy as np
# import os
# import traceback
# import time

# from ela import convert_to_ela_image
# from preprocess import preprocess_image
# from gemini_helper import gemini_image_analysis

# app = Flask(__name__)
# CORS(app)

# # Load your pre-trained model
# MODEL_PATH = "forgery_model.h5"
# model = load_model(MODEL_PATH)

# @app.route("/")
# def home():
#     return "Hybrid Image Forgery Detection API Running"

# @app.route("/predict", methods=["POST"])
# def predict():
#     img_path = "temp_input.jpg"

#     try:
#         if "image" not in request.files:
#             return jsonify({"error": "No image uploaded"}), 400

#         file = request.files["image"]
#         file.save(img_path)

#         # 1. CNN + ELA Analysis
#         ela_img = convert_to_ela_image(img_path)
#         processed_img = preprocess_image(ela_img)
#         processed_img = img_to_array(processed_img)
#         processed_img = np.expand_dims(processed_img, axis=0)

#         cnn_score = float(model.predict(processed_img)[0][0])
#         cnn_label = "Fake" if cnn_score > 0.5 else "Authentic"

#         # 2. Gemini Semantic Analysis
#         try:
#             gemini_label, gemini_reason = gemini_image_analysis(img_path)
#         except Exception:
#             gemini_label = "Unavailable"
#             gemini_reason = "Gemini API connection error or quota exceeded."

#         # 3. Hybrid Decision Logic
#         final_label = "Fake" if (cnn_label == "Fake" or gemini_label == "Fake") else "Authentic"

#         result = {
#             "final_prediction": final_label,
#             "cnn_prediction": cnn_label,
#             "cnn_confidence": round(cnn_score, 4),
#             "gemini_prediction": gemini_label,
#             "gemini_reason": gemini_reason[:300]
#         }

#         print("🔍 BACKEND LOG →", result)
#         return jsonify(result)

#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({"error": str(e)}), 500
#     finally:
#         time.sleep(0.2)
#         if os.path.exists(img_path):
#             try:
#                 os.remove(img_path)
#             except PermissionError: pass

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)


from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import os
import traceback
import tempfile

from ela import convert_to_ela_image
from preprocess import preprocess_image
from gemini_helper import gemini_image_analysis

app = Flask(__name__)
CORS(app)

# ---------------- LOAD MODEL SAFELY ----------------
MODEL_PATH = "forgery_model.h5"
model = None

def load_cnn_model():
    global model
    if model is None:
        model = load_model(MODEL_PATH)

# ---------------- ROUTES ----------------

@app.route("/")
def home():
    return "Hybrid Image Forgery Detection API Running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        load_cnn_model()  # load model only once

        file = request.files["image"]

        # Create temp file (cloud-safe)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            file.save(temp.name)
            img_path = temp.name

        # -------- CNN + ELA --------
        ela_img = convert_to_ela_image(img_path)
        processed_img = preprocess_image(ela_img)
        processed_img = img_to_array(processed_img)
        processed_img = np.expand_dims(processed_img, axis=0)

        cnn_score = float(model.predict(processed_img)[0][0])
        cnn_label = "Fake" if cnn_score > 0.5 else "Authentic"

        # -------- Gemini --------
        try:
            gemini_label, gemini_reason = gemini_image_analysis(img_path)
        except Exception:
            gemini_label = "Unavailable"
            gemini_reason = "Gemini API unavailable or quota exceeded."

        # -------- Final Decision --------
        final_label = "Fake" if (cnn_label == "Fake" or gemini_label == "Fake") else "Authentic"

        result = {
            "final_prediction": final_label,
            "cnn_prediction": cnn_label,
            "cnn_confidence": round(cnn_score, 4),
            "gemini_prediction": gemini_label,
            "gemini_reason": gemini_reason[:300]
        }

        print("🔍 BACKEND LOG →", result)
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        try:
            if img_path and os.path.exists(img_path):
                os.remove(img_path)
        except Exception:
            pass


# ---------------- RENDER ENTRY POINT ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
