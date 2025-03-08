from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# Load trained model
model = tf.keras.models.load_model("goku_model_v2.h5")

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read())).convert("L").resize((28, 28))
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)

    prediction = np.argmax(model.predict(image_array))
    return jsonify({"Predicted Number": int(prediction)})

# Ensure Render uses the correct port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Set default port to 10000
    app.run(host="0.0.0.0", port=port)
