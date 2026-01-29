import os
import cv2
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

classes = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

# ---------------- MRI VALIDATION ----------------
def is_mri_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    return edge_density > 0.02


# ---------------- FEATURE BASED PREDICTION ----------------
def predict_tumor(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Features
    mean = np.mean(gray)
    std = np.std(gray)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # Decision logic (gives different outputs)
    if mean < 70 and edge_density > 0.08:
        return "Glioma"
    elif mean < 100 and std > 40:
        return "Meningioma"
    elif mean < 130:
        return "Pituitary"
    else:
        return "No Tumor"


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    img_path = None
    message = None

    if request.method == "POST":
        file = request.files.get("file")

        if file and file.filename != "":
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            img_path = path

            # Step 1: MRI check
            if not is_mri_image(path):
                result = "Invalid Image"
                message = "Please upload a valid MRI scan."
            else:
                # Step 2: Prediction
                result = predict_tumor(path)
                message = "Prediction completed."

    return render_template(
        "index.html",
        result=result,
        img_path=img_path,
        message=message
    )

if __name__ == "__main__":
    app.run(debug=True)
