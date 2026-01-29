import os
import cv2
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

# Render-safe folder creation
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# MRI check
def is_mri_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    return edge_density > 0.02

# Rule-based prediction
def predict_tumor(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mean = np.mean(gray)
    std = np.std(gray)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

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

            if not is_mri_image(path):
                result = "Invalid Image"
                message = "Please upload MRI scan"
            else:
                result = predict_tumor(path)
                message = "Prediction completed"

    return render_template(
        "index.html",
        result=result,
        img_path=img_path,
        message=message
    )

if __name__ == "__main__":
    app.run()
