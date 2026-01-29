import os
import cv2
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def fake_predict(img_path):
    img = cv2.imread(img_path, 0)
    mean = np.mean(img)

    if mean < 60:
        return "Glioma"
    elif mean < 100:
        return "Meningioma"
    elif mean < 140:
        return "Pituitary"
    else:
        return "No Tumor"

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_path = None

    if request.method == "POST":
        if "file" not in request.files:
            return "No file sent"

        file = request.files["file"]

        if file.filename == "":
            return "No file selected"

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        result = fake_predict(file_path)
        image_path = "/" + file_path

    return render_template("index.html", result=result, image_path=image_path)

if __name__ == "__main__":
    app.run()
