from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def gamma_transformation(image, gamma, c=1):
    image_normalised = image / 255.0
    gamma_image = c * np.power(image_normalised, gamma)
    gamma_image = (gamma_image / np.max(gamma_image) * 255.0)
    return gamma_image.astype(np.uint8)

def negative(image):
    return 255 - image

def contrast_stretching(image):
    min_val, max_val = np.min(image), np.max(image)
    return ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

def log_transformation(image, c=1):
    return (c * np.log1p(image)).astype(np.uint8)

def equalized_histogram(image):
    return cv2.equalizeHist(image)

def resize_image(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def save_image_to_file(image, filename="processed_image.png"):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cv2.imwrite(file_path, image)
    return filename  # Return only the filename

def image_to_base64(image):
    _, buffer = cv2.imencode('.png', image)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return img_str

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    operation = request.form.get('operation')

    if operation == 'Negative':
        result_image = negative(image)
    elif operation == 'Contrast Stretching':
        result_image = contrast_stretching(image)
    elif operation == 'Resized':
        width = int(request.form.get('width'))
        height = int(request.form.get('height'))
        result_image = resize_image(image, width, height)
    elif operation == 'Equalized Histogram':
        result_image = equalized_histogram(image)
    elif operation == 'Log Transformation':
        result_image = log_transformation(image)
    elif operation == 'Gamma Transformation':
        gamma = float(request.form.get('gamma'))
        result_image = gamma_transformation(image, gamma)
    else:
        result_image = image

    # Save the processed image to a file for download
    processed_filename = save_image_to_file(result_image)

    # Convert result image to base64
    img_base64 = image_to_base64(result_image)

    return render_template('result.html', img_data=img_base64, operation=operation, file_path=processed_filename)


@app.route('/download/<filename>')
def download_image(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return "File not found!", 404


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the upload folder exists
    app.run(debug=True)
