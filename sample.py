from flask import Flask, render_template, request, redirect, url_for, send_from_directory, render_template_string
import os
import torch
from paddleocr import PaddleOCR
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
import base64
import re
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize PaddleOCR
ocr = PaddleOCR(lang='en')

# Load .h5 model
fruit_model = load_model('DenseNet20_model.h5')

# Class names for fruit freshness classification
class_names = {
    0: 'Banana_Bad',
    1: 'Banana_Good',
    2: 'Fresh',
    3: 'FreshCarrot',
    4: 'FreshCucumber',
    5: 'FreshMango',
    6: 'FreshTomato',
    7: 'Guava_Bad',
    8: 'Guava_Good',
    9: 'Lime_Bad',
    10: 'Lime_Good',
    11: 'Rotten',
    12: 'RottenCarrot',
    13: 'RottenCucumber',
    14: 'RottenMango',
    15: 'RottenTomato',
    16: 'freshBread',
    17: 'rottenBread'
}

# Helper function: Preprocess image for fruit classification
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Helper function: Extract expiry date from text
def extract_expiry_date(text):
    expiry_date_patterns = [
      
        r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Expiry Date: 20/07/2O24
    r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-]\d{4}))',  # Expiry Date: 20/07/2024
    r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Expiry Date: 20/07/2O24
    r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Expiry Date: 20 MAY 2O24
    r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*\d{4}))',  # Expiry Date: 20 MAY 2024
    r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Expiry Date: 20 MAY 2O24
    r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Expiry Date: 2024/07/2O24
    r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-]\d{2}))',  # Expiry Date: 2024/07/20
    r'(?:best\s*before\s*[:\-]?\s*.*?(\d{4}))',  # Best Before: 2025
    r'(?:best\s*before\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Best Before: 20/07/2O24
    r'(?:best\s*before\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-]\d{4}))',  # Best Before: 20/07/2024
    r'(?:best\s*before\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Best Before: 20/07/2O24
    r'(?:best\s*before\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Best Before: 20 MAY 2O24
    r'(?:best\s*before\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*\d{4}))',  # Best Before: 20 MAY 2024
    r'(?:best\s*before\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Best Before: 20 MAY 2O24
    r'(?:best\s*before\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Best Before: 2024/07/2O24
    r'(?:best\s*before\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-]\d{2}))',  # Best Before: 2024/07/20
    r'(?:best\s*before\s*[:\-]?\s*.*?(\d{1,2}\d{2}\d{2}))', 
    r'(?:best\s*before\s*[:\-]?\s*(\d{6}))',
    r'(?:consume\s*before\s*[:\-]?\s*.*?(\d{1,2}[A-Za-z]{3,}[0O]\d{2}))',  # Consume Before: 3ODEC2O24
    r'(?:consume\s*before\s*[:\-]?\s*.*?(\d{1,2}[A-Za-z]{3,}\d{2}))',  # Consume Before: 30DEC23
    r'(?:consume\s*before\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Consume Before: 20/07/2O24
    r'(?:consume\s*before\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-]\d{4}))',  # Consume Before: 20/07/2024
    r'(?:consume\s*before\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Consume Before: 20/07/2O24
    r'(?:consume\s*before\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Consume Before: 20 MAY 2O24
    r'(?:consume\s*before\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*\d{4}))',  # Consume Before: 20 MAY 2024
    r'(?:consume\s*before\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Consume Before: 20 MAY 2O24
    r'(?:consume\s*before\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Consume Before: 2024/07/2O24
    r'(?:consume\s*before\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-]\d{2}))',  # Consume Before: 2024/07/20
    r'(?:exp\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Exp: 20/07/2O24
    r'(?:exp\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-]\d{4}))',  # Exp: 20/07/2024
    r'(?:exp\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Exp: 20/07/2O24
    r'(?:exp\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Exp: 20 MAY 2O24
    r'(?:exp\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*\d{4}))',  # Exp: 20 MAY 2024
    r'(?:exp\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Exp: 20 MAY 2O24
    r'(?:exp\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Exp: 2024/07/2O24
    r'(?:exp\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-]\d{2}))',  # Exp: 2024/07/20
    r"Exp\.Date\s+(\d{2}[A-Z]{3}\d{4})",
    r'(?:exp\s*\.?\s*date\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Exp. Date: 16 MAR 2O30 (with typo)
    r'(?:exp\s*\.?\s*date\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Exp. Date: 15/12/2O30 (with typo)
    r'(?:exp\s*\.?\s*date\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Exp. Date: 15 MAR 2O30 (with typo)
    r'(?:exp\s*\.?\s*date\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Exp. Date cdsyubfuyef 15 MAR 2O30 (with typo)
    r'(\d{2}[\/\-]\d{2}[\/\-]\d{4})',  # 20/07/2024
    r'(\d{2}[\/\-]\d{2}[\/\-]\d{2})',  # 20/07/24
    r'(\d{2}\s*[A-Za-z]{3,}\s*\d{4})',  # 20 MAY 2024
    r'(\d{2}\s*[A-Za-z]{3,}\s*\d{2})',  # 20 MAY 24
    r'(\d{4}[\/\-]\d{2}[\/\-]\d{2})',  # 2024/07/20
    r'(\d{4}[\/\-]\d{2}[\/\-]\d{2})',  # 2024-07-20
    r'(\d{4}[A-Za-z]{3,}\d{2})',  # 2024MAY20
    r'(\d{2}[A-Za-z]{3,}\d{4})',  # 20MAY2024
    r'(?:DX3\s*[:\-]?\s*(\d{2}\s*\d{2}\s*\d{4}))',
    r'(?:exp\.?\s*date\s*[:\-]?\s*(\d{2}\s*[A-Za-z]{3,}\s*(\d{4}|\d{2})))',
    r'(?:exp\.?\s*date\s*[:\-]?\s*(\d{2}\s*\d{2}\s*\d{4}))',  # Exp. Date: 20 05 2025
    r'(\d{4}[A-Za-z]{3}\d{2})',  # 2025MAY11
    r'(?:best\s*before\s*[:\-]?\s*(\d+)\s*(days?|months?|years?))',  # Best Before: 6 months
    r'(?:best\s*before\s*[:\-]?\s*(three)\s*(months?))',
    r'(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b\s*\d{4})',
    r'\bUSE BY\s+(\d{1,2}[A-Za-z]{3}\d{4})\b',
    r"Exp\.Date\s*(\d{2}[A-Z]{3}\d{4})",
    r"EXP:\d{4}/\d{2}/\d{4}/\d{1}/[A-Z]"
    ]
    for pattern in expiry_date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None

# Route: Home
@app.route('/')
def index():
    return render_template_string('''
    <html>
        <head>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    margin: 0;
                    padding: 0;
                }

                .container {
                    width: 80%;
                    margin: auto;
                    padding: 20px;
                    background-color: white;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                    margin-top: 30px;
                }

                h1 {
                    color: #333;
                    text-align: center;
                }

                .links {
                    text-align: center;
                }

                .links a {
                    display: inline-block;
                    margin: 10px 15px;
                    padding: 10px 20px;
                    background-color: #337ab7;
                    color: white;
                    text-decoration: none;
                    border-radius: 5px;
                }

                .links a:hover {
                    background-color: #286090;
                }
            </style>
            <title>Integrated Application</title>
        </head>
        <body>
            <div class="container">
                <h1>Integrated Application</h1>
                <div class="links">
                    <a href="/object_detection">Object Detection</a>
                    <a href="/text_extraction">Text Extraction</a>
                    <a href="/upload">Fruit Freshness Prediction</a>
                </div>
            </div>
        </body>
    </html>
    ''')

# Route: Object Detection
@app.route('/object_detection', methods=['GET', 'POST'])
def object_detection():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(image_path)

            # Perform object detection
            results = yolo_model(image_path)
            results.render()

            # Get the confidence scores
            predictions = results.pandas().xywh[0]
            predictions['confidence'] = predictions['confidence'].round(2)  # Round confidence to 2 decimal places

            # Save output image
            output_image = results.ims[0]
            output_path = os.path.join('static', f'detected_{image_file.filename}')
            cv2.imwrite(output_path, output_image)

            return render_template_string('''
            <html>
                <head>
                    <style>
                        body {
                            font-family: Arial, sans-serif;
                            background-color: #f4f4f4;
                            margin: 0;
                            padding: 0;
                        }

                        .container {
                            width: 80%;
                            margin: auto;
                            padding: 20px;
                            background-color: white;
                            border-radius: 10px;
                            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                            margin-top: 30px;
                        }

                        h1 {
                            color: #333;
                            text-align: center;
                        }

                        img {
                            max-width: 100%;
                            height: auto;
                            display: block;
                            margin: 0 auto;
                        }

                        .confidence {
                            margin-top: 20px;
                            font-size: 1.1em;
                            text-align: center;
                        }

                        .home-link {
                            text-align: center;
                            margin-top: 20px;
                        }

                        .home-link a {
                            color: #337ab7;
                            text-decoration: none;
                        }
                    </style>
                    <title>Object Detection Result</title>
                </head>
                <body>
                    <div class="container">
                        <h1>Object Detection Result</h1>
                        <img src="{{ url_for('static', filename='detected_' + filename) }}" alt="Detected Image">
                        <div class="confidence">
                            <p>Confidence Scores:</p>
                            <ul>
                                {% for index, row in predictions.iterrows() %}
                                    <li>{{ row['name'] }}: {{ row['confidence'] * 100 }}%</li>
                                {% endfor %}
                            </ul>
                        </div>
                        <div class="home-link"><a href="/">Home</a></div>
                    </div>
                </body>
            </html>
            ''', filename=image_file.filename, predictions=predictions)
    return '''
    <html>
        <head>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    margin: 0;
                    padding: 0;
                }

                .container {
                    width: 80%;
                    margin: auto;
                    padding: 20px;
                    background-color: white;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                    margin-top: 30px;
                }

                h1 {
                    color: #333;
                    text-align: center;
                }

                form {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    margin-top: 20px;
                }

                input[type="file"] {
                    margin-right: 20px;
                }

                button {
                    background-color: #5cb85c;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    cursor: pointer;
                    border-radius: 5px;
                }

                button:hover {
                    background-color: #4cae4c;
                }
            </style>
            <title>Object Detection</title>
        </head>
        <body>
            <div class="container">
                <h1>Upload Image for Object Detection</h1>
                <form method="POST" enctype="multipart/form-data">
                    <input type="file" name="image" required>
                    <button type="submit">Detect</button>
                </form>
            </div>
        </body>
    </html>
    '''


# Route: Text Extraction
@app.route('/text_extraction', methods=['GET', 'POST'])
def text_extraction():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(image_path)

            # Extract text
            result = ocr.ocr(image_path)
            text = ' '.join([line[1][0] for line in result[0]])
            expiry_date = extract_expiry_date(text)

            return render_template_string('''
            <html>
                <head>
                    <style>
                        body {
                            font-family: Arial, sans-serif;
                            background-color: #f4f4f4;
                            margin: 0;
                            padding: 0;
                        }

                        .container {
                            width: 80%;
                            margin: auto;
                            padding: 20px;
                            background-color: white;
                            border-radius: 10px;
                            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                            margin-top: 30px;
                        }

                        h1 {
                            color: #333;
                            text-align: center;
                        }

                        .home-link {
                            text-align: center;
                            margin-top: 20px;
                        }

                        .home-link a {
                            color: #337ab7;
                            text-decoration: none;
                        }
                    </style>
                    <title>Text Extraction Result</title>
                </head>
                <body>
                    <div class="container">
                        <h1>Text Extraction Result</h1>
                        <p>Extracted Text: {{ text }}</p>
                        <p>Expiry Date: {{ expiry_date }}</p>
                        <div class="home-link"><a href="/">Home</a></div>
                    </div>
                </body>
            </html>
            ''', text=text, expiry_date=expiry_date)
    return '''
    <html>
        <head>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    margin: 0;
                    padding: 0;
                }

                .container {
                    width: 80%;
                    margin: auto;
                    padding: 20px;
                    background-color: white;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                    margin-top: 30px;
                }

                h1 {
                    color: #333;
                    text-align: center;
                }

                form {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    margin-top: 20px;
                }

                input[type="file"] {
                    margin-right: 20px;
                }

                button {
                    background-color: #5cb85c;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    cursor: pointer;
                    border-radius: 5px;
                }

                button:hover {
                    background-color: #4cae4c;
                }
            </style>
            <title>Text Extraction</title>
        </head>
        <body>
            <div class="container">
                <h1>Upload Image for Text Extraction</h1>
                <form method="POST" enctype="multipart/form-data">
                    <input type="file" name="image" required>
                    <button type="submit">Extract</button>
                </form>
            </div>
        </body>
    </html>
    '''

# Route: Fruit Freshness Prediction
# Route: Fruit Freshness Prediction
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(image_path)

            # Perform prediction
            img_array = preprocess_image(image_path)
            predictions = fruit_model.predict(img_array)

            # Get predicted class and confidence score
            predicted_class = np.argmax(predictions, axis=1)[0]
            predicted_label = class_names[predicted_class]
            confidence_score = predictions[0][predicted_class] * 100  # Confidence as a percentage

            # Render the result
            return render_template_string('''
            <html>
                <head>
                    <style>
                        body {
                            font-family: Arial, sans-serif;
                            background-color: #f4f4f4;
                            margin: 0;
                            padding: 0;
                        }

                        .container {
                            width: 80%;
                            margin: auto;
                            padding: 20px;
                            background-color: white;
                            border-radius: 10px;
                            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                            margin-top: 30px;
                        }

                        h1 {
                            color: #333;
                            text-align: center;
                        }

                        .home-link {
                            text-align: center;
                            margin-top: 20px;
                        }

                        .home-link a {
                            color: #337ab7;
                            text-decoration: none;
                        }

                        .image-container {
                            text-align: center;
                        }

                        img {
                            max-width: 100%;
                            height: auto;
                            display: inline-block;
                            margin-bottom: 20px;
                        }

                        .prediction-result {
                            text-align: center;
                            margin-top: 20px;
                        }

                        .confidence {
                            font-size: 1.2em;
                            color: #555;
                        }

                        .result {
                            font-size: 1.5em;
                            color: #337ab7;
                            font-weight: bold;
                        }
                    </style>
                    <title>Fruit Freshness Prediction</title>
                </head>
                <body>
                    <div class="container">
                        <h1>Fruit Freshness Prediction</h1>
                        
                        <div class="image-container">
                            <img src="/uploads/{{ filename }}" alt="Uploaded Image">
                        </div>

                        <div class="prediction-result">
                            <p class="result">Predicted Label: {{ label }}</p>
                            <p class="confidence">Confidence Score: {{ confidence }}%</p>
                        </div>

                        <div class="home-link"><a href="/">Home</a></div>
                    </div>
                </body>
            </html>
            ''', label=predicted_label, confidence=round(confidence_score, 2), filename=image_file.filename)
    
    return '''
    <html>
        <head>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    margin: 0;
                    padding: 0;
                }

                .container {
                    width: 80%;
                    margin: auto;
                    padding: 20px;
                    background-color: white;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                    margin-top: 30px;
                }

                h1 {
                    color: #333;
                    text-align: center;
                }

                form {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    margin-top: 20px;
                }

                input[type="file"] {
                    margin-right: 20px;
                }

                button {
                    background-color: #5cb85c;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    cursor: pointer;
                    border-radius: 5px;
                }

                button:hover {
                    background-color: #4cae4c;
                }
            </style>
            <title>Fruit Freshness Prediction</title>
        </head>
        <body>
            <div class="container">
                <h1>Upload Image for Freshness Prediction</h1>
                <form method="POST" enctype="multipart/form-data">
                    <input type="file" name="image" required>
                    <button type="submit">Predict</button>
                </form>
            </div>
        </body>
    </html>
    '''
@app.route('/uploads/<filename>' )
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=8000)
