import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Load the model
def load_classification_model():
    try:
        model = load_model('model.h5')
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Preprocess the image for prediction
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Load model
        model = load_classification_model()
        if model is None:
            return jsonify({'error': 'Failed to load model'}), 500
        
        # Preprocess image and make prediction
        try:
            processed_image = preprocess_image(file_path)
            prediction = model.predict(processed_image)
            
            # Get prediction result
            result = 'Dog' if prediction[0][0] > 0.5 else 'Cat'
            confidence = float(prediction[0][0]) if result == 'Dog' else float(1 - prediction[0][0])
            confidence = round(confidence * 100, 2)
            
            return jsonify({
                'result': result,
                'confidence': confidence,
                'filename': filename
            })
        except Exception as e:
            return jsonify({'error': f'Prediction error: {str(e)}'}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    app.run(debug=True)