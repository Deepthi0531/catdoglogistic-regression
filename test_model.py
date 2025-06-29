import os
import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def test_model_loading():
    """Test if the model can be loaded successfully"""
    try:
        model_path = 'model.h5'
        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' not found.")
            print("Please place your trained model file in the root directory.")
            return False
        
        model = load_model(model_path)
        print("Model loaded successfully!")
        print(f"Model summary:\n{model.summary()}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def test_prediction_pipeline():
    """Test the prediction pipeline with a sample image if available"""
    # Check if model loads first
    if not test_model_loading():
        return
    
    # Look for sample images in the uploads directory
    upload_dir = 'uploads'
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
        print(f"\nCreated '{upload_dir}' directory. Please place sample images there for testing.")
        return
    
    # Find any image files
    image_files = [f for f in os.listdir(upload_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"\nNo sample images found in '{upload_dir}' directory.")
        print("Please add some .jpg, .jpeg, or .png files for testing.")
        return
    
    # Test with the first image found
    test_image = os.path.join(upload_dir, image_files[0])
    print(f"\nTesting prediction with image: {test_image}")
    
    try:
        # Load model
        model = load_model('model.h5')
        
        # Preprocess image
        img = image.load_img(test_image, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Make prediction
        prediction = model.predict(img_array)
        result = 'Dog' if prediction[0][0] > 0.5 else 'Cat'
        confidence = float(prediction[0][0]) if result == 'Dog' else float(1 - prediction[0][0])
        confidence = round(confidence * 100, 2)
        
        print(f"Prediction: {result} (Confidence: {confidence}%)")
        print("Prediction pipeline is working correctly!")
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    print("===== Testing Cat/Dog Classifier Model =====\n")
    test_prediction_pipeline()