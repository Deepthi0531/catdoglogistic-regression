import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

def create_sample_model():
    """Create a simple sample model for demonstration purposes.
    
    This is NOT a properly trained model and should only be used for testing
    the application functionality. For real use, replace with a properly trained model.
    """
    print("Creating a sample model for demonstration purposes...")
    print("NOTE: This is NOT a properly trained model and will give random results.")
    print("For real use, replace with a properly trained model.")
    
    # Create a base model from MobileNetV2 without the top layers
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
    
    # Add custom layers for binary classification
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification (cat or dog)
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Save the model
    model.save('model.h5')
    print("\nSample model saved as 'model.h5'")
    print("You can now run the Flask application with 'python app.py'")

if __name__ == "__main__":
    try:
        from tensorflow.keras.layers import GlobalAveragePooling2D
        create_sample_model()
    except ImportError as e:
        print(f"Error: {e}")
        print("\nPlease install the required dependencies first:")
        print("pip install -r requirements.txt")
    except Exception as e:
        print(f"Error creating sample model: {e}")