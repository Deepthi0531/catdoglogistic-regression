# Cat or Dog Image Classifier

A Flask web application that uses a trained Logistic Regression model to classify uploaded images as either a cat or a dog.

## Features

- Upload images for classification
- Real-time prediction using a pre-trained model
- Modern UI with Tailwind CSS
- Responsive design for all devices

## Prerequisites

- Python 3.7 or higher
- Pip package manager
- A trained model file named `model.h5` (place it in the root directory)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/catdoglogistic-regression.git
   cd catdoglogistic-regression
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Make sure you have the model file `model.h5` in the root directory of the project.

## Usage

1. Start the Flask application:
   ```
   python app.py
   ```

2. Open your web browser and navigate to `http://127.0.0.1:5000/`

3. Upload an image of a cat or dog and click "Classify Image"

4. View the prediction result

## Project Structure

```
├── app.py                 # Main Flask application
├── model.h5               # Pre-trained model (you need to provide this)
├── requirements.txt       # Python dependencies
├── templates/             # HTML templates
│   └── index.html         # Main page template
└── uploads/               # Directory for uploaded images (created automatically)
```

## Model Information

The application expects a trained Keras model saved as an H5 file. The model should be designed to classify images as either cats or dogs, with an output of:
- Values closer to 0 indicating a cat
- Values closer to 1 indicating a dog

## Customization

You can customize the application by:

- Modifying the UI in `templates/index.html`
- Adjusting the image preprocessing in `app.py`
- Changing the prediction threshold in the `/predict` route

## License

This project is licensed under the MIT License - see the LICENSE file for details.