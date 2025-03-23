import argparse
import os

import numpy as np
import onnxruntime as ort
import tensorflow as tf
from PIL import Image, ImageOps

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Predict a digit from an image using a model.')
    parser.add_argument('--model', type=str, required=True, help='Name of the model file to use (with extension .keras, .onnx, or .tflite)')
    parser.add_argument('--input_file', type=str, required=True, help='Input file to predict')
    return parser.parse_args()

def load_image(image_path):
    """Load and preprocess the input image for prediction."""
    image = Image.open(image_path)
    img_resized = image.resize((28, 28))
    img_inverted = ImageOps.invert(img_resized.convert("L"))
    img_array = np.array(img_inverted)
    img_array = img_array.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 28, 28)
    img_array = np.expand_dims(img_array, axis=-1)  # (1, 28, 28, 1)
    return img_array

def predict_with_keras(model, img_array):
    """Predict using a Keras model."""
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    return predicted_class, prediction

def predict_with_onnx(session, img_array):
    """Predict using an ONNX model."""
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: img_array})
    predicted_class = np.argmax(result[0])
    return predicted_class, result

def predict_with_tflite(interpreter, img_array):
    """Predict using a TFLite model."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data)
    return predicted_class, output_data

def predict_digit(model_name, img_array):
    """Predict the digit from the input image using the specified model."""
    model_ext = os.path.splitext(model_name)[1]
    if model_ext == '.keras':
        print("Using TensorFlow for prediction.")
        model = tf.keras.models.load_model(model_name)
        predicted_class, prediction = predict_with_keras(model, img_array)
    elif model_ext == '.onnx':
        print("Using ONNX Runtime for prediction.")
        session = ort.InferenceSession(model_name)
        predicted_class, prediction = predict_with_onnx(session, img_array)
    elif model_ext == '.tflite':
        print("Using TFLite for prediction.")
        interpreter = tf.lite.Interpreter(model_path=model_name)
        interpreter.allocate_tensors()
        predicted_class, prediction = predict_with_tflite(interpreter, img_array)
    else:
        raise ValueError("Unsupported model format. Please provide a .keras, .onnx, or .tflite model.")
    
    return predicted_class, prediction

def main():
    args = parse_arguments()
    img_array = load_image(args.input_file)
    predicted_class, prediction = predict_digit(args.model, img_array)
    print(f"Predicted class: {predicted_class} with confidence: {prediction[0][predicted_class]}")

if __name__ == "__main__":
    main()