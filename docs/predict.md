# Predicting Digits with the MNIST Model

This guide provides instructions for using the `predict.py` script to predict digits from images using a trained MNIST model.

## Prerequisites

Ensure you have the following installed:
- Python 3.x
- Required Python packages (listed in `requirements.txt`)

## Predicting Digits

To predict digits from images, run the `predict.py` script with the desired parameters.

### Basic Prediction

To predict a digit from an image using a saved model:

```sh
python3 predict.py --model path/to/model_file --input_file path/to/image_file
```

### Example Commands

1. **Predict using a Keras Model**

```sh
python3 predict.py --model mnist_model.keras --input_file images/1.png
```

2. **Predict using an ONNX Model**

```sh
python3 predict.py --model mnist_model.onnx --input_file images/2.png
```

3. **Predict using a TFLite Model**

```sh
python3 predict.py --model mnist_model.tflite --input_file images/3.png
```

## Notes

- You can draw your own digits using [draw.py](./draw.md)