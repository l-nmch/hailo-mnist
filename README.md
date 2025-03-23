# MNIST Model for Hailo8L

This repository provides a guide to training and running a TensorFlow-based MNIST model locally, and deploying it on the [Hailo8L](https://www.raspberrypi.com/products/ai-hat/) chip. It includes tools for training, testing, generating images, as well as instructions for converting the trained model to a format that can be optimized and deployed on the Hailo8L.

## Features

- Train a Convolutional Neural Network (CNN) model on the MNIST dataset using TensorFlow.

- Export the trained model in various formats (.keras, .tflite, .onnx).

- Run inference locally on the trained model.

- A utility to generate images for the MNIST model, if needed.

- Instructions for converting and optimizing the model using Hailo8L's DFC (Hailo Dataflow Compiler) (installation and usage instructions in a separate document).

## Setup

1. Install Dependencies

To get started, clone this repository :

```sh
git clone https://github.com/l-nmch/hailo-mnist.git
cd hailo-mnist
```

Create a virtualenv :

```sh
python3 -m venv .venv
source .venv/bin/activate
```

Install the required Python libraries

```sh
python3 -m pip install -r requirements.txt
```

2. Train the Model

Once the dependencies are installed, you can train the MNIST model using the following command:

```sh
python3 src/model.py --save mnist.keras
```

This will train the model on the MNIST dataset and save it in the .keras format as mnist.keras.

3. Test the Model Locally

To perform inference locally, use the predict.py script. For example, if you have an image (e.g., images/0.png), run:

```sh
python3 src/predict.py --model mnist.keras --input_file images/0.png
```

## Documentation

For further explanation on how to run the model on Hailo NPU, head to the [Documentation](./docs/README.md)

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.