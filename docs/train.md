# Training the MNIST Model

This guide provides instructions for training the MNIST model using the `model.py` script.

## Training the Model

To train the model, run the `model.py` script with the desired parameters.

### Basic Training

To train the model and save it in the `.keras` format:

```sh
python3 model.py --save mnist_model.keras --epochs 6
```

### Exporting the Model

To export the model to a specific folder:

```sh
python3 model.py --export export_folder --epochs 6
```

### Saving in Different Formats

You can save the model in different formats by specifying the file extension:

- **Keras Format**: `.keras`
- **ONNX Format**: `.onnx`
- **TFLite Format**: `.tflite`

For example, to save the model in ONNX format:

```sh
python3 model.py --save mnist_model.onnx --epochs 6
```

## Model Summary and Evaluation

After training, the script will display the model summary and evaluate its accuracy on the test dataset.

## Notes

- The default number of epochs is set to 6. You can change this by using the `--epochs` argument.
- Ensure the file extension for the `--save` argument is correct to avoid unsupported file extension errors.

## Example Commands

1. **Train and Save as Keras Model**

```sh
python3 model.py --save mnist_model.keras --epochs 6
```

2. **Train and Save as ONNX Model**

```sh
python3 model.py --save mnist_model.onnx --epochs 6
```

3. **Train and Save as TFLite Model**

```sh
python3 model.py --save mnist_model.tflite --epochs 6
```

4. **Export the Model**

```sh
python3 model.py --export export_folder --epochs 6
```