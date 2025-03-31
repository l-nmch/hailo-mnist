# Drawing Digits for MNIST Prediction

This guide provides instructions for using the `draw.py` script to draw digits that can be used for MNIST model prediction.

## Drawing Digits

To draw digits, run the `draw.py` script. This script will open a drawing interface where you can draw digits using your mouse.

### Running the Script

To start the drawing interface:

```sh
python3 draw.py
```

### Saving the Drawn Image

After drawing a digit, you can save the image to a file. The saved image can then be used as input for the `predict.py` script to predict the digit.

## Example Workflow

1. **Draw a Digit**

```sh
python3 draw.py
```

2. **Save the Drawn Image**

Save the drawn image to a file, e.g., `drawn_digit.png`.

3. **Predict the Drawn Digit**

Use the saved image as input for the `predict.py` script:

```sh
python3 predict.py --model mnist_model.keras --input_file drawn_digit.png
```