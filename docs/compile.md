# Compiling the MNIST Model for Hailo8L NPU

This guide provides step-by-step instructions for compiling the trained MNIST model to run on the Hailo8L NPU. The compilation process requires using the Hailo DFC (Deep Learning Framework Compiler) available for x86 machines.

## Prerequisites

Before proceeding, ensure the following:

1. **Hailo DFC** :

- The Hailo DFC is only available for x86 architecture. You will need an x86 machine to perform the compilation.
- The DFC can be installed following the instructions provided in the [Hailo Developer Zone](https://hailo.ai/developer-zone/).

2. **Virtual Environment**:

- Create a new Python virtual environment for the compilation process, as the TensorFlow and library versions in the DFC environment will differ from those in your training environment.

3. **Trained Model**:

- Ensure that you have already trained the model and saved it in the `.tflite` format using the steps outlined in the [Training Guide](./train.md).
- The model should be saved as `mnist.tflite` before proceeding with the compilation process.

## Setting Up the Compilation Environment

1. Create a New Virtual Environment

Create and activate a new virtual environment for the Hailo DFC compilation:

```bash
python3 -m venv hailo_compilation_venv
source hailo_compilation_venv/bin/activate 
```

2. Install Hailo DFC

Follow the instructions provided in the Hailo Developer Zone to install the Hailo DFC.

Note: The versions of TensorFlow and other dependencies in this environment will differ from those in your training environment, but the compilation will work as expected.

3. Copy the Trained Model

Ensure that the trained model file is available within the new virtual environment for compilation.

## Compiling the Model for Hailo8L

Once you have set up the environment and the trained model, you can proceed with the following steps to compile the model for the Hailo8L NPU.

1. Parse the Model

Use the Hailo DFC parser to parse the trained .tflite model for the Hailo8L architecture:

```sh
hailo parser tf --hw-arch hailo8l --net-name mnist mnist.tflite 
```

This command will parse the mnist.tflite model and prepare it for optimization.

2. Prepare a calibation set 

To optimize the model you need to give hailo a prepation set:

```sh
python3 src/calibration_set.py --image_folder images/ --ouput calib_set.npy
```

3. Optimize the Model

Next, optimize the parsed model using the hailo optimize command. You can also use the `--use-random-calib-set` for the optimizer to use a random calibration set:

```sh
hailo optimize --calib-set-path calib_set.npy --hw-arch hailo8l mnist.har
```

This will optimize the model and save it as mnist.har (Hailo ARchitecture format).

4. Compile the Model

Finally, compile the optimized model to generate the .hef file that will be deployed to the Hailo8L NPU:

```sh
hailo compiler --hw-arch hailo8l mnist_optimized.har
```

The output of this command will be the mnist.hef file, which is ready to run on the Hailo8L NPU.

## Use it on Hailo8L NPU

After following the official installation [Documentation](https://www.raspberrypi.com/documentation/computers/ai.html) and compiled the model, you can predict images on the NPU:

```sh
python3 src/predict_hailo.py --model mnist_optimized.hef --input_file images/0.png
```

## Notes

- The Hailo DFC requires an x86 machine for the compilation process. Make sure you're using the correct architecture.

- The versions of TensorFlow and other libraries in your training environment may differ from those in the Hailo DFC environment, but this will not affect the functionality of the model when compiled.

- Ensure that you have correctly followed the setup steps for the DFC before starting the compilation process.