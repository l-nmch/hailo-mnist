import numpy as np
import argparse
from PIL import Image, ImageOps
from hailo_platform import (
    HEF,
    ConfigureParams,
    FormatType,
    HailoSchedulingAlgorithm,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    VDevice,
)

parser = argparse.ArgumentParser(description='Predict a digit from an image using hailo NPU.')
parser.add_argument('--model', type=str, required=True, help='Name of the model file to use (with extension .hef)')
parser.add_argument('--input_file', type=str, required=True, help='Input file to predict')
args = parser.parse_args()

params = VDevice.create_params()
params.scheduling_algorithm = HailoSchedulingAlgorithm.NONE
target = VDevice(params=params)

hef = HEF(args.model)

configure_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
network_groups = target.configure(hef, configure_params)
network_group = network_groups[0]
network_group_params = network_group.create_params()

input_vstreams_params = InputVStreamParams.make(network_group, quantized=False, format_type=FormatType.FLOAT32)
output_vstreams_params = OutputVStreamParams.make(network_group, quantized=True, format_type=FormatType.FLOAT32)

input_vstream_info = hef.get_input_vstream_infos()[0]
output_vstream_info = hef.get_output_vstream_infos()[0]
image_height, image_width, channels = input_vstream_info.shape

def load_image(image_path):
    """Load and preprocess the input image for prediction."""
    image = Image.open(image_path)
    img_resized = image.resize((28, 28))
    img_inverted = ImageOps.invert(img_resized.convert("L"))
    img_array = np.array(img_inverted)
    img_array = img_array.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array

input_image = load_image(args.input_file)

input_data = {input_vstream_info.name: input_image}

with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
    with network_group.activate(network_group_params):
        infer_results = infer_pipeline.infer(input_data)

prediction_result = infer_results[output_vstream_info.name]
print(f"Prediction result: {prediction_result}")

prediction = prediction_result[0].tolist()
confidence_percentage = (max(prediction) / sum(prediction)) * 100
print(f"Predicted class: {prediction.index(max(prediction))} with confidence: {confidence_percentage:.2f}%")
