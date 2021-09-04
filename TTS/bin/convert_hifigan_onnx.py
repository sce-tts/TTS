import argparse
import os

import torch

# prevent GPU use
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from TTS.config import load_config
from TTS.vocoder.utils.generic_utils import setup_generator

# define args
parser = argparse.ArgumentParser()
parser.add_argument("--torch_model_path", type=str, help="Path to target torch model to be converted to ONNX.")
parser.add_argument("--config_path", type=str, help="Path to config file of torch model.")
parser.add_argument("--output_path", type=str, help="path to output file including file name to save ONNX model.")
args = parser.parse_args()

# load model config
config_path = args.config_path
c = load_config(config_path)

# init torch model
model = setup_generator(c)
checkpoint = torch.load(args.torch_model_path, map_location=torch.device("cpu"))
state_dict = checkpoint["model"]
model.load_state_dict(state_dict)
model.remove_weight_norm()

torch.onnx.export(
    model,  # model being run
    torch.ones((1, 80, 10)),  # model input (or a tuple for multiple inputs)
    args.output_path,  # where to save the model (can be a file or file-like object)
    opset_version=12,
    export_params=True,  # store the trained parameter weights inside the model file
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=["input"],  # the model's input names
    output_names=["output"],  # the model's output names
    dynamic_axes={"input": {2: "seq_length"}, "output": {1: "seq_length"}},  # variable lenght axes
)

print(" > Model conversion is successfully completed :).")
