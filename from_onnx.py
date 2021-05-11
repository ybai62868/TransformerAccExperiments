import tvm
from tvm import relay
import onnx
import numpy as np


model_path = "./transformer_sim.onnx"

onnx_model = onnx.load(model_path)


target = "llvm"
input_name = "1"
x = np.random.random((35, 1))
shape_dict = {input_name: x.shape}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)