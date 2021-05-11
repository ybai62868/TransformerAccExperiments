import torch
import torch.nn as nn
import torchvision

model = nn.Transformer(d_model=256).cuda()
# src = torch.rand((10, 32, 512)).cuda()
# tgt = torch.rand((20, 32, 512)).cuda()

src = torch.rand((1050, 1, 256)).cuda()
tgt = torch.rand((100, 1, 256)).cuda()


model = model.eval()

torch.onnx.export(
                model,
                (src, tgt),
                "transformer_torch.onnx",
                opset_version = 11,
                input_names = ["src_input", "tgt_input"],
                output_names = ["output"],
)