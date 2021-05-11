import torch
import torch.nn as nn


transformer_model = nn.Transformer(nhead=8, num_encoder_layers=6)


src = torch.rand((1050, 1, 256))
tgt = torch.rand((100, 1, 256))


output = transformer_model(src, tgt)
