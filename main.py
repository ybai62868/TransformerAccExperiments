import torch
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from detr_transformer import build_transformer


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

     # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    return parser


def main(args):
    transformer = build_transformer(args)
    # print(transformer)
    src = torch.rand((1050, 1, 256))
    pos_emd = torch.rand((1050, 1, 256))
    query_emd = torch.rand(100, 1, 256)
    tgt = torch.rand((100, 1, 256))
    mask = torch.zeros((1, 1050))
    output = transformer(src, mask, query_emd, pos_emd)
    print("output shape:", output[0].shape)
    print("output shape:", output[1].shape)


    


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)