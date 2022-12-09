import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self):
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)

