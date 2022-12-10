import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm

from transformers import RobertaModel

import torch.optim as optim

from diffusion_model import DDPM

class DiffusionRoberta(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.roberta_model = RobertaModel.from_pretrained("roberta-base").to(device)
        self.diffusion_model = DDPM(args, dim_mults=(1, 2, 4)).to(device)

    def forward(self, encoded_caption, time):
        roberta_output = self.roberta_model(**encoded_caption)
        # I think the stuff below is wrong: Not sure if we should pass in to the network the output of the roberta model on top of the noise or as the noise
        self.diffusion_model(roberta_output, time)
    