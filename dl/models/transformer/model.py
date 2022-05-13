from torch.nn.modules.activation import ReLU

from dl.models.transformer.modules import Transformer
from ...base_lightning_modules.base_classification_model import BaseClassificationModel
import ipdb

# from ...base_torch_modules.resnetmodel import (
# )
# ..conv2dmodel import FrameDiscriminator
from ...base_torch_modules.resnetmodel import ResNetFrameDiscriminator
import torch as t
from torch import nn
import torch.nn.functional as F
from argparse import Namespace


class ConvModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.linear = nn.Linear(128, 4)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        ipdb.set_trace()
        x = t.softmax(self.linear(x.view(x.size(0), -1)), dim=1)
        return x


class Model(BaseClassificationModel):
    def __init__(self, params: Namespace):
        super().__init__(params)
        self.generator = Transformer(99, 248, 4, 4, 4)
