from torch.nn.modules.activation import ReLU
from ...base_lightning_modules.base_classification_model import (
    BaseClassificationModel,
)
import ipdb

import torch as t
from torch import nn
import torch.nn.functional as F
from argparse import Namespace
from .unet import UNet


class Model(BaseClassificationModel):
    def __init__(self, params: Namespace):
        super().__init__(params)
        self.generator = UNet(in_channels=248, out_classes=4, dimensions=1)

    def forward(self, x):
        return self.generator(x.squeeze(1))
