from ...base_lightning_modules.base_classification_model import BaseClassificationModel
from argparse import Namespace
from .module import ResNet1D
import ipdb

class Model(BaseClassificationModel):
    def __init__(self, params: Namespace):
        super().__init__(params)
        self.generator = ResNet1D(params)
    
    def forward(self, x):
        return self.generator(x.squeeze(1))