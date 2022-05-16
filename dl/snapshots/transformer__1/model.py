from torch.nn.modules.activation import ReLU

from .modules import Transformer
from ...base_lightning_modules.base_classification_model import BaseClassificationModel
import ipdb
from argparse import Namespace


class Model(BaseClassificationModel):
    def __init__(self, params: Namespace):
        super().__init__(params)
        self.generator = Transformer(57, 248, 4, 4, 4)
