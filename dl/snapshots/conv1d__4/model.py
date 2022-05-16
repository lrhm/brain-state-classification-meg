from argparse import Namespace
from ...base_lightning_modules.base_classification_model import BaseClassificationModel
from .modules import Conv1dClassifier, FitConv1dClassifier,Res1D


class Model(BaseClassificationModel):
    def __init__(self, params: Namespace):
        super().__init__(params)

        self.generator = Res1D()
