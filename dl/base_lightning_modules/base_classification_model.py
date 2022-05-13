from pytorch_lightning import LightningModule
import torch as t
import torch.nn.functional as F
from argparse import Namespace

import matplotlib.pyplot as plt
import ipdb
import os


class BaseClassificationModel(LightningModule):
    def __init__(self, params: Namespace):
        super().__init__()
        self.params = params
        self.save_hyperparameters()
        self.generator = t.nn.Sequential()
        self.loss = t.nn.BCELoss()

    def forward(self, z: t.Tensor) -> t.Tensor:
        out = self.generator(z)
        return out

    def training_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = t.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss, prog_bar=True)
        t.save(
            self.state_dict(), os.path.join(self.params.save_path, "checkpoint.ckpt"),
        )
        return {"val_mse": avg_loss}

    def training_epoch_end(self, outputs):
        avg_loss = t.stack([x["loss"] for x in outputs]).mean()
        self.log("loss", avg_loss, prog_bar=True)

    def validation_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        if batch_idx == 0:
            pass
        y = y.cpu()
        pred_y = self(x).cpu()
        loss = self.loss(pred_y, y)
        # self.log("val_mse", loss, prog_bar=True)
        return {"val_mse": loss, "val_loss": loss}

    def test_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        if batch_idx == 0:
            pass

        pred_y = self(x).cpu()
        tn, fp, fn, tp = t.bincount(
            y.cpu() * 2 + pred_y, minlength=4,
        )
        total_lengh = y.numel()
        return {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "total_lengh": total_lengh,
        }

    def test_epoch_end(self, outputs):
        total_lenght = sum([x["total_lengh"] for x in outputs])
        tn = t.stack([x["tn"] for x in outputs]).sum() / total_lenght
        fp = t.stack([x["fp"] for x in outputs]).sum() / total_lenght
        fn = t.stack([x["fn"] for x in outputs]).sum() / total_lenght
        tp = t.stack([x["tp"] for x in outputs]).sum() / total_lenght
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        test_metrics = {
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "f1": f1,
        }
        test_metrics = {k: v for k, v in test_metrics.items()}
        self.log("test_performance", test_metrics, prog_bar=True)

    def configure_optimizers(self):
        lr = self.params.lr
        b1 = self.params.b1
        b2 = self.params.b2

        generator_optimizer = t.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(b1, b2)
        )

        return generator_optimizer
