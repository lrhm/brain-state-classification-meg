from pytorch_lightning import LightningModule
import torch as t
import torch.nn.functional as F
from argparse import Namespace

import matplotlib.pyplot as plt
import ipdb
import os
from torchmetrics import Accuracy


class SubjectAccuracies:
    def __init__(self, device: t.device):
        self.device = device
        self.accuracies = []

    def label_to_one_hot(self, labels: t.Tensor):
        labels = labels.to(self.device)
        labels = t.eye(4)[labels].to(self.device)
        return labels

    def one_hot_to_label(self, one_hot: t.Tensor):
        return t.argmax(one_hot)

    def compute(self):
        return [acc.compute() for acc in self.accuracies]

    def reset(self):
        for acc in self.accuracies:
            acc.reset()

    def update(
        self, subject_label: t.Tensor, predicted_task: t.Tensor, task: t.Tensor,
    ):
        while ((subject_label + 1) > len(self.accuracies)).any():
            self.accuracies.append(Accuracy())
        for i, acc in enumerate(self.accuracies):
            subject_mask = subject_label == i
            predicted_task_label = predicted_task[subject_mask]
            task_label = task[subject_mask]
            if len(task_label) > 0:
                acc.update(predicted_task_label.to(self.device), task_label.to(self.device))


class BaseClassificationModel(LightningModule):
    def __init__(self, params: Namespace):
        super().__init__()
        self.params = params
        self.save_hyperparameters()
        self.generator = t.nn.Sequential()
        self.loss = t.nn.BCELoss()
        self.train_accuracies = SubjectAccuracies(self.device)
        self.val_accuracies = SubjectAccuracies(self.device)
        self.test_accuracies = SubjectAccuracies(self.device)

    def forward(self, z: t.Tensor) -> t.Tensor:
        out = self.generator(z)
        return out

    def label_to_one_hot(self, labels: t.Tensor):
        labels = labels.long()
        labels = labels.to(self.device)
        labels = t.eye(4)[labels].to(self.device)
        return labels

    def one_hot_to_label(self, one_hot: t.Tensor):
        return t.argmax(one_hot, dim=1)

    def training_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, labels = batch
        task = labels[:, 0]
        subject_label = labels[:, 1]
        y = self.label_to_one_hot(task)
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        self.train_accuracies.update(
            subject_label=subject_label,
            predicted_task=self.one_hot_to_label(y_pred).int(),
            task=self.one_hot_to_label(y).int(),
        )
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = t.stack([x["loss"] for x in outputs]).mean()
        self.log("avg_train_loss", avg_loss, prog_bar=True)
        accuracies = self.train_accuracies.compute()
        for i, acc in enumerate(accuracies):
            self.log(
                f"train_acc_{i}", acc, prog_bar=True,
            )
        self.train_accuracies.reset()

    def validation_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, labels = batch
        task = labels[:, 0].int()
        patient_label = labels[:, 1]
        if batch_idx == 0:
            pass
        pred_y = self(x)
        y = self.label_to_one_hot(task)
        pred_task = self.one_hot_to_label(pred_y)
        self.val_accuracies.update(
            subject_label=patient_label, predicted_task=pred_task, task=task,
        )
        return {"val_loss": self.loss(y, pred_y)}

    def validation_epoch_end(self, outputs):
        accuracies = self.val_accuracies.compute()
        for i, acc in enumerate(accuracies):
            self.log(
                f"val_acc_{i}", acc, prog_bar=True,
            )
        self.val_accuracies.reset()
        t.save(
            self.state_dict(), os.path.join(self.params.save_path, "checkpoint.ckpt"),
        )
        avg_loss = t.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss, prog_bar=True)
        return {"val_loss": avg_loss}

    def test_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, labels = batch
        task = labels[:, 0].long()
        patient_label = labels[:, 1]
        if batch_idx == 0:
            pass
        pred_y = self.one_hot_to_label(self(x))
        self.test_accuracies.update(
            subject_label=patient_label, predicted_task=pred_y, task=task,
        )

    def test_epoch_end(self, outputs):
        accuracies = self.val_accuracies.compute()
        self.val_accuracies.reset()
        t.save(
            self.state_dict(), os.path.join(self.params.save_path, "checkpoint.ckpt"),
        )
        test_metrics = {f"{i}": acc for i, acc in enumerate(accuracies)}
        self.log("test_performance", test_metrics, prog_bar=True)

    def configure_optimizers(self):
        lr = self.params.lr
        b1 = self.params.b1
        b2 = self.params.b2

        optimizer = t.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(b1, b2),  # weight_decay=0.001
        )

        scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=self.params.reduce_lr_on_plateau_patience,
            min_lr=1e-6,
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
