import os
import torch as t
import ipdb
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
from .preprocess import preprocess

class CustomDataModule(LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.data_location = params.data_location
        self.train_batch_size = params.train_batch_size
        self.test_batch_size = params.test_batch_size
        self.data_location = params.data_location
        self.params = params
        preprocessed = preprocess()
        self.train_data = preprocessed["train"]
        self.test_data = preprocessed["test"]

    def train_dataloader(self):
        # creates a DeepCoastalDataset object
        dataset = CustomDataset(
            self.train_data, train=True, imsize=self.params.imsize
        )
        return DataLoader(
            dataset, batch_size=self.train_batch_size, drop_last=True, num_workers=3,
        )

    def val_dataloader(self):
        # creates a DeepCoastalDataset object
        dataset = CustomDataset(
            self.test_data, train=False, imsize=self.params.imsize
        )
        return DataLoader(
            dataset, batch_size=self.train_batch_size, drop_last=True, num_workers=3,
        )

    def test_dataloader(self):
        # creates a DeepCoastalDataset object
        dataset = CustomDataset(
            self.test_data, train=False, imsize=self.params.imsize
        )
        return DataLoader(
            dataset, batch_size=self.train_batch_size, drop_last=True, num_workers=3,
        )


class CustomDataset(Dataset):
    def __init__(
        self,
        data: tuple[t.Tensor, t.Tensor],
        train: bool = True,
        imsize: int = 46,
    ):
        self.data = data
        self.train = train

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        return self.data[0][idx], self.data[1][idx]


def test():
    data_module = CustomDataModule()
    train_dataloader = data_module.train_dataloader()
    for i, (x, y) in enumerate(train_dataloader):
        print(i, x.shape, y.shape)
    """
    train_dl, test_dl = get_loaders(
        "/mnt/tmp/multi_channel_train_test",
        32,
        64,
        t.device("cuda" if t.cuda.is_available() else "cpu"),
        in_seq_len=8,
        out_seq_len=4,
    )
    for i, (x, y) in enumerate(tqdm(train_dl)):
        # plt.imshow(x[0, 0, 0].cpu())
        # plt.show()
        # print(x.shape)
        # return
        # print(f"iteration: {i}")
        pass
    """
    # reads file in h5 format


if __name__ == "__main__":
    test()
