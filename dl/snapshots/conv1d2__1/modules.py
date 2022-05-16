import torch as t
from torch import nn
import ipdb
import torch.nn.functional as F


class Conv1DBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        drouput=0.3,
    ):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.do = nn.Dropout(drouput)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.bn(x)
        x = self.do(x)
        x = self.relu(x)
        return x


# Conv1d Classifier
class Conv1dClassifier(nn.Module):
    def __init__(self, in_channels=248, num_class=4):
        super(Conv1dClassifier, self).__init__()

        self.conv1 = Conv1DBlock(in_channels, 256, kernel_size=16, stride=1, padding=1)
        self.conv2 = Conv1DBlock(256, 256, kernel_size=16, stride=1, padding=1)
        self.conv3 = Conv1DBlock(256, 256, kernel_size=16, stride=1, padding=1)
        self.conv4 = Conv1DBlock(256, 256, kernel_size=16, stride=1, padding=1)
        # spacial attention
        self.classifier = nn.Linear(256, num_class)

    def forward(self, x):
        # ipdb.set_trace()
        x = x.squeeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # ipdb.set_trace()
        x = x.max(dim=2)[0]
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.softmax(x, dim=1)


class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training:
            return x + t.randn(x.size()) * () * self.stddev
        else:
            return x


# FAT Conv1d Classifier
class FITConv1dClassifier(nn.Module):
    def __init__(self, in_channels=248, num_class=4):
        super(FITConv1dClassifier, self).__init__()
        hidden_channel = 256
        num_layers = 4
        for i in range(num_layers):
            setattr(
                self,
                f"conv{i}",
                Conv1DBlock(
                    in_channels,
                    hidden_channel,
                    kernel_size=64,
                    stride=1,
                    padding=1,
                    drouput=0.5 if i != num_layers - 1 else 0.0, # last layer no dropout
                ),
            )
            in_channels = hidden_channel

        self.classifier = nn.Linear(hidden_channel, num_class)
        self.num_layers = num_layers
        # self.noise_layer = Gaussian

    def forward(self, x):
        # ipdb.set_trace()
        x = x.squeeze(1)
        for i in range(self.num_layers):
            x = getattr(self, f"conv{i}")(x)

        x = x.max(dim=2)[0]
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.softmax(x, dim=1)
