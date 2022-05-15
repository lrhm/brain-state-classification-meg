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
        drouput=0.4,
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

        self.conv1 = Conv1DBlock(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv1DBlock(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = Conv1DBlock(64, 64, kernel_size=3, stride=1, padding=1)
        # self.conv4 = Conv1DBlock(64, 1, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Linear(64, num_class)

    def forward(self, x):
        # ipdb.set_trace()
        x = x.squeeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        # ipdb.set_trace()
        x = x.max(dim=2)[0]
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.softmax(x, dim=1)
