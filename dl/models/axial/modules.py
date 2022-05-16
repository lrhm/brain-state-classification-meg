from ipaddress import ip_address
import math
import torch as t
import torch.nn as nn
from axial_attention import AxialAttention, AxialPositionalEmbedding
import ipdb


class GaussianNoise(nn.Module):
    def __init__(self, variance=0.001):
        super().__init__()
        self.variance = variance

    def forward(self, x, variance=None):
        if variance is None:
            variance = self.variance
        noise = t.randn_like(x) * variance
        x = noise + x
        return x


# Encoder decoder that transforms numbers between 0 and 1 to embeddings
class AxialClassifier(nn.Module):
    def __init__(self, embedding_dim=4, num_classes=4):
        super().__init__()

        # self.embedding_dim = embedding_dim
        # # embedding encoder
        # self.embedding_encoder = nn.Sequential(
        #     nn.Linear(1, embedding_dim, bias=True),
        # )

        # # decoding the embedding to the output
        # self.embedding_decoder = nn.Sequential(
        #     nn.Linear(embedding_dim , 1, bias=True),
        #     nn.Sigmoid(),
        # )

        self.positional_embedding = AxialPositionalEmbedding(99, (248,), -1)

        self.attentions = nn.Sequential(
            AxialAttention(
                dim=99,  # embedding dimension
                dim_index=-1,  # where is the embedding dimension
                heads=9,  # number of heads for multi-head attention
                dim_heads=4,
                num_dimensions=1,  # number of axial dimensions (images is 2, video is 3, or more)
            ),
            nn.LeakyReLU(0.2),
            AxialAttention(
                dim=99,  # embedding dimension
                dim_index=-1,  # where is the embedding dimension
                heads=11,  # number of heads for multi-head attention
                dim_heads=4,
                num_dimensions=1,  # number of axial dimensions (images is 2, video is 3, or more)
            ),
            nn.LeakyReLU(0.2),
            AxialAttention(
                dim=99,  # embedding dimension
                dim_index=-1,  # where is the embedding dimension
                heads=9,  # number of heads for multi-head attention
                dim_heads=4,
                num_dimensions=1,  # number of axial dimensions (images is 2, video is 3, or more)
            ),
            nn.LeakyReLU(0.2),
            AxialAttention(
                dim=99,  # embedding dimension
                dim_index=-1,  # where is the embedding dimension
                heads=11,  # number of heads for multi-head attention
                dim_heads=4,
                num_dimensions=1,  # number of axial dimensions (images is 2, video is 3, or more)
            ),
            nn.LeakyReLU(0.2),
            AxialAttention(
                dim=99,  # embedding dimension
                dim_index=-1,  # where is the embedding dimension
                heads=9,  # number of heads for multi-head attention
                dim_heads=4,
                num_dimensions=1,  # number of axial dimensions (images is 2, video is 3, or more)
            ),
            nn.LeakyReLU(0.2),
        )

        self.noise_layer = GaussianNoise(0.001)

        self.classifier = nn.Sequential(

            nn.Linear(248, num_classes, bias=True),
            # nn.LeakyReLU(0.2),
            # nn.Linear(2048, 1024, bias=True),
            # nn.LeakyReLU(0.2),
            # nn.Linear(1024, num_classes, bias=True),
            nn.Softmax(dim=1),
        )

    def forward(self, x):

        x = x.permute(0, 2, 3, 1)
        # ipdb.set_trace()
        x = x.squeeze(-1)
        x = self.noise_layer(x)
        # x = self.embedding_encoder(x)
        # x = self.positional_embedding(x)
        x = self.attentions(x)
        # x = self.embedding_decoder(x)

        # x = x.flatten(start_dim=1)
        x = x.max(dim=2)[0]

        # x= x.squeeze(-1)
        x = x.view(x.size(0), -1)
        # ipdb.set_trace()

        x = self.classifier(x)

        return x
