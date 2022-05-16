import torch
from torch import nn
import torch.nn.functional as F
import ipdb


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, heads):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.to_keys = nn.Linear(embedding_dim, embedding_dim * heads, bias=False)
        self.to_queries = nn.Linear(embedding_dim, embedding_dim * heads, bias=False)
        self.to_values = nn.Linear(embedding_dim, embedding_dim * heads, bias=False)
        self.unify_heads = nn.Linear(heads * embedding_dim, embedding_dim)

    def forward(self, x):
        # ipdb.set_trace()
        batch_size, tweet_length, embedding_dim = x.size()
        keys = self.to_keys(x).view(batch_size, tweet_length, self.heads, embedding_dim)
        queries = self.to_queries(x).view(
            batch_size, tweet_length, self.heads, embedding_dim
        )
        values = self.to_values(x).view(
            batch_size, tweet_length, self.heads, embedding_dim
        )
        keys = (
            keys.transpose(1, 2)
            .contiguous()
            .view(batch_size * self.heads, tweet_length, embedding_dim)
        )
        queries = (
            queries.transpose(1, 2)
            .contiguous()
            .view(batch_size * self.heads, tweet_length, embedding_dim)
        )
        values = (
            values.transpose(1, 2)
            .contiguous()
            .view(batch_size * self.heads, tweet_length, embedding_dim)
        )
        queries = queries / (embedding_dim ** (1 / 4))
        keys = keys / (embedding_dim ** (1 / 4))

        dot = F.softmax(torch.bmm(queries, keys.transpose(1, 2)), dim=2)

        out = torch.bmm(dot, values).view(
            batch_size, self.heads, tweet_length, embedding_dim
        )
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(batch_size, tweet_length, self.heads * embedding_dim)
        )
        return self.unify_heads(out)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, fc_hidden_multiply=4):
        super().__init__()
        self.attention = SelfAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * fc_hidden_multiply),
            nn.ReLU(),
            nn.Linear(embedding_dim * fc_hidden_multiply, embedding_dim),
        )
        self.do = nn.Dropout(0.2)

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)  # TODO: explain skip connection
        # x = self.do(x)
        feedforward = self.fc(x)
        x = self.norm2(feedforward + x)
        x = self.do(x)
        return x


class GaussianNoise(nn.Module):
    def __init__(self, variance=0.001):
        super().__init__()
        self.variance = variance

    def forward(self, x, variance=None):
        if variance is None:
            variance = self.variance
        noise = torch.randn_like(x) * variance
        x = noise + x
        return x

class Transformer(nn.Module):
    def __init__(self, embedding_dim, seq_length, num_heads, depth, num_labels):
        super().__init__()

        self.positional_embedding = nn.Embedding(
            embedding_dim=embedding_dim, num_embeddings=seq_length
        )
        transformer_blocks = []
        for _ in range(depth):
            transformer_blocks.append(TransformerBlock(embedding_dim, num_heads))

        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.to_probabilities = nn.Linear(embedding_dim, num_labels)

        self.guassian_noise = GaussianNoise(0.01)
        

    def forward(self, x):
        # size x: (batch_size, channels, fourirer_length)

        # ipdb.set_trace()
        x = x.squeeze(1)
        x = self.guassian_noise(x)

        batch_size, tweet_length, embedding_dim = x.shape
        positions = torch.unsqueeze(
            self.positional_embedding(torch.arange(tweet_length, device=x.device)), 0
        ).expand(batch_size, tweet_length, embedding_dim)

        x = x + positions
        x = self.transformer_blocks(x)
        x = x.max(dim=1)[0]
        x = self.to_probabilities(x)
        # ipdb.set_trace()
        return F.softmax(x, dim=1)  # todo: why log_softmax instead of softmax
