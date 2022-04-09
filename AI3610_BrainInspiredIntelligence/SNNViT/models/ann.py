import torch
import torch.nn as nn  # 勇敢牛牛，不怕困难！

from .common import MHAttention, EmbeddingLayer


class FeedFwdLayer(nn.Module):
    def __init__(self, input_dims, hidden_dims):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, input_dims)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, nhead):
        super().__init__()
        self.feedforward = FeedFwdLayer(embed_dim, hidden_dim)
        self.attention = MHAttention(embed_dim, nhead)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        x = x + self.attention(x, x, x)
        x = self.norm1(x)

        x = x + self.feedforward(x)
        x = self.norm2(x)
        return x


class PredictionHead(nn.Module):
    def __init__(self, embedding_dims, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dims, embedding_dims)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(embedding_dims, n_classes)

    def forward(self, x: torch.Tensor):
        x = x[:, 0, :]  # [CLS] token
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class ViT(nn.Module):
    def __init__(
            self,
            img_size: int,
            n_channels: int,
            patch_size: int,
            embedding_dim: int,
            n_classes: int,
            nhead: int):
        super().__init__()
        self.embedding = EmbeddingLayer(img_size, n_channels, embedding_dim, patch_size)
        self.encoder = Encoder(embedding_dim, embedding_dim, nhead)
        self.head = PredictionHead(embedding_dim, n_classes)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.head(x)

        return x
