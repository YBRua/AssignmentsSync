import torch
import torch.nn as nn
import snntorch as snn

from typing import Callable
from .common import MHAttention, EmbeddingLayer


class SNNFeedFwdLayer(nn.Module):
    def __init__(
            self,
            input_dims: int,
            hidden_dims: int,
            beta: float,
            spike_grad: Callable):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, input_dims)
        self.lif = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def init_leaky(self):
        return self.lif.init_leaky()

    def forward(self, x: torch.Tensor, mem: torch.Tensor):
        x = self.fc1(x)
        x, mem_ = self.lif(x, mem)
        x = self.fc2(x)
        return x, mem_


class SNNPredictionHead(nn.Module):
    def __init__(
            self,
            embedding_dims: int,
            n_classes: int,
            beta: float,
            spike_grad: Callable):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dims, embedding_dims)
        self.lif = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(embedding_dims, n_classes)

    def init_leaky(self):
        return self.lif.init_leaky()

    def forward(self, x: torch.Tensor, mem: torch.Tensor):
        x = x[:, 0, :]  # [CLS] token
        x = self.fc1(x)
        x, mem_ = self.lif(x, mem)
        x = self.fc2(x)
        return x, mem_


class SNNEncoder(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            hidden_dim: int,
            nhead: int,
            beta: float,
            spike_grad: Callable):
        super().__init__()
        self.feedforward = SNNFeedFwdLayer(
            embed_dim,
            hidden_dim,
            beta,
            spike_grad)
        self.attention = MHAttention(embed_dim, nhead)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def init_leaky(self):
        return self.feedforward.init_leaky()

    def forward(self, x: torch.Tensor, mem: torch.Tensor):
        x = x + self.attention(x, x, x)
        x = self.norm1(x)

        x_, mem_ = self.feedforward(x, mem)
        x = x + x_

        x = self.norm2(x)
        return x, mem_


class SNNViT(nn.Module):
    def __init__(
            self,
            img_size: int,
            n_channels: int,
            patch_size: int,
            embedding_dim: int,
            n_classes: int,
            nhead: int,
            beta: float,
            spike_grad: Callable,
            steps: int):
        super().__init__()
        self.embedding = EmbeddingLayer(img_size, n_channels, embedding_dim, patch_size)
        self.encoder = SNNEncoder(embedding_dim, embedding_dim, nhead, beta, spike_grad)
        self.head = SNNPredictionHead(embedding_dim, n_classes, beta, spike_grad)

        self.steps = steps

    def forward(self, x: torch.Tensor):
        mem_rec = []
        spk_rec = []

        encoder_mem = self.encoder.init_leaky()
        head_mem = self.head.init_leaky()

        for step in range(self.steps):
            xt = x[step]
            xt = self.embedding(xt)
            xt, encoder_mem = self.encoder(xt, encoder_mem)
            xt, head_mem = self.head(xt, head_mem)
            mem_rec.append(head_mem)
            spk_rec.append(xt)

        return torch.stack(spk_rec), torch.stack(mem_rec)
