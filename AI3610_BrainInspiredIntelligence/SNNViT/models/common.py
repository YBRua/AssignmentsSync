import torch
import torch.nn as nn  # 勇敢牛牛，不怕困难！

from typing import Optional


class EmbeddingLayer(nn.Module):
    def __init__(
            self,
            img_size: int,
            n_channels: int,
            embedding_dim: int,
            patch_size: int,):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # [CLS] Token
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embedding_dim),
            requires_grad=True)

        # Positional Embedding
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, (img_size // patch_size) ** 2 + 1, embedding_dim),
            requires_grad=True)

        self.embedding_dim = embedding_dim

    def _forward(self, x: torch.Tensor):
        B = x.shape[0]
        x = self.conv1(x)
        x = x.reshape(B, self.embedding_dim, -1)  # batch, embedding, length
        x = x.transpose(1, 2)  # batch, length, embedding
        x = torch.cat((torch.repeat_interleave(
            self.cls_token, x.shape[0], 0), x), dim=1)
        x = x + self.pos_embedding
        return x

    def forward(self, x: torch.Tensor):
        x = self._forward(x)
        return x


class MHAttention(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            nhead: int):
        super().__init__()
        if embed_dim % nhead != 0:
            raise ValueError(
                'embed_dim must be divisible by nhead '
                f'embed_dim={embed_dim}, nhead={nhead}')
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

        self.nheads = nhead
        self.d_k = embed_dim // nhead

    def _forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor):
        B, N, E = query.shape

        # batch, length, nhead, d_k
        q = self.q_linear(query).reshape(B, N, self.nheads, self.d_k)
        k = self.k_linear(key).reshape(B, N, self.nheads, self.d_k)
        v = self.v_linear(value).reshape(B, N, self.nheads, self.d_k)

        # batch, nhead, length, d_k
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # batch * nheads, length, d_k
        q = q.reshape(B * self.nheads, N, self.d_k)
        k = k.reshape(B * self.nheads, N, self.d_k)
        v = v.reshape(B * self.nheads, N, self.d_k)

        attn = torch.bmm(q, k.transpose(-1, -2)) / \
            (self.d_k ** 0.5)  # B * nheads, length, length
        attn = torch.softmax(attn, dim=-1)

        out = torch.bmm(attn, v)  # B * nheads, length, d_k
        out = out.reshape(B, self.nheads, N, self.d_k).transpose(
            1, 2)  # B, length, nheads, d_k
        # B, length * nheads, d_k
        out = out.reshape(B, -1, self.nheads * self.d_k)
        return out

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor):
        return self._forward(query, key, value)