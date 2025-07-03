import math

import torch
import torch.nn as nn
from torch import Tensor
from .base import Embedding

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.shape[-1]
        x = x + self.pe[:, :seq_len]
        return x


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        self.pe = nn.Parameter(
            torch.empty(1, d_model, max_len).normal_(std=0.02)
        )

    def forward(self, x):
        return x + self.pe


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=3,
            bias=True,
        )
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.GELU()
        self.conv_last = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=True,
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x.transpose(2, 1)).transpose(2, 1)
        x = self.act(x)
        x = self.conv_last(x)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        init_dim: int,
        proj_dim: int,
        out_dim: int,
        d_model: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_layers: int = 18,
        prenorm: bool = True,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.in_channels = in_channels 
        self.dropout = dropout
        self.out_dim = out_dim

        self.down_proj = nn.Linear(init_dim, proj_dim)
        self.input_proj = nn.Linear(in_channels, d_model)

        self.pos_encoder = PositionalEncoding(d_model, max_len=proj_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
            norm_first=prenorm,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=self.num_layers
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.d_model, self.out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.down_proj(x)
        # Transpose to [batch, proj_dim, num_ifos] for input projection
        x = x.transpose(1, 2)
        # Project to transformer dimension: [batch, proj_dim, num_ifos] -> [batch, proj_dim, d_model]
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        x = self.pos_encoder(x)

        # (batch, seq, features)
        x = x.permute(0, 2, 1)
        x = self.transformer_encoder(x)

        # (batch, features, seq)
        x = x.permute(0, 2, 1)
        
        # pooling and flatten
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

