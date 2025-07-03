from typing import Literal, Optional

import torch
from ml4gw.nn.norm import NormLayer
from ml4gw.nn.resnet.resnet_1d import ResNet1D
import torch.nn as nn
from .base import Embedding
from .transformer import Transformer

class GatedFusion(nn.Module):
    """
    Fusion layer for combining time and frequency domain features intelligently
    """

    def __init__(
        self,
        time_dim: int,
        freq_dim: int,
        fusion_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        fusion_dim = fusion_dim or time_dim + freq_dim
        # cross-domain enhancement
        self.time_to_freq = nn.Linear(time_dim, freq_dim)
        self.freq_to_time = nn.Linear(freq_dim, time_dim)

        # gating mechanism - controls how much time and frequency 
        # content influence eachother on a per sample basis 
        self.time_gate = nn.Sequential(nn.Linear(time_dim, 1), nn.Sigmoid())
        self.freq_gate = nn.Sequential(nn.Linear(freq_dim, 1), nn.Sigmoid())

        # final fusion
        self.fusion = nn.Sequential(
            nn.Linear(time_dim + freq_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
        )

    def forward(self, time_embed, freq_embed):
        time_enhanced = time_embed + self.time_gate(
            time_embed
        ) * self.freq_to_time(freq_embed)
        
        freq_enhanced = freq_embed + self.freq_gate(
            freq_embed
        ) * self.time_to_freq(time_embed)

        # Concatenate and fuse
        combined = torch.cat([time_enhanced, freq_enhanced], dim=1)
        fused = self.fusion(combined)

        return fused


class MultiModal(Embedding):
    def __init__(
        self,
        num_ifos: int,
        time_context_dim: int,
        freq_context_dim: int,
        time_layers: list[int],
        freq_layers: list[int],
        time_kernel_size: int = 3,
        freq_kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: Optional[list[Literal["stride", "dilation"]]] = None,
        norm_layer: Optional[NormLayer] = None,
        asds: bool = False,
        fusion: bool = False,
        **kwargs,
    ):
        """
        MultiModal embedding network that processes time, frequency, and possibly asd data.
        """
        super().__init__()
        # set the context dimension so
        # the flow can access it
        self.context_dim = time_context_dim + freq_context_dim
        
        self.asds = asds

        self.time_domain_resnet = ResNet1D(
            in_channels=num_ifos,
            layers=time_layers,
            classes=time_context_dim,
            kernel_size=time_kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )

        channels_per_ifo = 3 if self.asds else 2
        
        self.frequency_domain_resnet = ResNet1D(
            in_channels=int(num_ifos * channels_per_ifo),
            layers=freq_layers,
            classes=freq_context_dim,
            kernel_size=freq_kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )

        if fusion:
            self.fusion_layer = GatedFusion(
                time_dim=time_context_dim,
                freq_dim=freq_context_dim,
                fusion_dim=time_context_dim + freq_context_dim,
            )
        else:
            self.fusion_layer = lambda x, y: torch.concat((x, y), dim=1)


    def forward(self, X):
        strain, asds = X

        time_domain_embedded = self.time_domain_resnet(strain)

        # if asds is specified, append it to frequency domain data
        X_fft = torch.fft.rfft(strain)
        if self.asds:
            asds *= 1e23
            asds = asds.float()
            inv_asds = 1 / asds
            X_fft = X_fft[..., -asds.shape[-1] :]
            X_fft = torch.cat((X_fft.real, X_fft.imag, inv_asds), dim=1)
        else:
            X_fft = torch.cat((X_fft.real, X_fft.imag), dim=1)
        
        frequency_domain_embedded = self.frequency_domain_resnet(X_fft)

        # fusion instead (or concatenation)
        embedding = self.fusion_layer(
            time_domain_embedded, frequency_domain_embedded
        )

        return embedding


class FrequencyPsd(Embedding):
    """
    Single embedding for frequency domain data with ASDS
    """

    def __init__(
        self,
        num_ifos: int,
        context_dim: int,
        layers: list[int],
        kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: Optional[list[Literal["stride", "dilation"]]] = None,
        norm_layer: Optional[NormLayer] = None,
    ):
        super().__init__()
        self.freq_psd_resnet = ResNet1D(
            in_channels=int(num_ifos * 3),
            layers=layers,
            classes=context_dim,
            kernel_size=kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )
        self.context_dim = context_dim

    def forward(self, X):
        strain, asds = X

        asds *= 1e23
        asds = asds.float()
        inv_asds = 1 / asds

        X_fft = torch.fft.rfft(strain)
        X_fft = X_fft[..., -asds.shape[-1] :]
        X_fft = torch.cat((X_fft.real, X_fft.imag, inv_asds), dim=1)
        embedding = self.freq_psd_resnet(X_fft)

        return embedding


    def forward(self, X):
        strain, asds = X
        time_domain_embedded = self.time_domain_resnet(strain)
        frequency_domain_embedded = self.freq_domain_transformer(strain, asds)

        # Fusion instead of concatenation
        embedding = self.fusion_layer(
            time_domain_embedded, frequency_domain_embedded
        )

        return embedding

class MultiModalFreqTransformer(Embedding):
    """
    MultiModal embedding network that embeds both time and frequency data.

    We pass the data through their own ResNets defined by their layers
    and context dims, then concatenate the output embeddings.
    """

    def __init__(
        self,
        num_ifos: int,
        time_context_dim: int,
        freq_context_dim: int,
        time_layers: list[int],
        freq_layers: int = 4,
        d_model: int = 32,
        proj_dim: int = 512,
        num_heads: int = 4,
        time_kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: Optional[list[Literal["stride", "dilation"]]] = None,
        norm_layer: Optional[NormLayer] = None,
        fusion: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.context_dim = time_context_dim + freq_context_dim
        self.time_domain_resnet = ResNet1D(
            in_channels=num_ifos,
            layers=time_layers,
            classes=time_context_dim,
            kernel_size=time_kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )

        self.freq_domain_transformer = Transformer(
            num_ifos * 3,
            4036,
            proj_dim,
            freq_context_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=freq_layers,
        )

        if fusion:
            self.fusion_layer = GatedFusion(
                time_dim=time_context_dim,
                freq_dim=freq_context_dim,
                fusion_dim=time_context_dim + freq_context_dim,
            )
        else:
            self.fusion_layer = lambda x, y: torch.concat((x, y), dim=1)

    def forward(self, X):
        strain, asds = X

        asds *= 1e23
        asds = asds.float()
        inv_asds = 1 / asds

        time_domain_embedded = self.time_domain_resnet(strain)
        X_fft = torch.fft.rfft(strain)
        X_fft = X_fft[..., -asds.shape[-1] :]
        X_fft = torch.cat((X_fft.real, X_fft.imag, inv_asds), dim=1)

        frequency_domain_embedded = self.freq_domain_transformer(X_fft)

        # Fusion instead of concatenation
        embedding = self.fusion_layer(
            time_domain_embedded, frequency_domain_embedded
        )

        return embedding