import torch

import torch.nn as nn 
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Callable, List
from amplfi.architectures.embeddings.base import Embedding


class ComplexSvdDecompressor(torch.nn.Module):
    def __init__(self, Vh: torch.Tensor):
        super().__init__()
        basis_dim, data_dim = Vh.shape

        # create single real valued matrix that is equivalent 
        # to performing matrix multiplication in complex space;
        # use numpy since it has convenient block matrix creation
        Vh = Vh.resolve_conj().cpu().numpy()
        block = np.block([[Vh.real, -Vh.imag], [Vh.imag, Vh.real]])
        block = torch.as_tensor(block)

        layer = nn.Linear(2 * basis_dim, 2 * data_dim, bias=False)
        layer.weight.data[...] = block.transpose(0, 1)
        self.layer = layer
    
    def forward(self, x):
        return self.layer(x)


class ComplexSvdCompressor(torch.nn.Module):
    """
    Embed an SVD compression matrix into a single linear layer

    The forward method expects a tensor of size `(batch_size, 2, data_dim)` 
    where the first channel is the real component of the signal and the second 
    channel is the complex component of the signal. 
    
    Returns a tensor of size `(batch_size, 1, 2 * basis_dim)`

    Args:
        V: 
            Complex tensor of shape `(data_dim, basis_dim)` 
            where `data_dim` is the number of complex numbers in the data, 
            and `basis_dim` is the dimension of the compression
        Vh: 
            Complex tensor of shape `(basis_dim, data_dim)` 
            where `data_dim` is the number of complex numbers in the data, 
            and `basis_dim` is the dimension of the compression.
    """
    def __init__(self, V: torch.Tensor):
        super().__init__()

        # extract dimensions of data and 
        # the compressed basis
        data_dim, basis_dim = V.shape

        # create single real valued matrix that is equivalent 
        # to performing matrix multiplication in complex space;
        # use numpy since it has convenient block matrix creation
        V = V.resolve_conj().cpu().numpy()
        block = np.block([[V.real, -V.imag], [V.imag, V.real]])
        block = torch.as_tensor(block)

        # set up single linear layer that performs the 
        # matrix multiplication in real space equivalent to compression
        layer = nn.Linear(2 * data_dim, 2 * basis_dim, bias=False)
        layer.weight.data[...] = block.transpose(0, 1)
        self.layer = layer
    
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = x.unsqueeze(1)
        return self.layer(x)
    



class ResidualBlock(nn.Module):
    """A general-purpose residual block. Works only with 1-dim inputs."""

    def __init__(
        self,
        features,
        context_features,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        zero_initialization=True,
    ):
        super().__init__()
        self.activation = activation

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(features, eps=1e-3) for _ in range(2)]
            )
        if context_features is not None:
            self.context_layer = nn.Linear(context_features, features)
        self.linear_layers = nn.ModuleList(
            [nn.Linear(features, features) for _ in range(2)]
        )
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)
            init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs, context=None):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        if context is not None:
            temps = F.glu(torch.cat((temps, self.context_layer(context)), dim=1), dim=1)
        return inputs + temps
    
class DenseResidualNet(nn.Module):
    """
    A nn.Module consisting of a sequence of dense residual blocks. This is
    used to embed high dimensional input to a compressed output. Linear
    resizing layers are used for resizing the input and output to match the
    first and last hidden dimension, respectively.

    Module specs
    --------
        input dimension:    (batch_size, input_dim)
        output dimension:   (batch_size, output_dim)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Tuple,
        activation: Callable = F.elu,
        dropout: float = 0.0,
        batch_norm: bool = True,
    ):
        """
        Parameters
        ----------
        input_dim : int
            dimension of the input to this module
        output_dim : int
            output dimension of this module
        hidden_dims : tuple
            tuple with dimensions of hidden layers of this module
        activation: callable
            activation function used in residual blocks
        dropout: float
            dropout probability for residual blocks used for reqularization
        batch_norm: bool
            flag that specifies whether to use batch normalization
        """

        super(DenseResidualNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.num_res_blocks = len(self.hidden_dims)

        self.initial_layer = nn.Linear(self.input_dim, hidden_dims[0])
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    features=self.hidden_dims[n],
                    context_features=None,
                    activation=activation,
                    dropout_probability=dropout,
                    use_batch_norm=batch_norm,
                )
                for n in range(self.num_res_blocks)
            ]
        )
        self.resize_layers = nn.ModuleList(
            [
                nn.Linear(self.hidden_dims[n - 1], self.hidden_dims[n])
                if self.hidden_dims[n - 1] != self.hidden_dims[n]
                else nn.Identity()
                for n in range(1, self.num_res_blocks)
            ]
            + [nn.Linear(self.hidden_dims[-1], self.output_dim)]
        )

    def forward(self, x):
        x = self.initial_layer(x)
        for block, resize_layer in zip(self.blocks, self.resize_layers):
            x = block(x, context=None)
            x = resize_layer(x)
        return x
    

class SVDDenseEnet(Embedding):
    def __init__(
        self,
        num_ifos, 
        context_dim, 
        strain_dim,
        hidden_dims: List[int], 
    ):
        super().__init__()
        self.num_ifos = num_ifos
        self.context_dim = context_dim
        self._fit = False
        self.net = DenseResidualNet(num_ifos * 2 * 200, context_dim, hidden_dims)

    def fit(self, Vs, Vhs):
        self.compressors = nn.ModuleList()
        self.decompressors = nn.ModuleList()
        for i in range(self.num_ifos):
            self.compressors.append(ComplexSvdCompressor(Vs[i]))
            self.decompressors.append(ComplexSvdDecompressor(Vhs[i]))
        self._fit = True

    def decompress(self, x):
        if not self._fit:
            raise ValueError("SVD basis has not been fit yet")
        # x is shape (batch_size, num_ifos, basis_dim)
        # decompress each ifo
        out = []
        for i, decompressor in enumerate(self.decompressors):
            data = x[:, i, :]
            data = decompressor(data)
            out.append(data)

        out = torch.stack(out, dim=1)
        return out

    def compress(self, x):
        out = []
        for i, compressor in enumerate(self.compressors):
            data = x[:, i, :]
            stacked = torch.stack([data.real, data.imag], dim=1)
            stacked = stacked.float()
            compressed = compressor(stacked)
            out.append(compressed)
        return out

    def forward(self, x):
        if not self._fit:
            raise ValueError("SVD basis has not been fit yet")
        # x is shape (batch_size, num_ifos, data_dim)
        # compress each ifo
        out = []
        for i, compressor in enumerate(self.compressors):
            data = x[:, i, :]
            stacked = torch.stack([data.real, data.imag], dim=1)
            stacked = stacked.float()
            compressed = compressor(stacked)
            out.append(compressed)

        out = torch.cat(out, dim=1)
        out = out.flatten(start_dim=1)

        x = self.net(out)
        return x