from typing import Optional

import torch


class Embedding(torch.nn.Module):
    """
    Base class for embedding networks.
    """

    def __init__(
        self,
        num_ifos: int,
        context_dim: int,
        strain_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_ifos = num_ifos
        self.context_dim = context_dim
        self.strain_dim = strain_dim
