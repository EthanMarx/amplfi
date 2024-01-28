from typing import Callable

import torch
from pyro.distributions import ConditionalTransformedDistribution, transforms


class FlowArchitecture(torch.nn.Module):
    def __init__(
        self, num_params: int, context_dim: int, embedding_net: torch.nn.Module
    ):
        super().__init__()

        self.num_params = num_params
        self.context_dim = context_dim
        self.embedding_net = embedding_net

        # build the transform - sets the transforms attrib
        self.build_flow()

    @property
    def transform_block(
        self, *args, **kwargs
    ) -> Callable[[int], transforms.Transform]:
        raise NotImplementedError

    @property
    def distribution(self) -> torch.distributions.Distribution:
        raise NotImplementedError

    def build_flow(self) -> None:
        """Sets the ``transforms`` attribute"""
        raise NotImplementedError

    @property
    def flow(self):
        return ConditionalTransformedDistribution(
            self.distribution(), self.transforms
        )

    def log_prob(self, x, context):
        """Wrapper around :meth:`log_prob` from
        `TransformedDistribution` object.
        """
        if not hasattr(self, "transforms"):
            raise RuntimeError("Flow is not built")

        embedded_context = self.embedding_net(context)
        return self.flow.condition(embedded_context).log_prob(x)

    def sample(self, n, context):
        """Wrapper around :meth:`sample` from
        `TransformedDistribution` object.
        """
        if not hasattr(self, "transforms"):
            raise RuntimeError("Flow is not built")

        embedded_context = self.embedding_net(context)
        n = [n] if isinstance(n, int) else n
        return self.flow.condition(embedded_context).sample(n)
