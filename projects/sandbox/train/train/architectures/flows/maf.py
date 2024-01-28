from typing import Callable

import torch
import torch.distributions as dist
from pyro.distributions.conditional import ConditionalComposeTransformModule
from pyro.distributions.transforms import ConditionalAffineAutoregressive
from pyro.nn import ConditionalAutoRegressiveNN
from train.architectures.flows import FlowArchitecture


class MaskedAutoRegressiveFlow(FlowArchitecture):
    def __init__(
        self,
        *args,
        hidden_features: int = 50,
        num_transforms: int = 5,
        num_blocks: int = 2,
        activation: Callable = torch.tanh,
        **kwargs,
    ):
        self.hidden_features = hidden_features
        self.num_blocks = num_blocks
        self.num_transforms = num_transforms
        self.activation = activation
        super().__init__(*args, **kwargs)

    def transform_block(self):
        """Returns single autoregressive transform"""
        arn = ConditionalAutoRegressiveNN(
            self.num_params,
            self.context_dim,
            self.num_blocks * [self.hidden_features],
            nonlinearity=self.activation,
        )
        return ConditionalAffineAutoregressive(arn)

    def distribution(self):
        """Returns the base distribution for the flow"""
        return dist.Normal(
            torch.zeros(self.param_dim, device=self.device),
            torch.ones(self.param_dim, device=self.device),
        )

    def build_flow(self):
        """Build the transform"""
        self.transforms = []
        for _ in range(self.num_transforms):
            _transform = self.transform_block()
            self.transforms.extend([_transform])
        self.transforms = ConditionalComposeTransformModule(self.transforms)
