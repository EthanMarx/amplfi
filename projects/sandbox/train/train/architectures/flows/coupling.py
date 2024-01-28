import torch
import torch.distributions as dist
from pyro.distributions.conditional import ConditionalComposeTransformModule
from pyro.distributions.transforms import ConditionalAffineCoupling
from pyro.nn import ConditionalDenseNN
from train.architectures.flows import FlowArchitecture


class CouplingFlow(FlowArchitecture):
    def __init__(
        self,
        *args,
        hidden_features: int = 512,
        num_transforms: int = 5,
        num_blocks: int = 2,
        activation: torch.nn.modules.activation = torch.nn.ReLU,
        **kwargs,
    ):

        self.split_dim = self.num_params // 2
        self.hidden_features = hidden_features
        self.num_blocks = num_blocks
        self.num_transforms = num_transforms
        self.activation = activation

        super().__init__(*args, **kwargs)

    def transform_block(self):
        """Returns single affine coupling transform"""
        arn = ConditionalDenseNN(
            self.split_dim,
            self.context_dim,
            [self.hidden_features],
            param_dims=[
                self.num_params - self.split_dim,
                self.num_params - self.split_dim,
            ],
            nonlinearity=self.activation,
        )
        transform = ConditionalAffineCoupling(self.split_dim, arn)
        return transform

    def distribution(self):
        """Returns the base distribution for the flow"""
        return dist.Normal(
            torch.zeros(self.num_params, device=self.device),
            torch.ones(self.num_params, device=self.device),
        )

    def build_flow(self):
        self.transforms = []
        for _ in range(self.num_transforms):
            self.transforms.extend([self.transform_block()])

        self.transforms = ConditionalComposeTransformModule(self.transforms)
