from typing import Callable, Dict

import torch


class ParameterTransformer(torch.nn.Module):
    def __init__(self, transforms: Dict[str, Callable]):
        super().__init__()
        self.transforms = transforms

    def forward(
        self,
        parameters: Dict[str, torch.Tensor],
    ):
        # transform parameters
        transformed = {k: v(parameters[k]) for k, v in self.transforms.items()}
        # update parameter dict
        parameters.update(transformed)
        return parameters


class ParameterSampler(torch.nn.Module):
    def __init__(self, parameters: Dict[str, Callable]):
        super().__init__()
        self.parameters = parameters

    def forward(
        self,
        N: int,
        device: str = "cpu",
    ):
        parameters = {k: v(N).to(device) for k, v in self.parameters.items()}
        return parameters
