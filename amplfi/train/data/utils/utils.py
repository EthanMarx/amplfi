from typing import Callable, Dict, Optional

import torch


class ParameterTransformer(torch.nn.Module):
    """
    Helper class for applying preprocessing
    transformations to inference parameters
    """

    def __init__(self, **transforms: Callable):
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
    """
    Helper class for sampling inference parameters

    Args:
        conversion_fn:
            A callable that takes a dictionary of parameters
            and applies any necessary conversions. Useful
            for converting from an astrophysical prior to
            parameters used for waveform generation
        **parameters:
            A dictionary of parameter distributions

    """

    def __init__(
        self, conversion_fn: Optional[Callable] = None, **parameters: Callable
    ):

        super().__init__()
        self.parameters = parameters
        self.conversion_fn = conversion_fn or (lambda x: x)

    def forward(
        self,
        N: int,
        device: str = "cpu",
    ):

        # sample parameters from priors
        parameters = {
            k: v.sample((N,)).to(device) for k, v in self.parameters.items()
        }
        # apply any conversions, appending to parameter dics
        parameters = self.conversion_fn(parameters)
        return parameters


class ZippedDataset(torch.utils.data.IterableDataset):
    def __init__(self, *datasets):
        super().__init__()
        self.datasets = datasets

    def __len__(self):
        lengths = []
        for dset in self.datasets:
            try:
                lengths.append(len(dset))
            except Exception as e:
                raise e from None
        return min(lengths)

    def __iter__(self):
        return zip(*self.datasets)
