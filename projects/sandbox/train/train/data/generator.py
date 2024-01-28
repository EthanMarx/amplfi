from typing import TYPE_CHECKING, Callable, Dict

import numpy as np
import torch
from train.data.base import BaseDataset
from train.data.utils import ParameterSampler

if TYPE_CHECKING:
    from ml4gw.transforms import ChannelWiseScaler


class WaveformGeneratorDataset(BaseDataset):
    """
    Waveform dataset that generates waveforms on the fly

    Args:
        waveform_generator:
            Callable that takes a dictionary of Tensors, each of length `N`,
            as input and returns a Tensor containing `N`
            waveforms of shape `(N, num_ifos, strain_dim)`
        parameter_sampler:
            Callable that takes an integer `N` as input and
            returnes a dictionary of Tensors, each of length `N`
        num_val_waveforms:
            Total number of validaton waveforms to use.
            This total will be split up among all devices
        num_fit_params: N
            Number of parameters to use for fitting the standard scaler
    """

    def __init__(
        self,
        *args,
        waveform_generator: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
        parameter_sampler: ParameterSampler,
        num_val_waveforms: int = 10000,
        num_fit_params: int = 100000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_val_waveforms = num_val_waveforms
        self.num_fit_params = num_fit_params
        self.waveform_generator = waveform_generator
        self.parameter_sampler = self.build_parameter_sampler(
            parameter_sampler
        )

    @property
    def val_waveforms_per_device(self):
        world_size, _ = self.get_world_size_and_rank()
        return self.num_val_waveforms // world_size

    def fit_scaler(self, scaler: "ChannelWiseScaler") -> "ChannelWiseScaler":
        # sample parameters from parameter sampler
        # so we can fit the standard scaler
        parameters = self.parameter_sampler(self.num_fit_params)
        dec, phi, psi = self.sample_extrinsic(self.num_fit_params)
        parameters.update({"dec": dec, "phi": phi, "psi": psi})

        # downselect only to those requested to do inference on
        fit_params = {
            k: v for k, v in parameters.items() if k in self.inference_params
        }

        # transform any relevant parameters
        transformed = self.transform(fit_params)

        parameters = np.row_stack(transformed)
        scaler.fit(parameters)
        return scaler

    def get_val_waveforms(self):
        return self.sample_waveforms(self.val_waveforms_per_device)

    def sample_waveforms(self, N: int):
        # sample intrinsic parameters and generate
        # intrinsic h+ and hx polarizations
        parameters = self.parameter_sampler(N)
        waveforms = self.waveform_generator(**parameters)
        return waveforms, parameters
