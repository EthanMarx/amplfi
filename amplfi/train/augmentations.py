from typing import Optional, Tuple

import torch
from ml4gw import gw
from ml4gw.transforms import SpectralDensity

Tensor = torch.Tensor


class WaveformProjector(torch.nn.Module):
    def __init__(
        self,
        ifos: list[str],
        sample_rate: float,
    ) -> None:
        super().__init__()
        tensors, vertices = gw.get_ifo_geometry(*ifos)
        self.register_buffer("tensors", tensors)
        self.register_buffer("vertices", vertices)

        self.sample_rate = sample_rate

    def forward(
        self,
        dec: torch.Tensor,
        psi: torch.Tensor,
        phi: torch.Tensor,
        **polarizations: torch.Tensor,
    ) -> torch.Tensor:
        responses = gw.compute_observed_strain(
            dec,
            psi,
            phi,
            detector_tensors=self.tensors,
            detector_vertices=self.vertices,
            sample_rate=self.sample_rate,
            **polarizations,
        )
        return responses


class PsdEstimator(torch.nn.Module):
    """
    Module that takes a sample of data, splits it into
    two unequal-length segments, calculates the PSD of
    the first section, then returns this PSD along with
    the second section.

    Args:
        length:
            The length, in seconds, of timeseries data
            to be returned for whitening. Note that the
            length of time used for the PSD will then be
            whatever remains along first part of the time
            axis of the input.
        sample_rate:
            Rate at which input data has been sampled in Hz
        fftlength:
            Length of FFTs to use when computing the PSD
        overlap:
            Amount of overlap between FFT windows when
            computing the PSD. Default value of `None`
            uses `fftlength / 2`
        average:
            Method for aggregating spectra from FFT
            windows, either `"mean"` or `"median"`
        fast:
            If `True`, use a slightly faster PSD algorithm
            that is inaccurate for the lowest two frequency
            bins. If you plan on highpassing later, this
            should be fine.
    """

    def __init__(
        self,
        length: float,
        sample_rate: float,
        fftlength: float,
        overlap: Optional[float] = None,
        average: str = "mean",
        fast: bool = True,
    ) -> None:
        super().__init__()
        self.size = int(length * sample_rate)
        self.spectral_density = SpectralDensity(
            sample_rate, fftlength, overlap, average, fast=fast
        )

    def forward(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        splits = [X.size(-1) - self.size, self.size]
        background, X = torch.split(X, splits, dim=-1)

        # if we have 2 batch elements in our input data,
        # it will be assumed that the 0th element corresponds
        # to true background and the 1th element corresponds
        # to injected data, in which case we'll only compute
        # the background PSD on the former
        if X.ndim == 3 and X.size(0) == 2:
            background = background[0]

        self.spectral_density.to(device=background.device)
        psds = self.spectral_density(background.double())
        return X, psds


class IfoMasker(torch.nn.Module):
    """
    Torch module that will randomly mask ifo channels with zeros
    """
    def __init__(self, ifo_combos: list[tuple[int]]):
        self.ifo_combos = ifo_combos
    
    def forward(self, X: torch.Tensor):
        batch_size = X.shape[0]
        num_channels = X.shape[1]
        indices = torch.randperm(batch_size) 
        num_groups = len(self.ifo_combos)

        ends = [i * (batch_size // num_groups) for i in range(num_groups)]
        all_channels = set(range(num_channels)) 
        # TODO: dont think this is a perf hit 
        # but probably a way to avoid this for loop
        for idx_end, combo in zip(ends, self.ifo_combos):
            combo_indices = indices[] 
            masked_channels = torch.tensor(all_channels - set(combo))
            X[combo_indices, masked_channels, :] = 0.0
            
        return X
