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


class Decimator(torch.nn.Module):
    """
    Downsample a timeseries via decimation.

    Args:
        initial_sample_rate: 
            The sample of rate of the initial
            timeseries being downsampled
        decimate_schedule:
            A list of tuples (start, end, target_sample_rate)
            where start and end correspond to the portion of 
            the original timeseries in seconds, and target_sample_rate
            corresponds to sample rate that portion will be resampled to. 
            Note that consecutive start and ends should match.
    """
    def __init__(
        self, 
        initial_sample_rate: float,
        decimate_schedule: list[tuple[float, float, float]],
    ):

        super().__init__()
        self.decimate_schedule = decimate_schedule
        self.initial_sample_rate = initial_sample_rate

        indices = self.build_indices() 
        self.register_buffer("indices", indices)

    def build_indices(self):
        start, end, target_sr = self.decimate_schedule.T
        samples = ((end - start) * target_sr).long()
        rel_idx = torch.arange(samples.max()).expand(len(samples), -1)
        mask = rel_idx < samples.unsqueeze(1)
        abs_idx = (start.unsqueeze(1) * self.initial_sample_rate + rel_idx * (self.initial_sample_rate / target_sr.unsqueeze(1))).long()
        return abs_idx[mask]

    def forward(self, X):
        return X.index_select(dim=-1, index=self.indices)

