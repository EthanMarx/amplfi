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
        **polarizations: torch.Tensor
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
    
def time_delay_from_geocenter(
    theta, phi, vertices, sample_rate
):
    omega = torch.column_stack(
        [
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta),
        ]
    ).view(-1, 1, 3)


    dt = -(omega * vertices).sum(axis=-1)
    return dt

def compute_observed_strain_frequency(
    dec, psi, phi, detector_tensors, detector_vertices, sample_rate, frequencies, **polarizations
):
    
    theta = torch.pi / 2 - dec
    antenna_responses = gw.compute_antenna_responses(
        theta, psi, phi, detector_tensors, list(polarizations)
    )


    # cast antenna responses to complex

    polarizations = torch.stack(list(polarizations.values()), axis=1)
    antenna_responses = antenna_responses.type(polarizations.dtype)

    waveforms = torch.einsum(
        "...pi,...pt->...it", antenna_responses, polarizations
    )

    dt = time_delay_from_geocenter(theta, phi, detector_vertices, sample_rate)
    frequencies = frequencies[None,None]
    dt = dt[...,None]
    # calculate phase shift for each detector

    phase_shift = 2 * torch.pi * frequencies * dt

    # apply phase shift
    waveforms = waveforms * torch.exp(-1j * phase_shift)
    return waveforms


class FrequencyWhitener(torch.nn.Module):

    def forward(self, X: Tensor, psds: Tensor) -> Tensor:
        psds = torch.nn.functional.interpolate(psds, size=X.size(-1), mode="linear")
        whitened = X / torch.sqrt(psds)
        mask = torch.isnan(whitened)
        whitened[mask] = 0
        return whitened

class FrequencyDomainProjector(WaveformProjector):
    def __init__(self, *args, frequencies: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("frequencies", frequencies)

    def forward(
        self,
        dec: torch.Tensor,
        psi: torch.Tensor,
        phi: torch.Tensor,
        **polarizations: torch.Tensor
    ):
        return compute_observed_strain_frequency(
            dec,
            psi,
            phi,
            self.tensors,
            self.vertices,
            self.sample_rate,
            self.frequencies,
            **polarizations,
        )
        

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
