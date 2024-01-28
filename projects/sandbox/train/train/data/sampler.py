from typing import TYPE_CHECKING

import h5py
import torch
from train.data.base import BaseDataset

if TYPE_CHECKING:
    from ml4gw.transforms import ChannelWiseScaler

EXTRINSIC_PARAMS = ["dec", "phi", "psi"]


def x_per_y(x, y):
    return int((x - 1) // y) + 1


# TODO: add ability to sample from disk for larger datasets
class WaveformSamplerDataset(BaseDataset):
    """
    Waveform dataset that for sampling waveforms in memory
    """

    @property
    def waveform_file(self):
        return self.data_dir / "signals.hdf5"

    # ================================================ #
    # Utilities for loading and splitting
    # waveforms among (potentially) several devices
    # ================================================ #
    def get_slice_bounds(self, total, world_size, rank) -> tuple[int, int]:
        """
        Figure which chunk of waveforms we should be
        slicing given our rank and world size
        """
        per_dev = x_per_y(abs(total), world_size)
        start = rank * per_dev
        stop = (rank + 1) * per_dev
        return start, stop

    def load_parameters(self, f, start, stop):
        # only load in parameters that we
        # have requested to do inference on
        parameters = {}
        for param in self.inference_params:
            try:
                values = f[param][start:stop]
            except KeyError:
                if param not in EXTRINSIC_PARAMS:
                    raise KeyError(f"Parameter {param} not found in dataset")
            parameters[param] = values

        # waveforms are already generated so we can
        # perform any transforms on the parameters here
        parameters = self.transform(parameters)
        parameters = torch.cat(parameters, dim=0)
        return parameters

    def load_signals(self, dataset, start, stop):
        """
        Loads waveforms assuming that the coalescence
        is in the middle, but we should really stop
        this. TODO: stop this
        """
        size = int(dataset.shape[-1] // 2)
        pad = int(0.02 * self.sample_rate)
        signals = torch.Tensor(dataset[start:stop, : size + pad])
        self._logger.info("Waveforms loaded")

        cross, plus = signals[:, 0], signals[:, 1]
        return cross, plus

    def load_train_signals(self, f, world_size, rank):
        dataset = f["signals"]
        num_valid = int(len(dataset) * self.hparams.valid_frac)
        num_train = len(dataset) - num_valid
        if not rank:
            self._logger.info(
                f"Training on {num_train} waveforms, with {num_valid} "
                "reserved for validation"
            )

        start, stop = self.get_slice_bounds(num_train, world_size, rank)
        waveforms = self.load_signals(dataset, start, stop)
        parameters = self.load_parameters(f, start, stop)
        return waveforms, parameters

    def get_val_waveforms(self, world_size, rank):
        with h5py.File(self.waveform_file, "r") as f:
            # infer the total number of validation signals,
            # and calculate how many validation signals to load
            # for this device
            dataset = f["signals"]
            total = int(len(dataset) * self.hparams.valid_frac)
            stop, start = self.get_slice_bounds(total, world_size, rank)
            self._logger.info(f"Loading {start - stop} validation signals")
            start, stop = -start, -stop or None

            # load in signals and parameters from this slice
            waveforms = self.load_signals(dataset, start, stop)
            parameters = self.load_parameters(f, start, stop)
        return waveforms, parameters

    def fit_scaler(self, scaler: "ChannelWiseScaler") -> "ChannelWiseScaler":
        # use training waveform parameters to fit standard scaler
        scaler.fit(self.train_parameters)
        return scaler

    def setup(self):
        world_size, rank = super().setup()

        self._logger.info("Loading training waveforms")
        with h5py.File(self.waveform_file, "r") as f:
            train_waveforms, train_parameters = self.load_train_signals(
                f, world_size, rank
            )

        self.train_waveforms = train_waveforms
        self.train_parameters = train_parameters

    def sample_waveforms(self, N: int):
        idx = torch.randperm(len(self.train_waveforms))[:N]
        waveforms = self.train_waveforms[idx]
        parameters = {k: v[idx] for k, v in self.train_parameters.items()}
        return waveforms, parameters
