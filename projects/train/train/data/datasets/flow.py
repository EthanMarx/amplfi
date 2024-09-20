import torch

from train.data.datasets.base import AmplfiDataset
import numpy as np

class FlowDataset(AmplfiDataset):
    """
    Lightning DataModule for training normalizing flow networks
    """

    def inject(self, X, cross, plus, parameters):
    

        self.projector.to(self.device)
        self.whitener.to(self.device)

        X, psds = self.psd_estimator(X)
        dec, psi, phi = self.waveform_sampler.sample_extrinsic(X)
        waveforms = self.projector(dec, psi, phi, cross=cross, plus=plus)

        # append extrinsic parameters to parameters
        parameters.update({"dec": dec, "psi": psi, "phi": phi})

        # downselect to requested inference parameters
        parameters = {
            k: v
            for k, v in parameters.items()
            if k in self.hparams.inference_params
        }

        # make any requested parameter transforms
        freqs = torch.fft.rfftfreq(X.shape[-1], 1 / self.hparams.sample_rate)
        parameters = self.transform(parameters)
        parameters = [
            torch.Tensor(parameters[k]) for k in self.hparams.inference_params
        ]
        parameters = torch.vstack(parameters).T

        if self.domain == "frequency":
            X = X * self.window
            X = torch.fft.rfft(X) / self.hparams.sample_rate

        X += waveforms
        X = self.whitener(X, psds)
        X = X[..., freqs > self.hparams.highpass]

        # scale parameters
        parameters = self.scale(parameters)

        return X, parameters
    
