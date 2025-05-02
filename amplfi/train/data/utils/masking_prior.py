import torch


class IfoMasker:
    """
    Class to sample an interferometer mask based on a given prior distribution.

    Interferometer masking is used to train a single neural network that can
    handle different combinations of interferometers.

    Args:
        prior:
            A dictionary where the keys are strings of comma separated
            interferometer names and the values are the fraction of each batch
            where that interferometer combination will have data present.
        ifos:
            A list of all the available interferometers.
    """

    def __init__(self, prior: dict[str, float], ifos: list[str]):
        self.ifos = ifos

        # convert the string of comma separated interferometer names to a tuple
        prior = {tuple(ifos.split(",")): prob for ifos, prob in prior.items()}

        for ifo_set in prior.keys():
            if not all(ifo in self.ifos for ifo in ifo_set):
                raise ValueError(
                    f"Interferometer mask {ifo_set} not in list "
                    f"of interferometers {self.ifos}"
                )

        # check that the prior is a valid probability distribution
        total_prob = 0.0
        for prob in prior.values():
            total_prob += prob

        if total_prob != 1.0:
            raise ValueError(
                "Interferometer masking prior probabilities must sum to 1.0"
                f", but got {total_prob}"
            )

        self.prior = prior

    def sample_mask(self, N: int, device: str) -> torch.Tensor:
        """
        Sample a mask for the given number of samples
        """
        mask = torch.zeros(
            (N, len(self.ifos)), dtype=torch.bool, device=device
        )
        indices = torch.randperm(N, device=device)
        sizes = [int(N * prob) for prob in self.prior.values()]

        # adjust the last size to ensure the total size is N
        adjustment = N - sum(sizes)
        sizes[-1] += adjustment

        start_idx = 0
        for ifos, size in zip(self.prior.keys(), sizes, strict=False):
            # channel indices corresponding to ifo combination
            ifo_indices = torch.tensor(
                [self.ifos.index(ifo) for ifo in ifos], device=device
            )
            batch_indices = indices[start_idx : start_idx + size]
            start_idx += size
            mask[batch_indices, ifo_indices.unsqueeze(1)] = 1

        # add extra dimension for broadcasting along
        # the time dimension of the data tensor
        return mask[..., None]

    def __str__(self):
        output = ""
        for ifos, prob in self.prior.values():
            output += f"{ifos} : {prob}\n"
        return output
