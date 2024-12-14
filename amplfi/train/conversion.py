from typing import Dict

import torch
from ml4gw.waveforms.conversion import bilby_spins_to_lalsim


def components_from_chirp_mass_and_mass_ratio(chirp_mass, mass_ratio):
    """
    Compute component masses from chirp mass and mass ratio
    """
    total_mass = chirp_mass * (1 + mass_ratio) ** 1.2 / mass_ratio**0.6
    mass_1 = total_mass / (1 + mass_ratio)
    mass_2 = mass_1 * mass_ratio
    return mass_1, mass_2


def to_lalsimulation_parameters(parameters: Dict[str, torch.Tensor]):
    """
    Append lalsimulation spins to parameter dictionary
    """
    mass_1, mass_2 = components_from_chirp_mass_and_mass_ratio(
        parameters["chirp_mass"], parameters["mass_ratio"]
    )
    iota, s1x, s1y, s1z, s2x, s2y, s2z = bilby_spins_to_lalsim(
        parameters["theta_jn"],
        parameters["phi_jl"],
        parameters["tilt_1"],
        parameters["tilt_2"],
        parameters["phi_12"],
        parameters["a_1"],
        parameters["a_2"],
        mass_1,
        mass_2,
        40,
        parameters["phic"],
    )

    parameters.update(
        {
            "mass_1": mass_1,
            "mass_2": mass_2,
            "s1x": s1x,
            "s1y": s1y,
            "s1z": s1z,
            "s2x": s2x,
            "s2y": s2y,
            "s2z": s2z,
            "inclination": iota,
        }
    )
    return parameters
