import sys

from multiprocessing import Pool
from typing import Any

import numpy as np
import numpy.typing as npt

import emcee

from borehole_temperature_models.ModelMeasured import ModelMeasured


# ----------------------------------------------------------------------
DEFAULT_NUM_WALKERS                         = 256
DEFAULT_NUM_ITERATIONS                      = 1000
DEFAULT_NUM_BURNIN_ITERATIONS               = 100


# ----------------------------------------------------------------------
def Simulate(
    z: npt.NDArray[npt.NDArray[np.float64]],
    Tmeasured: npt.NDArray[npt.NDArray[np.float64]],
    Terr: float,
    p0: npt.NDArray[npt.NDArray[np.float64]],
    *,
    num_walkers: int=DEFAULT_NUM_WALKERS,
    num_iterations: int=DEFAULT_NUM_ITERATIONS,
    num_burnin_iterations: int=DEFAULT_NUM_BURNIN_ITERATIONS,
    no_progress: bool=False,
) -> tuple[npt.NDArray[npt.NDArray[np.float64]], npt.NDArray[npt.NDArray[np.float64]]]:
    with Pool() as pool:
        num_dimensions = len(p0[0])

        sampler = emcee.EnsembleSampler(
            num_walkers,
            num_dimensions,
            _lnprob,
            args=(z, Tmeasured, Terr),
            pool=pool,
        )

        if not no_progress:
            sys.stdout.write("Running burn-in...")

        p0, _, _ = sampler.run_mcmc(p0, num_burnin_iterations, progress=not no_progress)
        sampler.reset()

        if not no_progress:
            sys.stdout.write("Running production...")

        sampler.run_mcmc(p0, num_iterations, progress=not no_progress)

        return sampler.flatchain, sampler.flatlnprobability


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def _lnprob(
    data: npt.NDArray,
    x: npt.NDArray,
    y: npt.NDArray,
    yerr: float,
) -> np.float64:
    lp = _lnprior(data)
    if not np.isfinite(lp):
        return -np.inf

    return lp + _lnlike(data, y, yerr)


# ----------------------------------------------------------------------
def _lnlike(
    data: npt.NDArray,
    y: npt.NDArray,
    yerr: float,
) -> np.float64:
    return -0.5 * np.sum(((y - ModelMeasured(*data)) / yerr) ** 2)


# ----------------------------------------------------------------------
def _lnprior(
    data: npt.NDArray,
) -> float:
    Hinitial, t_grounding, G, a_yr, T_sim = data

    if (
        350 < Hinitial < 470
        and 5 < t_grounding < 10000
        and 0.06 < G < 0.09
        and 0.05 < a_yr < 0.2
        and -26 < T_sim < -22
    ):
        return 0.0

    return -np.inf
