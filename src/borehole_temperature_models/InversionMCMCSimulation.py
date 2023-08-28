import sys

from multiprocessing import Pool
from typing import Union

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
    dt_years: float,
    z: npt.NDArray,
    Tmeasured: npt.NDArray,
    Terr: float,
    p0: npt.NDArray,
    *,
    h_constraints: tuple[float, float],
    grounding_constraints: tuple[int, int],
    G_constraints: tuple[float, float],
    yr_constraints: tuple[float, float],
    sim_constraints: tuple[int, int],
    num_walkers: int=DEFAULT_NUM_WALKERS,
    num_iterations: int=DEFAULT_NUM_ITERATIONS,
    num_burnin_iterations: int=DEFAULT_NUM_BURNIN_ITERATIONS,
    no_progress: bool=False,
) -> tuple[npt.NDArray, npt.NDArray]:
    if h_constraints[0] > h_constraints[1]:
        raise ValueError("Invalid 'h_constraints' input")
    if grounding_constraints[0] > grounding_constraints[1]:
        raise ValueError("Invalid 'grounding_constraints' input")
    if G_constraints[0] > G_constraints[1]:
        raise ValueError("Invalid 'G_constraints' input")
    if yr_constraints[0] > yr_constraints[1]:
        raise ValueError("Invalid 'yr_constraints' input")
    if sim_constraints[0] > sim_constraints[1]:
        raise ValueError("Invalid 'sim_constraints' input")

    with Pool() as pool:
        num_dimensions = len(p0[0])

        controller = _SamplerController(
            dt_years,
            h_constraints,
            grounding_constraints,
            G_constraints,
            yr_constraints,
            sim_constraints,
        )

        sampler = emcee.EnsembleSampler(
            num_walkers,
            num_dimensions,
            controller.lnprob,
            args=(z, Tmeasured, Terr),
            pool=pool,
        )

        if not no_progress:
            sys.stdout.write("Running burn-in...")

        result = sampler.run_mcmc(p0, num_burnin_iterations, progress=not no_progress)
        assert result is not None
        burnin_p0 = result[0]

        sampler.reset()

        if not no_progress:
            sys.stdout.write("Running production...")

        sampler.run_mcmc(burnin_p0, num_iterations, progress=not no_progress)

        assert sampler.flatchain is not None
        assert sampler.flatlnprobability is not None

        return sampler.flatchain, sampler.flatlnprobability


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
class _SamplerController(object):
    """Object that stores data used to control a MCMC simulation via emcee."""

    # ----------------------------------------------------------------------
    def __init__(
        self,
        dt_years: float,
        h_constraints: tuple[float, float],
        grounding_constraints: tuple[int, int],
        G_constraints: tuple[float, float],
        yr_constraints: tuple[float, float],
        sim_constraints: tuple[int, int],
    ):
        assert h_constraints[0] <= h_constraints[1]
        assert grounding_constraints[0] <= grounding_constraints[1]
        assert G_constraints[0] <= G_constraints[1]
        assert yr_constraints[0] <= yr_constraints[1]
        assert sim_constraints[0] <= sim_constraints[1]

        self.dt_years                       = dt_years
        self.h_constraints                  = h_constraints
        self.grounding_constraints          = grounding_constraints
        self.G_constraints                  = G_constraints
        self.yr_constraints                 = yr_constraints
        self.sim_constraints                = sim_constraints

    # ----------------------------------------------------------------------
    def lnprob(
        self,
        data: npt.NDArray,
        x: npt.NDArray,
        y: npt.NDArray,
        yerr: float,
    ) -> Union[float, np.float64]:
        lp = self.lnprior(data)
        if not np.isfinite(lp):
            return -np.inf

        return lp + self.lnlike(data, y, yerr)

    # ----------------------------------------------------------------------
    def lnlike(
        self,
        data: npt.NDArray,
        y: npt.NDArray,
        yerr: float,
    ) -> np.float64:
        return -0.5 * np.sum(((y - ModelMeasured(self.dt_years, *data)) / yerr) ** 2)

    # ----------------------------------------------------------------------
    def lnprior(
        self,
        data: npt.NDArray,
    ) -> float:
        Hinitial, t_grounding, G, a_yr, T_sim = data

        if (
            self.h_constraints[0] < Hinitial < self.h_constraints[1]
            and self.grounding_constraints[0] < t_grounding < self.grounding_constraints[1]
            and self.G_constraints[0] < G < self.G_constraints[1]
            and self.yr_constraints[0] < a_yr < self.yr_constraints[1]
            and self.sim_constraints[0] < T_sim  < self.sim_constraints[1]
        ):
            return 0.0

        return -np.inf
