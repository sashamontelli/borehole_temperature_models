import math
import sys
import time

from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
import numpy.typing as npt

from scipy.interpolate import interp1d


# ----------------------------------------------------------------------
def RMS(
    measured: npt.NDArray[np.float64],
    model: npt.NDArray[np.float64],
) -> float:
    """Root mean square standard error"""

    d = 0
    for i in range(0,len(measured)):
        d1 = (measured[i] - model[i])**2
        d = d + d1
    RMS = math.sqrt(d/(len(measured)))
    return RMS


# ----------------------------------------------------------------------
def rsmpl(
    T_in: npt.NDArray[np.float64],
    z_in: npt.NDArray[np.float64],
    dz: float,
) -> list[npt.NDArray[np.float64]]: # TODO: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Resample temperature profiles into a required number of data points/vertical grid"""

    flinear = interp1d(z_in, T_in)

    z_out = np.arange(0, int(max(z_in))+dz, dz)
    T_out = flinear(z_out)

    return [T_out, z_out]


# ----------------------------------------------------------------------
@contextmanager
def Timer(
    header: str="",
) -> Iterator[None]:
    counter = time.perf_counter()

    try:
        yield
    finally:
        elapsed = time.perf_counter() - counter

        seconds_per_hour = 60 * 60

        hours = int(elapsed / seconds_per_hour)
        elapsed -= hours * seconds_per_hour

        minutes = int(elapsed / 60)
        elapsed -= minutes * 60

        sys.stdout.write(
            "{}{}:{:02d}:{:.02f}\n".format(
                "{}:".format(header) if header else "",
                hours,
                minutes,
                elapsed,
            ),
        )
