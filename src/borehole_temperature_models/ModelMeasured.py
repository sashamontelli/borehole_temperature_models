from typing import Any

import numpy as np
import numpy.typing as npt

from scipy.interpolate import InterpolatedUnivariateSpline, interp1d

from borehole_temperature_models import Constants
from borehole_temperature_models import Utilities


# ----------------------------------------------------------------------
def ModelMeasured(
    Hinitial: float,
    t_grounding: float,
    G: float,
    a_yr: float,
    T_sim: float,
    z: npt.NDArray[np.float64],
    *,
    Hr: float=500.0,
    Hend: float=470.0,
    dz: float=10.0,
) -> npt.NDArray[np.float64]:
    zi = np.linspace(0, Hinitial, int(Hinitial/dz) + 1)  # Space calculation domain for ice
    zr = np.linspace(-Hr, 0, int(Hr/dz) + 1) #  Space calculation domain for rock
    l = len(zr)  # Index for ice-bed interface to be used in the loop calculations
    Lzi = len(zi) #  Number of elements in space domain for ice

    a = a_yr/Constants.yr_to_s

    Hisim = [Hinitial, Hend]

    dHyr = (Hend-Hinitial)/t_grounding
    dH = dHyr/Constants.yr_to_s

    time_sim_Hi = np.linspace(1, int(t_grounding), num=len(Hisim))
    tmnew = np.linspace(1, int(t_grounding), int(t_grounding), endpoint=True)
    fthk = interp1d(time_sim_Hi, Hisim)
    thkhistory = fthk(tmnew)

    dzr = dz

    z = np.concatenate((zr[:-1],zi))

    Ts_surf_steady = T_sim
    Ts = -z*0 + Ts_surf_steady

    for i in range(0, int(15e3)):

        Hi = thkhistory[0]
        zi = np.linspace(0, int(Hi), Lzi)
        dz = int(Hi/Lzi)

        z = np.concatenate((zr[:-1],zi))

        # Ice shelf vertical velocity profile

        w = -np.linspace(a,a,Lzi) # Accumulation rate and basal melt/freeze rate are in balance

        # Above ice-rock interface

        Ts[l:-1] = Ts[l:-1] + Constants.dt*(Constants.alpha_i*(Ts[l+1:]-2*Ts[l:-1]+Ts[l-1:-2])/dz**2 - np.multiply(w[1:-1],(Ts[l+1:]-Ts[l-1:-2])/(2*dz)))
        Ts[-1] = Ts_surf_steady # Temperature forcing is constant and equals the first value of the imported Monte-Carlo surface temperature vectors

        Ts[l-1] = -1.89 - 7.53e-4*Hi


        # Below ice-rock interface

        Ts[1:l-1] = Ts[1:l-1] + Constants.dt*Constants.alpha_r*(Ts[2:l]-2*Ts[1:l-1]+Ts[0:l-2])/dzr**2
        Ts[0] = Ts[1] + (G/Constants.k_r*dzr)

        # Recording the result into an empty matrix for comparison

    for j in range(0, int(t_grounding)):

        Hi = thkhistory[j]
        zi = np.linspace(0, Hi, Lzi)
        dz = Hi/Lzi
        z = np.concatenate((zr[:-1],zi))

        #  Interpolating previous temperature profile on to the new ice column space domain

        zint = np.linspace(0, thkhistory[j-1], Lzi)
        s = InterpolatedUnivariateSpline(zint, Ts[l-1:], k=1)
        Ts[l-1:] = s(zi)

        # Grounded ice vertical velocity profile

        dws = a - dH # When thickness is variable through time
        wzt = (1 - (((Constants.n + 2) / (Constants.n + 1)) * (1 - zi / Hi)) + (1 / (Constants.n + 1)) * np.power((1 - zi / Hi), (Constants.n + 2)))
        ws = -dws * wzt

        # Temperature calculation above ice-rock interface

        Ts[l:-1] = Ts[l:-1] + Constants.dt*(Constants.alpha_i*(Ts[l+1:]-2*Ts[l:-1]+Ts[l-1:-2])/dz**2 - np.multiply(ws[1:-1],(Ts[l+1:]-Ts[l-1:-2])/(2*dz)))
        Ts[-1] = T_sim # Temperature forcing is constant and equals the first value of the imported Monte-Carlo surface temperature vectors

        # Temperature calculation below ice-rock interface

        Ts[1:l] = Ts[1:l] + Constants.dt * Constants.alpha_r * (Ts[2:l+1]-2*Ts[1:l]+Ts[0:l-1])/dzr**2
        Ts[0] = Ts[1] + (G/Constants.k_r*dzr)

    Tx = Utilities.rsmpl(Ts[l-1:], zi, 10)[0]

    return Tx
