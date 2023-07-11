# TODO: Remove imports that are not used
import base64
import json
import math
import multiprocessing as mp
import random
import time

from dataclasses import dataclass
from io import BytesIO
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, ClassVar, Optional

import cmocean
import dask
import dask.bag as db
import emcee
import fsspec
import gcsfs
import geopandas as gpd
import h5py
import matplotlib as mpl
import matplotlib.font_manager
import matplotlib.font_manager as fm
import netCDF4 as nc
import numpy as np
import pandas as pd
import pyproj
import scipy
import xarray as xr

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import Axes3D
from netCDF4 import Dataset
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from tqdm import tqdm


# ----------------------------------------------------------------------
# |
# |  Public Types
# |
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class TemperatureModel(object):
    # ----------------------------------------------------------------------
    # |  Data
    yr_to_s: ClassVar[float]                = 365.25*24*60*60 # Seconds in a year
    dt_years: ClassVar[int]                 = 1 # TODO: Is this a param?
    dt: ClassVar[float]                     = dt_years * yr_to_s

    c_i: ClassVar[float]                    = 2e3                           # Head capacity of ice
    c_r: ClassVar[float]                    = 7.9e2                         # Heat capacity of basal rock
    k_i: ClassVar[float]                    = 2.3                           # Thermal conductivity of ice
    k_r: ClassVar[float]                    = 2.8                           # Thermal conductivity of basal rock
    rho_i: ClassVar[float]                  = 918                           # Density of ice
    rho_r: ClassVar[float]                  = 2750                          # Density of basal rock
    alpha_i: ClassVar[float]                = k_i / rho_i / c_i             # Alpha for ice
    alpha_r: ClassVar[float]                = k_r / rho_r / c_r             # Alpha for basal rock
    n: ClassVar[float]                      = 3                             # Glen's flow law exponent
    g: ClassVar[float]                      = 9.81                          # Gravitational acceleration
    A: ClassVar[float]                      = 2e-16                         # Rheological constant determining soft ice as a function of temperature

    # ----------------------------------------------------------------------
    Tmeasured: np.ndarray
    z: np.ndarray
    Tvar_H_Tmatrix: np.ndarray
    zvar_H_Tmatrix: np.ndarray

    # ----------------------------------------------------------------------
    # |  Methods
    @classmethod
    def Create(
        cls,
        Hi_sim: list[int],                  # Ice thickness history H(t)
        Hr: int,                            # Time-constant thickness of subglacial bedrock
        dz: int,                            # Spatial grid resolution in metres "
        t_steady_years_total: float,        # Time (in years) required to reach steady state temperature profile
        t_forcing_years: float,             # Time period t (in years) when forcings are time-variable
        t_ungrounding: float,               # Time (years before present) when ice becomes ungrounded
        t_grounding: float,                 # Time (years before present) when ice becomes grounded
        T_sim: list[float],                 # Temperature history T(t)
        a_sim: list[float],                 # Accumulation history a(t)
        G: float,                           # Geothermal heat flux
    ) -> "TemperatureModel":
        # TODO: Assert valid parameters

        #### Defining spatial and temporal domains / resolution
        Hm = min(Hi_sim)  # Taking the minimum value of thickness history for domain calculation
        zi = np.linspace(0, Hm, int((Hm)/dz) + 1)  # Space calculation domain for ice
        dzr = dz  # Space resolution for bedrock
        zr = np.linspace(-Hr, 0, int(Hr/dz) + 1) #  Space calculation domain for bedrock
        l = len(zr)  # Index for ice-bed interface to be used in the loop calculations
        Lzi = len(zi) #  Number of elements in space domain for ice
        Lzr = len(zr) #  Number of elements in space domain for bedrock
        z = np.concatenate((zr[:-1],zi))  #  Merging space domains for ice and bedrock into one single unified space domain
        Lz = len(z) # Calculating the number of nodes in the unified space domain

        Tvar_H_matrix = np.zeros((Lz,int(t_forcing_years))) # Preparing temperature matrix for each step of temperature calculations
        zvar_H_matrix = np.zeros((Lz,int(t_forcing_years))) # Preparing depth matrix for each step of temperature calculations

        #### Interpolating time-variable forcing vectors

        time_sim_T = np.linspace(1, int(t_forcing_years), num=len(T_sim))  # Creating an evenly spaced time vector for input surface temperature vector
        time_sim_Hi = np.linspace(1, int(t_forcing_years), num=len(Hi_sim)) # Creating an evenly spaced time vector for input thickness vector
        time_sim_a = np.linspace(1, int(t_forcing_years), num=len(a_sim)) # Creating an evenly spaced time vector for input accumulation vector
        tmnew = np.linspace(1, t_forcing_years, num = int(t_forcing_years), endpoint=True) # Creating a uniform time vector with 1-year spacing
        f = interp1d(time_sim_T, T_sim) # Preparing interpolation function
        facc = interp1d(time_sim_a, a_sim) # Preparing interpolation function
        fthk = interp1d(time_sim_Hi, Hi_sim) # Preparing interpolation function

        thkhistory = fthk(tmnew) # Interpolating thickness vector over the new time vector
        temphistory = f(tmnew) # Interpolating surface temperature vector over the new time vector
        acchistory_yr = facc(tmnew) # Interpolating accumulation vector over the new time vector

        acchistory = acchistory_yr / cls.yr_to_s # Converting accumulation vector from m/yr to m/s

        a_steady = acchistory[0] # Taking the first value of the accumulation vector for steady state profile calculation
        Ts_surf_steady = temphistory[0] # Taking the first value of the temperature vector for steady state profile calculation
        Ts = -z*0 + Ts_surf_steady # Creating the initial temperature vector prior to the steady state profle calculation

        #### Defining time before ungrounding/grounding events

        t_before_ungrounding = int(t_forcing_years) - int(t_ungrounding) - int(t_grounding) # Defining the time before the ungrounding
        t_before_grounding =  int(t_forcing_years) - int(t_grounding)  # Defining the time before the grounding

        ##################################################################################################################

        # Start of loop iterations

        ### Grounded ice steady state T profile with constant ice thickness:

        for i in range(0, int(t_steady_years_total)):

            Hi = thkhistory[0]  # Taking the first value of thickness history for steady state calculation
            zi = np.linspace(0, Hi, Lzi)  #  Space calculation domain for ice of initial thickness
            dz = Hi/Lzi  # Space resolution for ice column
            z = np.concatenate((zr[:-1],zi))  #  Merging space domains for ice and bedrock into one single unified space domain


            # Grounded ice vertical velocity profile after Lliboutry (1979)

            dws = a_steady
            wzt = (1 - (((cls.n + 2) / (cls.n + 1)) * (1 - zi / Hi)) + (1 / (cls.n + 1)) * np.power((1 - zi / Hi), (cls.n + 2)))
            ws = -dws * wzt

            # Advection-diffusion temperature calculation above ice-rock interface

            Ts[l:-1] = Ts[l:-1] + cls.dt*(cls.alpha_i*(Ts[l+1:]-2*Ts[l:-1]+Ts[l-1:-2])/dz**2 - np.multiply(ws[1:-1],(Ts[l+1:]-Ts[l-1:-2])/(2*dz)))
            Ts[-1] = Ts_surf_steady # Temperature forcing is constant and equals the first value of the input surface temperature vector

            # Diffusion temperature calculation below ice-rock interface

            Ts[1:l] = Ts[1:l] + cls.dt * cls.alpha_r * (Ts[2:l+1]-2*Ts[1:l]+Ts[0:l-1])/dzr**2
            Ts[0] = Ts[1] + (G/cls.k_r*dzr)


        ### Grounded ice T profile with time-variable forcings:

        for j in range(0, int(t_before_ungrounding)):
            # TODO: This code isn't invoked in the standard tests; create new test cases

            #  Introducing time-variable ice thickness and recalculating ice column mesh grid at each step

            Hi = thkhistory[j]
            zi = np.linspace(0, Hi, Lzi)
            dz = Hi/Lzi
            z = np.concatenate((zr[:-1],zi))

            #  Interpolating temperature profile at a previous time step on to the new(current) ice column grid

            zint = np.linspace(0, thkhistory[j-1], Lzi)
            s = InterpolatedUnivariateSpline(zint, Ts[l-1:], k=1)

            Ts[l-1:] = s(zi)


            # Grounded ice vertical velocity profile after Lliboutry (1979)

            dws = acchistory[j] - ((thkhistory[j] - thkhistory[j - 1]) / cls.dt); # When thickness is variable through time
            wzt = (1 - (((cls.n + 2) / (cls.n + 1)) * (1 - zi / Hi)) + (1 / (cls.n + 1)) * np.power((1 - zi / Hi), (cls.n + 2)))
            ws = -dws * wzt

            # Advection-diffusion temperature calculation above ice-rock interface

            Ts[l:-1] = Ts[l:-1] + cls.dt*(cls.alpha_i*(Ts[l+1:]-2*Ts[l:-1]+Ts[l-1:-2])/dz**2 - np.multiply(ws[1:-1],(Ts[l+1:]-Ts[l-1:-2])/(2*dz)))
            Ts[-1] = temphistory[j] # Temperature forcing equals value of the imported surface temperature vector at each point in time

            # Diffusion temperature calculation below ice-rock interface

            Ts[1:l] = Ts[1:l] + cls.dt * cls.alpha_r * (Ts[2:l+1]-2*Ts[1:l]+Ts[0:l-1])/dzr**2
            Ts[0] = Ts[1] + (G/cls.k_r*dzr)


            # Recording the results into the temperature and depth matrixes

            Tvar_H_matrix[:,j] = Ts
            zvar_H_matrix[:,j] = z


        ### Introduce ungrounding event:

        for jj in range(0, int(t_ungrounding)):


            #  Recalculating ice column space domain based on thickness at each step

            Hi = thkhistory[jj+t_before_ungrounding]
            zi = np.linspace(0, Hi, Lzi)
            dz = Hi/Lzi
            z = np.concatenate((zr[:-1],zi))

            #  Interpolating temperature profile at a previous time step on to the new(current) ice column grid

            zint = np.linspace(0, thkhistory[jj+t_before_ungrounding-1], Lzi)
            s = InterpolatedUnivariateSpline(zint, Ts[l-1:], k=1)
            Ts[l-1:] = s(zi)


            # Ice shelf vertical velocity profile (linear function)

            w = -np.linspace(acchistory[jj+t_before_ungrounding],acchistory[jj+t_before_ungrounding],Lzi) # In this case, vertical velocity is constant, and accumulation rate and basal melt/freeze rate are in balance, following Jenkins and Holland (1999). It could be modified to the one where velocity varies linearly from the surface to the base of the ice shelf.

            # Advection-diffusion temperature calculation above ice-rock interface

            Ts[l:-1] = Ts[l:-1] + cls.dt*(cls.alpha_i*(Ts[l+1:]-2*Ts[l:-1]+Ts[l-1:-2])/dz**2 - np.multiply(w[1:-1],(Ts[l+1:]-Ts[l-1:-2])/(2*dz)))
            Ts[-1] = temphistory[jj+t_before_ungrounding] # Temperature forcing is constant and equals the first value of the imported Monte-Carlo surface temperature vectors

            Ts[l-1] = -1.89 - 7.53e-4*Hi # Empirically-derived, thickness-dependent seawater freezing point

            # Diffusion temperature calculation below ice-rock interface

            Ts[1:l-1] = Ts[1:l-1] + cls.dt*cls.alpha_r*(Ts[2:l]-2*Ts[1:l-1]+Ts[0:l-2])/dzr**2
            Ts[0] = Ts[1] + (G/cls.k_r*dzr)


            # Recording the results into the temperature and depth matrixes

            Tvar_H_matrix[:,jj+t_before_ungrounding] = Ts
            zvar_H_matrix[:,jj+t_before_ungrounding] = z


        ### Introduce grounding event:

        for jjj in range(0, int(t_grounding)):

            Hi = thkhistory[jjj+t_before_grounding]
            zi = np.linspace(0, Hi, Lzi)
            dz = Hi/Lzi
            z = np.concatenate((zr[:-1],zi))

            #  Interpolating previous temperature profile on to the new ice column space domain

            zint = np.linspace(0, thkhistory[jjj+t_before_grounding-1], Lzi)
            s = InterpolatedUnivariateSpline(zint, Ts[l-1:], k=1)
            Ts[l-1:] = s(zi)


            # Grounded ice vertical velocity profile

            dws = acchistory[jjj+t_before_grounding] - ((thkhistory[jjj+t_before_grounding] - thkhistory[jjj+t_before_grounding - 1]) / cls.dt) # When thickness is variable through time
            wzt = (1 - (((cls.n + 2) / (cls.n + 1)) * (1 - zi / Hi)) + (1 / (cls.n + 1)) * np.power((1 - zi / Hi), (cls.n + 2)))
            ws = -dws * wzt

            # Advection-diffusion temperature calculation above ice-rock interface

            Ts[l:-1] = Ts[l:-1] + cls.dt*(cls.alpha_i*(Ts[l+1:]-2*Ts[l:-1]+Ts[l-1:-2])/dz**2 - np.multiply(ws[1:-1],(Ts[l+1:]-Ts[l-1:-2])/(2*dz)))
            Ts[-1] = temphistory[jjj+t_before_grounding] # Temperature forcing is constant and equals the first value of the imported Monte-Carlo surface temperature vectors

            # Diffusion temperature calculation below ice-rock interface

            Ts[1:l] = Ts[1:l] + cls.dt * cls.alpha_r * (Ts[2:l+1]-2*Ts[1:l]+Ts[0:l-1])/dzr**2
            Ts[0] = Ts[1] + (G/cls.k_r*dzr)


            # Recording the results into the temperature and depth matrixes

            Tvar_H_matrix[:,jjj+t_before_grounding] = Ts
            zvar_H_matrix[:,jjj+t_before_grounding] = z


            # Recording all of the results into the temperature-depth profile tuple that contains all calculations at each step of the simulation

        return cls(Ts, z, Tvar_H_matrix, zvar_H_matrix)

    # ----------------------------------------------------------------------
    @classmethod
    def Load(
        cls,
        filename: Path,
    ) -> "TemperatureModel":
        if not filename.is_file():
            raise ValueError("The file '{}' does not exist.".format(filename))

        with filename.open() as f:
            content = json.load(f)

        # ----------------------------------------------------------------------
        def GetValue(
            attribute_name: str,
        ) -> Any:
            value = content.get(attribute_name, None)
            if value is None:
                raise Exception(
                    "The content in '{}' appears to be corrupt; '{}' was not found.".format(
                        filename,
                        attribute_name,
                    ),
                )

            return value

        # ----------------------------------------------------------------------
        def ConvertArray(
            array: str,
        ) -> np.ndarray:
            source = BytesIO(base64.b64decode(array.encode("ascii")))
            return np.load(source,  allow_pickle=False)

        # ----------------------------------------------------------------------

        version = GetValue("version")

        if version == "0.1.0":
            return TemperatureModel(
                ConvertArray(GetValue("Tmeasured")),
                ConvertArray(GetValue("z")),
                ConvertArray(GetValue("Tvar_H_Tmatrix")),
                ConvertArray(GetValue("zvar_H_Tmatrix")),
            )

        raise Exception("'{}' is not a supported version.".format(version))

    # ----------------------------------------------------------------------
    def __post_init__(self):
        for attribute_name in [
            "Tmeasured",
            "z",
            "Tvar_H_Tmatrix",
            "zvar_H_Tmatrix",
        ]:
            for coordinate, value in np.ndenumerate(getattr(self, attribute_name)):
                if np.isnan(value):
                    raise ValueError("A NaN was encountered in '{}' at {}.".format(attribute_name, coordinate))

    # ----------------------------------------------------------------------
    def Save(
        self,
        filename: Path,
    ) -> None:
        filename.parent.mkdir(parents=True, exist_ok=True)

        with filename.open("w") as f:
            # ----------------------------------------------------------------------
            def ConvertArray(
                array: np.ndarray,
            ) -> str:
                sink = BytesIO()

                np.save(sink, array, allow_pickle=False)
                return base64.b64encode(sink.getvalue()).decode("ascii")

            # ----------------------------------------------------------------------

            json.dump(
                {
                    "version": "0.1.0",
                    "Tmeasured": ConvertArray(self.Tmeasured),
                    "z": ConvertArray(self.z),
                    "Tvar_H_Tmatrix": ConvertArray(self.Tvar_H_Tmatrix),
                    "zvar_H_Tmatrix": ConvertArray(self.zvar_H_Tmatrix),
                },
                f,
            )

    # ----------------------------------------------------------------------
    def __eq__(self, other) -> bool:
        return isinstance(other, TemperatureModel) and self.__class__.Compare(self, other)

    # ----------------------------------------------------------------------
    def __ne__(self, other) -> bool:
        return not isinstance(other, TemperatureModel) or not self.__class__.Compare(self, other)

    # ----------------------------------------------------------------------
    @staticmethod
    def Compare(
        this: "TemperatureModel",
        that: "TemperatureModel",
        *,
        allow_tolerance: bool=False,
    ) -> bool:
        equality_func: Optional[Callable[[np.ndarray, np.ndarray], bool]] = None

        if allow_tolerance:
            equality_func = lambda a, b: np.array_equal(a, b, equal_nan=True)
        else:
            equality_func = lambda a, b: np.allclose(a, b)

        assert equality_func is not None

        for attribute_name in [
            "Tmeasured",
            "z",
            "Tvar_H_Tmatrix",
            "zvar_H_Tmatrix",
        ]:
            if not equality_func(getattr(this, attribute_name), getattr(that, attribute_name)):
                return False

        return True

# TODO: RMS
# TODO: rsmpl
# TODO: model_syntentic
# TODO: model_measured
# TODO: lnlike
# TODO: lnprior
# TODO: lnprob
