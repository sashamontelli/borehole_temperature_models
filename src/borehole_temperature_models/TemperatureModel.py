import base64
import json

from dataclasses import dataclass, InitVar, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import numpy.typing as npt

from scipy.interpolate import InterpolatedUnivariateSpline, interp1d

from borehole_temperature_models import Constants


# ----------------------------------------------------------------------
# |
# |  Public Types
# |
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class TemperatureModel(object):
    # ----------------------------------------------------------------------
    # |  Data
    Tmeasured: npt.NDArray[np.float64]
    Tvar_H_Tmatrix: npt.NDArray[np.float64]
    zvar_H_Tmatrix: npt.NDArray[np.float64]

    # ----------------------------------------------------------------------
    # |  Methods
    @classmethod
    def Create(
        cls,
        dt_years: float,
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
        # TODO (dave.brownell): Assert valid parameters

        Hm = min(Hi_sim)

        tmnew = np.linspace(1, t_forcing_years, num = int(t_forcing_years), endpoint=True) # Creating a uniform time vector with 1-year spacing

        calculator = _Calculator(
            dt_years * Constants.yr_to_s,
            dz,
            G,
            int(t_steady_years_total),
            int(t_forcing_years),
            int(t_ungrounding),
            int(t_grounding),
            zi=np.linspace(0, Hm, int((Hm) / dz) + 1),
            zr=np.linspace(-Hr, 0, int(Hr / dz) + 1),
            temphistory=(
                # Interpolating surface temperature vector over the new time vector
                interp1d(
                    np.linspace(1, int(t_forcing_years), len(T_sim)),  # Creating an evenly spaced time vector for input surface temperature vector,
                    T_sim,
                )(tmnew)
            ),
            thkhistory=(
                # Interpolating thickness vector over the new time vector
                interp1d(
                    np.linspace(1, int(t_forcing_years), len(Hi_sim)), # Creating an evenly spaced time vector for input thickness vector,
                    Hi_sim,
                )(tmnew)
            ),
            acchistory=(
                # Interpolating accumulation vector over the new time vector
                interp1d(
                    np.linspace(1, int(t_forcing_years), len(a_sim)), # Creating an evenly spaced time vector for input accumulation vector,
                    a_sim,
                )(tmnew) / Constants.yr_to_s
            ),
        )

        calculator.Loop1()
        calculator.Loop2()
        calculator.Loop3()
        calculator.Loop4()

        return calculator.CreateTemperatureModel()

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
        ) -> npt.NDArray[np.float64]:
            source = BytesIO(base64.b64decode(array.encode("ascii")))
            return np.load(source,  allow_pickle=False)

        # ----------------------------------------------------------------------

        version = GetValue("version")

        if version == "0.1.0":
            return TemperatureModel(
                ConvertArray(GetValue("Tmeasured")),
                ConvertArray(GetValue("Tvar_H_Tmatrix")),
                ConvertArray(GetValue("zvar_H_Tmatrix")),
            )

        raise Exception("'{}' is not a supported version.".format(version))

    # ----------------------------------------------------------------------
    def __post_init__(self):
        for attribute_name in [
            "Tmeasured",
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
                array: npt.NDArray[np.float64],
            ) -> str:
                sink = BytesIO()

                np.save(sink, array, allow_pickle=False)
                return base64.b64encode(sink.getvalue()).decode("ascii")

            # ----------------------------------------------------------------------

            json.dump(
                {
                    "version": "0.1.0",
                    "Tmeasured": ConvertArray(self.Tmeasured),
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
        equality_func: Optional[Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], bool]] = None

        if allow_tolerance:
            equality_func = lambda a, b: np.array_equal(a, b, equal_nan=True)
        else:
            equality_func = lambda a, b: np.allclose(a, b)

        assert equality_func is not None

        for attribute_name in [
            "Tmeasured",
            "Tvar_H_Tmatrix",
            "zvar_H_Tmatrix",
        ]:
            if not equality_func(getattr(this, attribute_name), getattr(that, attribute_name)):
                return False

        return True


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def Lliboutry(
    zi: npt.NDArray[np.float64],
    Hi: float,
) -> Any:
    return (1 - (((Constants.n + 2) / (Constants.n + 1)) * (1 - zi / Hi)) + (1 / (Constants.n + 1)) * np.power((1 - zi / Hi), (Constants.n + 2)))


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
@dataclass
class _Calculator(object):
    # ----------------------------------------------------------------------
    # |  Data
    dt: float

    dzr: int
    G: float

    t_steady_years_total: int
    t_forcing_years: InitVar[int]
    t_ungrounding: int
    t_grounding: int

    zi: npt.NDArray[np.float64]
    zr: npt.NDArray[np.float64]

    temphistory: npt.NDArray[np.float64]
    thkhistory: npt.NDArray[np.float64]
    acchistory: npt.NDArray[np.float64]

    _len_zi: int                            = field(init=False)
    _len_zr: int                            = field(init=False)

    _t_before_ungrounding: int              = field(init=False)
    _t_before_grounding: int                = field(init=False)

    _Ts: npt.NDArray[np.float64]            = field(init=False)

    _Tvar_H_matrix: npt.NDArray[Any]        = field(init=False)
    _zvar_H_matrix: npt.NDArray[Any]        = field(init=False)

    # ----------------------------------------------------------------------
    # |  Methods
    def __post_init__(
        self,
        t_forcing_years: int,
    ):
        len_zi = len(self.zi)
        len_zr = len(self.zr)

        t_before_ungrounding = t_forcing_years - self.t_ungrounding - self.t_grounding
        t_before_grounding =  t_forcing_years - self.t_grounding  # Defining the time before the grounding

        Ts = np.full(len_zi + len_zr - 1, self.temphistory[0])

        Tvar_H_matrix = np.zeros((len(Ts), t_forcing_years)) # Preparing temperature matrix for each step of temperature calculations
        zvar_H_matrix = np.zeros((len(Ts), t_forcing_years)) # Preparing depth matrix for each step of temperature calculations

        # Commit values
        self._len_zi = len_zi
        self._len_zr = len_zr

        self._t_before_ungrounding = t_before_ungrounding
        self._t_before_grounding = t_before_grounding

        self._Ts = Ts
        self._Tvar_H_matrix = Tvar_H_matrix
        self._zvar_H_matrix = zvar_H_matrix

    # ----------------------------------------------------------------------
    def CreateTemperatureModel(self) -> TemperatureModel:
        return TemperatureModel(
            self._Ts,
            self._Tvar_H_matrix,
            self._zvar_H_matrix,
        )

    # ----------------------------------------------------------------------
    def Loop1(self) -> None:
        dz = self.thkhistory[0] / self._len_zi
        ws = (
            -self.acchistory[0]
            * Lliboutry(
                np.linspace(0, self.thkhistory[0], self._len_zi),
                self.thkhistory[0],
            )
        )

        # ----------------------------------------------------------------------
        def PreLoopCalc(
            loop_index: int,
        ) -> tuple[
            float,
            npt.NDArray[np.float64],
            Optional[npt.NDArray[np.float64]],
        ]:
            return dz, ws, None

        # ----------------------------------------------------------------------
        def IntraLoopCallback(
            loop_index: int,
        ) -> None:
            self._Ts[-1] = self.temphistory[0]

        # ----------------------------------------------------------------------

        self._LoopImpl(
            self.t_steady_years_total,
            PreLoopCalc,
            IntraLoopCallback,
            update_matrix_index_offset_or_post_loop_callback=None,
        )

    # ----------------------------------------------------------------------
    def Loop2(self) -> None:
        # ----------------------------------------------------------------------
        def PreLoopCalc(
            loop_index: int,
        ) -> tuple[
            float,
            npt.NDArray[np.float64],
            Optional[npt.NDArray[np.float64]],
        ]:
            Hi = self.thkhistory[loop_index]

            new_zi = np.linspace(0, Hi, self._len_zi)
            z = np.concatenate((self.zr[:-1], new_zi))

            #  Interpolating temperature profile at a previous time step on to the new(current) ice column grid
            self._Ts[self._len_zr - 1:] = InterpolatedUnivariateSpline(
                np.linspace(0, self.thkhistory[loop_index - 1], self._len_zi),
                self._Ts[self._len_zr - 1:],
                k=1,
            )(new_zi)

            # Grounded ice vertical velocity profile after Lliboutry (1979)

            dws = self.acchistory[loop_index] - ((self.thkhistory[loop_index] - self.thkhistory[loop_index - 1]) / self.dt) # When thickness is variable through time

            return (
                Hi / self._len_zi,  # type: ignore
                # When thickness is variable through time
                -dws * Lliboutry(new_zi, Hi),  # type: ignore
                z,
            )

        # ----------------------------------------------------------------------
        def IntraLoopCallback(
            loop_index: int,
        ) -> None:
            # Temperature forcing equals value of the imported surface temperature vector at each point in time
            self._Ts[-1] = self.temphistory[loop_index]

        # ----------------------------------------------------------------------
        def PostLoopCallback(
            loop_index: int,
            z: npt.NDArray[np.float64],
        ) -> None:
            self._Tvar_H_matrix[:, loop_index] = self._Ts
            self._zvar_H_matrix[:, loop_index] = z

        # ----------------------------------------------------------------------

        self._LoopImpl(
            self._t_before_ungrounding,
            PreLoopCalc,
            IntraLoopCallback,
            update_matrix_index_offset_or_post_loop_callback=PostLoopCallback,
        )

    # ----------------------------------------------------------------------
    def Loop3(self) -> None:
        Hi: Optional[float] = None

        # ----------------------------------------------------------------------
        def PreLoopCalc(
            loop_index: int,
        ) -> tuple[
            float,
            npt.NDArray[np.float64],
            Optional[npt.NDArray[np.float64]],
        ]:
            nonlocal Hi

            Hi = self.thkhistory[loop_index + self._t_before_ungrounding]

            new_zi = np.linspace(0, Hi, self._len_zi)  # type: ignore
            z = np.concatenate((self.zr[:-1], new_zi))

            #  Interpolating temperature profile at a previous time step on to the new(current) ice column grid
            self._Ts[self._len_zr - 1:] = InterpolatedUnivariateSpline(
                np.linspace(0, self.thkhistory[loop_index + self._t_before_ungrounding - 1], self._len_zi),
                self._Ts[self._len_zr - 1:],
                k=1,
            )(new_zi)

            # Ice shelf vertical velocity profile (linear function)

            return (
                Hi / self._len_zi,  # type: ignore
                # In this case, vertical velocity is constant, and accumulation rate and basal melt/freeze rate are in balance, following Jenkins and Holland (1999). It could be modified to the one where velocity varies linearly from the surface to the base of the ice shelf.
                -np.linspace(
                    self.acchistory[loop_index + self._t_before_ungrounding],
                    self.acchistory[loop_index + self._t_before_ungrounding],
                    self._len_zi,
                ),
                z,
            )

        # ----------------------------------------------------------------------
        def IntraLoopCallback(
            loop_index: int,
        ) -> None:
            self._Ts[-1] = self.temphistory[loop_index + self._t_before_ungrounding] # Temperature forcing is constant and equals the first value of the imported Monte-Carlo surface temperature vectors

            self._Ts[self._len_zr - 1] = -1.89 - 7.53e-4 * Hi #  type: ignore # Empirically-derived, thickness-dependent seawater freezing point

        # ----------------------------------------------------------------------

        self._LoopImpl(
            self.t_ungrounding,
            PreLoopCalc,
            IntraLoopCallback,
            calculation_offset=1,
            update_matrix_index_offset_or_post_loop_callback=self._t_before_ungrounding,
        )

    # ----------------------------------------------------------------------
    def Loop4(self) -> None:
        # ----------------------------------------------------------------------
        def PreLoopCalc(
            loop_index: int,
        ) -> tuple[
            float,
            npt.NDArray[np.float64],
            Optional[npt.NDArray[np.float64]],
        ]:
            Hi = self.thkhistory[loop_index + self._t_before_grounding]

            new_zi = np.linspace(0, Hi, self._len_zi)
            z = np.concatenate((self.zr[:-1], new_zi))

            #  Interpolating previous temperature profile on to the new ice column space domain
            self._Ts[self._len_zr - 1:] = InterpolatedUnivariateSpline(
                np.linspace(0, self.thkhistory[loop_index + self._t_before_grounding - 1], self._len_zi),
                self._Ts[self._len_zr - 1:],
                k=1,
            )(new_zi)

            # Grounded ice vertical velocity profile

            dws = self.acchistory[loop_index + self._t_before_grounding] - ((self.thkhistory[loop_index + self._t_before_grounding] - self.thkhistory[loop_index + self._t_before_grounding - 1]) / self.dt) # When thickness is variable through time

            return (
                Hi / self._len_zi,
                -dws * Lliboutry(new_zi, Hi),
                z,
            )

        # ----------------------------------------------------------------------
        def IntraLoopCallback(
            loop_index: int,
        ) -> None:
            self._Ts[-1] = self.temphistory[loop_index + self._t_before_grounding] # Temperature forcing is constant and equals the first value of the imported Monte-Carlo surface temperature vectors

        # ----------------------------------------------------------------------

        self._LoopImpl(
            self.t_grounding,
            PreLoopCalc,
            IntraLoopCallback,
            update_matrix_index_offset_or_post_loop_callback=self._t_before_grounding,
        )

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    def _LoopImpl(
        self,
        num_iterations: int,
        pre_loop_calc_func: Callable[
            [int],
            tuple[
                float,                                  # dz
                npt.NDArray[np.float64],                # ws
                Optional[npt.NDArray[np.float64]],      # z
            ],
        ],
        intra_loop_callback_func: Callable[[int], None],
        *,
        calculation_offset: int=0,
        update_matrix_index_offset_or_post_loop_callback: Union[
            None,
            int,
            Callable[
                [
                    int,                                # loop_index
                    npt.NDArray[np.float64],            # z
                ],
                None,
            ],
        ],
    ) -> None:
        if num_iterations == 0:
            return

        if callable(update_matrix_index_offset_or_post_loop_callback):
            post_loop_callback_func = update_matrix_index_offset_or_post_loop_callback
        elif isinstance(update_matrix_index_offset_or_post_loop_callback, int):
            update_matrix_index_offset = update_matrix_index_offset_or_post_loop_callback

            # ----------------------------------------------------------------------
            def UpdateMatrixValues(
                loop_index: int,
                z: npt.NDArray[np.float64],
            ) -> None:
                self._Tvar_H_matrix[:, loop_index + update_matrix_index_offset] = self._Ts
                self._zvar_H_matrix[:, loop_index + update_matrix_index_offset] = z

            # ----------------------------------------------------------------------

            post_loop_callback_func = UpdateMatrixValues
        else:
            post_loop_callback_func = lambda *args, **kwargs: None

        # Example Loop views given:
        #
        #   `len_zr` == 4
        #   `calculation_offset` == 0
        #
        #                 len_zr
        #                   |
        #                   V
        # | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
        #                   -----------------: view1
        #                       -----------------: view2
        #               -----------------: view3
        #       ---------: view4
        #           ---------: view5
        #   ---------: view6

        view1 = self._Ts[self._len_zr:-1]
        view2 = self._Ts[self._len_zr + 1:]
        view3 = self._Ts[self._len_zr - 1:-2]
        view4 = self._Ts[1:self._len_zr - calculation_offset]
        view5 = self._Ts[2:self._len_zr + 1 - calculation_offset]
        view6 = self._Ts[:self._len_zr - 1 - calculation_offset]

        drz_squared = self.dzr ** 2

        post_loop_delta_value = (self.G / Constants.k_r * self.dzr)

        z: Optional[npt.NDArray[np.float64]] = None

        for loop_index in range(num_iterations):
            dz, ws, z = pre_loop_calc_func(loop_index)

            view1 += (
                self.dt
                * (
                    Constants.alpha_i
                    * (view2 - 2 * view1 + view3)
                    / dz ** 2
                    - np.multiply(
                        ws[1:-1],
                        (view2 - view3) / (dz * 2),
                    )
                )
            )

            # Perform loop-specific updates in the calling function
            intra_loop_callback_func(loop_index)

            view4 += (
                self.dt
                * Constants.alpha_r
                * (view5 - 2 * view4 + view6)
                / drz_squared
            )

            self._Ts[0] = self._Ts[1] + post_loop_delta_value

            post_loop_callback_func(loop_index, z)  # type: ignore
