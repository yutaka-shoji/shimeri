import warnings
from typing import Callable

import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy.optimize import root


# CONSTANTS
MOL_WEIGHT_WATER = 18.0153  # g/mol
MOL_WEIGHT_AIR = 28.9645  # g/mol


class ConvergenceError(Exception):
    """Raised when the system of equations cannot be solved."""

    pass


class PsychrometricCalculator:
    """Class to calculate psychrometric variables."""

    def __init__(self, pressure: float = 101.325):
        """Initialize the Psychrometrics class.

        Args:
            pressure: Atmospheric pressure (kPa)
        """
        self.pressure = pressure

    def get_all(
        self,
        db: ArrayLike = np.nan,
        wb: ArrayLike = np.nan,
        rh: ArrayLike = np.nan,
        hr: ArrayLike = np.nan,
        en: ArrayLike = np.nan,
    ) -> tuple[
        NDArray[np.float64] | float,
        NDArray[np.float64] | float,
        NDArray[np.float64] | float,
        NDArray[np.float64] | float,
        NDArray[np.float64] | float,
    ]:
        """
        Calculate all psychrometric variables given any two of them.

        This function takes in any two of the five psychrometric variables (dry bulb temperature, wet bulb temperature,
        relative humidity, humidty ratio, specific air enthalpy) and calculates the remaining three. The inputs are
        broadcasted to have the same shape, and the calculation is performed element-wise.

        Args:
            db: Dry bulb temperature (degC).
            wb: Wet bulb temperature (degC).
            rh: Relative humidity (%).
            hr: Humidty ratio in g/kg.
            en: Specific air enthalpy in kJ/kg.

        Returns:
            A tuple of five numpy arrays or float, each representing one of the psychrometric variables. The arrays have the same shape as the broadcasted input arrays.

        Raises:
            ValueError: If the number of provided variables is not exactly two.
            ConvergenceError: If the calculation does not converge for a particular set of inputs.
        """
        # Check args number (2 vars of 5)
        provided_args_number = sum(
            np.isfinite(arg).any() for arg in [db, wb, rh, hr, en]
        )
        if provided_args_number != 2:
            raise ValueError("Input 2 Variables of 5.")

        db, wb, rh, hr, en = np.broadcast_arrays(db, wb, rh, hr, en)
        db = db.flatten()
        wb = wb.flatten()
        rh = rh.flatten()
        hr = hr.flatten()
        en = en.flatten()

        for idx in range(db.size):
            try:
                db[idx], wb[idx], rh[idx], hr[idx], en[idx] = self._calc_single(
                    db[idx], wb[idx], rh[idx], hr[idx], en[idx]
                )
            except (ValueError, ConvergenceError) as e:
                warnings.warn(
                    "Calculation failed for input at "
                    + "(db,wb,rh,hr,en)="
                    + f"({db[idx]:.1f},{wb[idx]:.1f},{rh[idx]:.1f},{hr[idx]:.1f},{en[idx]:.1f}): "
                    + f"{str(e)}"
                )
                db[idx] = np.nan
                wb[idx] = np.nan
                rh[idx] = np.nan
                hr[idx] = np.nan
                en[idx] = np.nan

        if db.size == 1:
            return db[0], wb[0], rh[0], hr[0], en[0]
        return db, wb, rh, hr, en

    def _calc_single(
        self,
        db: float,
        wb: float,
        rh: float,
        hr: float,
        en: float,
        ps: float = np.nan,
        ps_wb: float = np.nan,
        pw: float = np.nan,
    ) -> tuple[float, float, float, float, float]:
        """
        Calculate psychrometric variables for a single set of inputs.

        Args:
            db: Dry bulb temperature (degC).
            wb: Wet bulb temperature (degC).
            rh: Relative humidity (%).
            hr: Humidity ratio (g/kg).
            en: Specific air enthalpy (kJ/kg).
            ps: Saturation pressure at dry bulb temperature (kPa).
            ps_wb: Saturation pressure at wet bulb temperature (kPa).
            pw: Partial pressure of water vapor (kPa).

        Returns:
            A tuple of calculated psychrometric variables: dry bulb temperature, wet bulb temperature,
            relative humidity, humidity ratio, and specific air enthalpy.

        Raises:
            ConvergenceError: If the root finding algorithm fails to converge.
            ValueError: If the calculated relative humidity is greater than 100% or the calculated
            humidity ratio is less than 0 g/kg.
        """
        input_arr = np.array([ps, ps_wb, pw, db, wb, rh, hr, en])
        input_idxs = np.where(np.isfinite(input_arr))[0]
        input_vals = input_arr[input_idxs]

        # system of equations
        f = self._make_eqs_function(input_idxs, input_vals)
        # initial guess
        x_0 = np.array([3.0, 2.0, 2.0, 25.0, 20.0, 50.0, 10.0, 50.0])
        sol = root(f, x_0)

        if not sol.success:
            raise ConvergenceError(f"Convergence failed: {sol.message}")

        result = np.round(sol.x[3:], 2)
        db = result[0]
        wb = result[1]
        rh = result[2]
        hr = result[3]
        en = result[4]

        if rh > 100:
            raise ValueError(f"RH={rh:.1f}>100%")

        if hr < 0:
            raise ValueError(f"HR={hr:.1f}<0g.kg-1")

        return db, wb, rh, hr, en

    def _make_eqs_function(
        self, input_idxs: NDArray[np.int32], input_vals: NDArray[np.float64]
    ) -> Callable:
        def equation(x: NDArray[np.float64]):
            ps = x[0]
            ps_wb = x[1]
            pw = x[2]
            db = x[3]
            wb = x[4]
            rh = x[5]
            hr = x[6]
            en = x[7]

            c = self._psychrometer_constant(wb)  # psychrometer constant

            eqs = np.zeros(8)

            eqs[0] = ps - get_saturation_pressure(db)  # saturation pressure eq
            eqs[1] = ps_wb - get_saturation_pressure(wb)  # wb saturation pressure eq
            eqs[2] = pw - ps_wb + c * self.pressure * (db - wb)  # Sprung eq
            eqs[3] = rh - (100 * pw) / ps  # relative humidity eq
            eqs[4] = hr * 1e-3 - ((MOL_WEIGHT_WATER / MOL_WEIGHT_AIR) * pw) / (
                self.pressure - pw
            )  # humidity ratio eq
            eqs[5] = en - 1.006 * db - (1.86 * db + 2501) * hr * 1e-3  # enthalpy eq
            eqs[6] = x[input_idxs[0]] - input_vals[0]  # input equal
            eqs[7] = x[input_idxs[1]] - input_vals[1]  # input equal

            return eqs

        return equation

    @staticmethod
    def get_en_from_db_hr(db: ArrayLike, hr: ArrayLike) -> NDArray[np.float64]:
        """
        Calculate specific air enthalpy from dry bulb temperature and humidity ratio.

        Args:
            db: Dry bulb temperature (degC).
            hr: Humidity ratio (g/kg).

        Returns:
            Specific air enthalpy (kJ/kg).
        """
        # Broadcast the input arrays to the same shape
        db, hr = np.broadcast_arrays(db, hr)
        # Calculate specific air enthalpy for each pair of db and hr
        return 1.006 * db + (1.86 * db + 2501) * hr * 1e-3

    @staticmethod
    def get_hr_from_db_en(db: ArrayLike, en: ArrayLike) -> NDArray[np.float64]:
        """
        Calculate humidity ratio from dry bulb temperature and specific air enthalpy.

        Args:
            db: Dry bulb temperature (degC).
            en: Specific air enthalpy (kJ/kg).

        Returns:
            Humidity ratio (g/kg).
        """
        # Broadcast the input arrays to the same shape
        db, en = np.broadcast_arrays(db, en)
        # Calculate humidity ratio for each pair of db and en
        return (en - 1.006 * db) / (1.86 * db + 2501) * 1e3

    def get_hr_from_db_rh(self, db: ArrayLike, rh: ArrayLike) -> NDArray[np.float64]:
        """
        Calculate humidity ratio from dry bulb temperature and relative humidity.

        Args:
            db: Dry bulb temperature (degC).
            rh: Relative humidity (%).

        Returns:
            Humidity ratio (g/kg).
        """
        # Broadcast the input arrays to the same shape
        db, rh = np.broadcast_arrays(db, rh)
        # Calculate saturation pressure
        ps = get_saturation_pressure(db)
        # Calculate partial pressure of water vapor
        pw = rh * ps / 100
        # Calculate humidity ratio
        return (MOL_WEIGHT_WATER / MOL_WEIGHT_AIR) * pw / (self.pressure - pw) * 1e3

    def get_rh_from_db_hr(self, db: ArrayLike, hr: ArrayLike) -> NDArray[np.float64]:
        """
        Calculate relative humidity from dry bulb temperature and humidity ratio.

        Args:
            db: Dry bulb temperature (degC).
            hr: Humidity ratio (g/kg).

        Returns:
            Relative humidity (%).
        """
        # Broadcast the input arrays to the same shape
        db, hr = np.broadcast_arrays(db, hr)
        # Calculate saturation pressure
        ps = get_saturation_pressure(db)
        # Calculate partial pressure of water vapor
        pw = (
            self.pressure
            * hr
            * 1e-3
            / ((MOL_WEIGHT_WATER / MOL_WEIGHT_AIR) + hr * 1e-3)
        )
        # Calculate relative humidity
        return 100 * pw / ps

    @staticmethod
    def get_db_from_hr_en(hr: ArrayLike, en: ArrayLike) -> NDArray[np.float64]:
        """
        Calculate dry bulb temperature from humidity ratio and specific air enthalpy.

        Args:
            hr: Humidity ratio (g/kg).
            en: Specific air enthalpy (kJ/kg).

        Returns:
            Dry bulb temperature (degC).
        """
        # Broadcast the input arrays to the same shape
        hr, en = np.broadcast_arrays(hr, en)
        # Calculate dry bulb temperature
        return (en - 2501 * hr * 1e-3) / (1.006 + 1.86 * hr * 1e-3)

    @staticmethod
    def _psychrometer_constant(wb: float) -> float:
        """Calculate the psychrometer constant."""
        return 0.000662 if wb >= 0.01 else 0.000583


def get_saturation_pressure(temp: ArrayLike) -> NDArray[np.float64]:
    """
    Calculate saturation pressure (Hyland and Wexler, 1983).

    Args:
        temp: Temperature (degC).

    Returns:
        Saturation pressure (kPa)
    """
    temp_in_kelvine = np.asarray(temp) + 273.15
    result = np.empty_like(temp_in_kelvine)

    # Calculate for temperatures above 0.01
    mask = temp_in_kelvine > 0.01
    result[mask] = (
        1e-3
        * np.exp(
            -(0.58002206 * 1e4) / temp_in_kelvine[mask]
            + (0.13914993 * 1e1)
            - (0.48640239 * 1e-1) * temp_in_kelvine[mask]
            + (0.41764768 * 1e-4) * temp_in_kelvine[mask] ** 2
            - (0.14452093 * 1e-7) * temp_in_kelvine[mask] ** 3
        )
        * temp_in_kelvine[mask] ** 6.5459673
    )

    # Calculate for temperatures below or equal to 0.01
    mask = ~mask
    result[mask] = (
        1e-3
        * np.exp(
            -(0.56745359 * 1e4) / temp_in_kelvine[mask]
            + (0.63925247 * 1e1)
            - (0.96778430 * 1e-2) * temp_in_kelvine[mask]
            + (0.62215701 * 1e-6) * temp_in_kelvine[mask] ** 2
            + (0.20747825 * 1e-8) * temp_in_kelvine[mask] ** 3
            - (0.94840240 * 1e-12) * temp_in_kelvine[mask] ** 4
        )
        * temp_in_kelvine[mask] ** 4.1635019
    )

    return result
