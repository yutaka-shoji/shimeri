from typing import Union

import numpy as np
from CoolProp.HumidAirProp import HAPropsSI
from numpy.typing import ArrayLike, NDArray


class PsychrometricCalculator:
    """Class to calculate psychrometric variables using CoolProp."""

    def __init__(self, pressure: float = 101.325):
        """Initialize the Psychrometrics class.

        Args:
            pressure: Atmospheric pressure (kPa)
        """
        self.pressure = pressure
        self._pressure_pa = pressure * 1000  # Convert to Pa for CoolProp

    def get_all(
        self,
        db: ArrayLike = np.nan,
        wb: ArrayLike = np.nan,
        rh: ArrayLike = np.nan,
        hr: ArrayLike = np.nan,
        en: ArrayLike = np.nan,
    ) -> tuple[
        Union[NDArray[np.float64], float],
        Union[NDArray[np.float64], float],
        Union[NDArray[np.float64], float],
        Union[NDArray[np.float64], float],
        Union[NDArray[np.float64], float],
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
        """
        # Check args number (2 vars of 5)
        provided_args_number = sum(
            np.isfinite(arg).any() for arg in [db, wb, rh, hr, en]
        )
        if provided_args_number != 2:
            raise ValueError("Input 2 Variables of 5.")

        db, wb, rh, hr, en = np.broadcast_arrays(db, wb, rh, hr, en)
        original_shape = db.shape
        db = db.flatten().astype(float)
        wb = wb.flatten().astype(float)
        rh = rh.flatten().astype(float)
        hr = hr.flatten().astype(float)
        en = en.flatten().astype(float)

        # Build input parameters for HAPropsSI
        inputs = []

        if np.isfinite(db).any():
            inputs.append(("T", db + 273.15))  # degC to K
        if np.isfinite(wb).any():
            inputs.append(("B", wb + 273.15))  # degC to K (wet bulb)
        if np.isfinite(rh).any():
            inputs.append(("R", rh / 100))  # % to fraction
        if np.isfinite(hr).any():
            inputs.append(("W", hr / 1000))  # g/kg to kg/kg
        if np.isfinite(en).any():
            inputs.append(("H", en * 1000))  # kJ/kg to J/kg

        input1_name, input1_val = inputs[0]
        input2_name, input2_val = inputs[1]

        # Calculate all outputs using HAPropsSI
        db_out = (
            HAPropsSI(
                "T",
                input1_name,
                input1_val,
                input2_name,
                input2_val,
                "P",
                self._pressure_pa,
            )
            - 273.15
        )  # K to degC
        wb_out = (
            HAPropsSI(
                "B",
                input1_name,
                input1_val,
                input2_name,
                input2_val,
                "P",
                self._pressure_pa,
            )
            - 273.15
        )  # K to degC
        rh_out = (
            HAPropsSI(
                "R",
                input1_name,
                input1_val,
                input2_name,
                input2_val,
                "P",
                self._pressure_pa,
            )
            * 100
        )  # fraction to %
        hr_out = (
            HAPropsSI(
                "W",
                input1_name,
                input1_val,
                input2_name,
                input2_val,
                "P",
                self._pressure_pa,
            )
            * 1000
        )  # kg/kg to g/kg
        en_out = (
            HAPropsSI(
                "H",
                input1_name,
                input1_val,
                input2_name,
                input2_val,
                "P",
                self._pressure_pa,
            )
            / 1000
        )  # J/kg to kJ/kg

        # Ensure outputs are numpy arrays
        db_out = np.atleast_1d(db_out)
        wb_out = np.atleast_1d(wb_out)
        rh_out = np.atleast_1d(rh_out)
        hr_out = np.atleast_1d(hr_out)
        en_out = np.atleast_1d(en_out)

        if db_out.size == 1:
            return db_out[0], wb_out[0], rh_out[0], hr_out[0], en_out[0]
        return (
            db_out.reshape(original_shape),
            wb_out.reshape(original_shape),
            rh_out.reshape(original_shape),
            hr_out.reshape(original_shape),
            en_out.reshape(original_shape),
        )

    def get_en_from_db_hr(self, db: ArrayLike, hr: ArrayLike) -> NDArray[np.float64]:
        """
        Calculate specific air enthalpy from dry bulb temperature and humidity ratio.

        Args:
            db: Dry bulb temperature (degC).
            hr: Humidity ratio (g/kg).

        Returns:
            Specific air enthalpy (kJ/kg).
        """
        db, hr = np.broadcast_arrays(db, hr)
        original_shape = db.shape

        result = (
            HAPropsSI(
                "H",
                "T",
                np.asarray(db).flatten() + 273.15,
                "W",
                np.asarray(hr).flatten() / 1000,
                "P",
                self._pressure_pa,
            )
            / 1000
        )  # J/kg to kJ/kg

        return np.atleast_1d(result).reshape(original_shape)

    def get_en_from_db_rh(self, db: ArrayLike, rh: ArrayLike) -> NDArray[np.float64]:
        """
        Calculate specific air enthalpy from dry bulb temperature and relative humidity.

        Args:
            db: Dry bulb temperature (degC).
            rh: Relative humidity (%).
        Returns:
            Specific air enthalpy (kJ/kg).
        """
        db, rh = np.broadcast_arrays(db, rh)
        original_shape = db.shape

        result = (
            HAPropsSI(
                "H",
                "T",
                np.asarray(db).flatten() + 273.15,
                "R",
                np.asarray(rh).flatten() / 100,
                "P",
                self._pressure_pa,
            )
            / 1000
        )  # J/kg to kJ/kg

        return np.atleast_1d(result).reshape(original_shape)

    def get_hr_from_db_en(self, db: ArrayLike, en: ArrayLike) -> NDArray[np.float64]:
        """
        Calculate humidity ratio from dry bulb temperature and specific air enthalpy.

        Args:
            db: Dry bulb temperature (degC).
            en: Specific air enthalpy (kJ/kg).

        Returns:
            Humidity ratio (g/kg).
        """
        db, en = np.broadcast_arrays(db, en)
        original_shape = db.shape

        result = (
            HAPropsSI(
                "W",
                "T",
                np.asarray(db).flatten() + 273.15,
                "H",
                np.asarray(en).flatten() * 1000,
                "P",
                self._pressure_pa,
            )
            * 1000
        )  # kg/kg to g/kg

        return np.atleast_1d(result).reshape(original_shape)

    def get_hr_from_db_rh(self, db: ArrayLike, rh: ArrayLike) -> NDArray[np.float64]:
        """
        Calculate humidity ratio from dry bulb temperature and relative humidity.

        Args:
            db: Dry bulb temperature (degC).
            rh: Relative humidity (%).

        Returns:
            Humidity ratio (g/kg).
        """
        db, rh = np.broadcast_arrays(db, rh)
        original_shape = db.shape

        result = (
            HAPropsSI(
                "W",
                "T",
                np.asarray(db).flatten() + 273.15,
                "R",
                np.asarray(rh).flatten() / 100,
                "P",
                self._pressure_pa,
            )
            * 1000
        )  # kg/kg to g/kg

        return np.atleast_1d(result).reshape(original_shape)

    def get_rh_from_db_hr(self, db: ArrayLike, hr: ArrayLike) -> NDArray[np.float64]:
        """
        Calculate relative humidity from dry bulb temperature and humidity ratio.

        Args:
            db: Dry bulb temperature (degC).
            hr: Humidity ratio (g/kg).

        Returns:
            Relative humidity (%).
        """
        db, hr = np.broadcast_arrays(db, hr)
        original_shape = db.shape

        result = (
            HAPropsSI(
                "R",
                "T",
                np.asarray(db).flatten() + 273.15,
                "W",
                np.asarray(hr).flatten() / 1000,
                "P",
                self._pressure_pa,
            )
            * 100
        )  # fraction to %

        return np.atleast_1d(result).reshape(original_shape)

    def get_db_from_hr_en(self, hr: ArrayLike, en: ArrayLike) -> NDArray[np.float64]:
        """
        Calculate dry bulb temperature from humidity ratio and specific air enthalpy.

        Args:
            hr: Humidity ratio (g/kg).
            en: Specific air enthalpy (kJ/kg).

        Returns:
            Dry bulb temperature (degC).
        """
        hr, en = np.broadcast_arrays(hr, en)
        original_shape = hr.shape

        result = (
            HAPropsSI(
                "T",
                "W",
                np.asarray(hr).flatten() / 1000,
                "H",
                np.asarray(en).flatten() * 1000,
                "P",
                self._pressure_pa,
            )
            - 273.15
        )  # K to degC

        return np.atleast_1d(result).reshape(original_shape)

    def get_db_from_rh_en(self, rh: ArrayLike, en: ArrayLike) -> NDArray[np.float64]:
        """
        Calculate dry bulb temperature from relative humidity and specific air enthalpy.

        Args:
            rh: Relative humidity (%).
            en: Specific air enthalpy (kJ/kg).
        Returns:
            Dry bulb temperature (degC).
        """
        rh, en = np.broadcast_arrays(rh, en)
        original_shape = rh.shape

        result = (
            HAPropsSI(
                "T",
                "R",
                np.asarray(rh).flatten() / 100,
                "H",
                np.asarray(en).flatten() * 1000,
                "P",
                self._pressure_pa,
            )
            - 273.15
        )  # K to degC

        return np.atleast_1d(result).reshape(original_shape)

    def get_hr_from_db_wb(self, db: ArrayLike, wb: ArrayLike) -> NDArray[np.float64]:
        """
        Calculate humidity ratio from dry bulb temperature and wet bulb temperature.

        Args:
            db: Dry bulb temperature (degC).
            wb: Wet bulb temperature (degC).

        Returns:
            Humidity ratio (g/kg).
        """
        db, wb = np.broadcast_arrays(db, wb)
        original_shape = db.shape

        result = (
            HAPropsSI(
                "W",
                "T",
                np.asarray(db).flatten() + 273.15,
                "B",
                np.asarray(wb).flatten() + 273.15,
                "P",
                self._pressure_pa,
            )
            * 1000
        )  # kg/kg to g/kg

        return np.atleast_1d(result).reshape(original_shape)

    def get_hr_from_rh_en(self, rh: ArrayLike, en: ArrayLike) -> NDArray[np.float64]:
        """
        Calculate humidity ratio from relative humidity and specific air enthalpy.

        Args:
            rh: Relative humidity (%).
            en: Specific air enthalpy (kJ/kg).
        Returns:
            Humidity ratio (g/kg).
        """
        rh, en = np.broadcast_arrays(rh, en)
        original_shape = rh.shape

        result = (
            HAPropsSI(
                "W",
                "R",
                np.asarray(rh).flatten() / 100,
                "H",
                np.asarray(en).flatten() * 1000,
                "P",
                self._pressure_pa,
            )
            * 1000
        )  # kg/kg to g/kg

        return np.atleast_1d(result).reshape(original_shape)

    def get_hr_from_pw(self, pw: ArrayLike) -> NDArray[np.float64]:
        """
        Calculate humidity ratio from partial pressure of water vapor.

        Args:
            pw: Partial pressure of water vapor (kPa).

        Returns:
            Humidity ratio (g/kg).
        """
        pw = np.asarray(pw)
        original_shape = pw.shape
        is_scalar = pw.ndim == 0

        result = (
            HAPropsSI(
                "W",
                "P_w",
                np.atleast_1d(pw).flatten() * 1000,  # kPa to Pa
                "T",
                293.15,  # Reference temperature (20°C)
                "P",
                self._pressure_pa,
            )
            * 1000
        )  # kg/kg to g/kg

        if is_scalar:
            return np.atleast_1d(result)[0]
        return np.atleast_1d(result).reshape(original_shape)
