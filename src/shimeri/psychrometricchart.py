from typing import Union

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray

from shimeri.psychrometrics import PsychrometricCalculator


class PsychrometricChart(go.Figure):
    """
    A class to create and manipulate a psychrometric chart.

    The PsychrometricChart class extends the go.Figure class from the plotly.graph_objects module.
    """

    def __init__(self, pressure: float = 101.325):
        """
        Initialize the PsychrometricChart class.

        Args:
            pressure: Atmospheric pressure in kPa.
        """
        super().__init__()

        self._pressure = pressure
        self._pc = PsychrometricCalculator(pressure)
        self._slope = self._calc_skew_slope()

        bg_lines_layout = {
            "mode": "lines",
            "line": {"color": "#E0E0E0", "width": 1},
            "name": "",
            "showlegend": False,
            "hoverinfo": "skip",
        }

        # Draw iso RH lines
        _ = [
            self.draw_iso_rh_line(rh, **bg_lines_layout) for rh in np.arange(0, 101, 10)
        ]
        # Draw iso DB lines
        _ = [
            self.draw_iso_db_line(db, **bg_lines_layout)
            for db in np.arange(-10, 71, 10)
        ]
        # Draw iso EN lines
        _ = [
            self.draw_iso_en_line(en, **bg_lines_layout)
            for en in np.arange(-10, 161, 10)
        ]

        # Add iso RH lines annotations
        self._add_iso_rh_annotation()
        # Add iso EN lines annotations
        self._add_iso_en_annotation()

        # Set xtick labels as the dry-bulb temperature values
        self._xtick_label_to_db()

        self.update_layout(
            template="plotly_white",
        )
        self.update_xaxes(
            title="Dry-Bulb Temperature (degC)",
            range=self._db_to_en_at_hr0([-10, 50]),
            linecolor="black",
            linewidth=1.0,
            mirror=True,
            showgrid=False,
            zeroline=False,
        )
        self.update_yaxes(
            title="Humidity Ratio (g<sub>water</sub>/kg<sub>air</sub>)",
            range=[0, 30],
            linecolor="black",
            linewidth=1.0,
            mirror=True,
            showgrid=True,
            zeroline=False,
            minor=dict(showgrid=True),
        )

    def add_points(
        self,
        en: Union[NDArray[np.float64], float],
        hr: Union[NDArray[np.float64], float],
        **kwargs,
    ):
        """
        Add points to the psychrometric chart.

        Args:
            en: Moist air enthalpy (kJ/kg). Can be a numpy array.
            hr: Humidity Ratio (g/kg). Can be a numpy array.
            **kwargs: Additional keyword arguments to be passed to plotly's go.Scatter.
        """
        db, wb, rh, hr, en = self._pc.get_all(en=en, hr=hr)
        customdata = np.vstack([db, wb, rh, hr, en]).T
        x, y = self._skew_transform(np.atleast_1d(en), np.atleast_1d(hr))

        # if "mode" is not specified, set it to "markers"
        if "mode" not in kwargs:
            kwargs["mode"] = "markers"

        self.add_trace(
            go.Scatter(
                x=x,
                y=y,
                customdata=customdata,
                hovertemplate=(
                    "DB: %{customdata[0]:.1f}degC<br>"
                    + "WB: %{customdata[1]:.1f}degC<br>"
                    + "RH: %{customdata[2]:.1f}%<br>"
                    + "HR: %{customdata[3]:.1f}g/kg<br>"
                    + "EN: %{customdata[4]:.1f}kJ/kg"
                ),
                **kwargs,
            )
        )

    def draw_iso_rh_line(
        self,
        rh: float,
        db_range: Union[list[float], NDArray[np.float64]] = [-10, 70],
        **kwargs,
    ):
        """
        Draw a line of constant relative humidity on the psychrometric chart.

        Args:
            rh: Relative humidity (%) as a float.
            db_range: Range of dry bulb temperatures (degC) for which to draw the line.
            **kwargs: Additional keyword arguments to be passed to plotly's go.Scatter.
        """
        dbs = np.linspace(db_range[0], db_range[-1], 100)
        hrs = self._pc.get_hr_from_db_rh(dbs, rh)
        ens = self._pc.get_en_from_db_hr(dbs, hrs)
        x, y = self._skew_transform(ens, hrs)
        if "name" not in kwargs:
            kwargs["name"] = f"RH={rh:.0f}%"
        self._draw_line_from_xy(x, y, **kwargs)

    def draw_iso_db_line(
        self,
        db: float,
        rh_range: Union[list[float], NDArray[np.float64]] = [0, 100],
        **kwargs,
    ):
        """
        Draw a line of constant dry-bulb temperature on the psychrometric chart.

        Args:
            db: Dry bulb temperature (degC) as a float.
            rh_range: Range of relative humidities (%) for which to draw the line.
            **kwargs: Additional keyword arguments to be passed to plotly's go.Scatter.
        """
        rhs = np.linspace(rh_range[0], rh_range[-1], 100)
        hrs = self._pc.get_hr_from_db_rh(db, rhs)
        ens = self._pc.get_en_from_db_hr(db, hrs)
        x, y = self._skew_transform(ens, hrs)
        if "name" not in kwargs:
            kwargs["name"] = f"DB={db:.0f}degC"
        self._draw_line_from_xy(x, y, **kwargs)

    def draw_iso_hr_line(
        self,
        hr: float,
        db_range: Union[list[float], NDArray[np.float64]] = [-10, 70],
        **kwargs,
    ):
        """
        Draw a line of constant humidity ratio on the psychrometric chart.

        Args:
            hr: Humidity ratio (g/kg) as a float.
            db_range: Range of dry bulb temperatures (degC) for which to draw the line.
            **kwargs: Additional keyword arguments to be passed to plotly's go.Scatter.
        """
        dbs = np.array(db_range)
        hrs = np.ones_like(dbs) * hr
        ens = self._pc.get_en_from_db_hr(dbs, hrs)
        x, y = self._skew_transform(ens, hrs)
        if "name" not in kwargs:
            kwargs["name"] = f"HR={hr:.0f}g/kg"
        self._draw_line_from_xy(x, y, **kwargs)

    def draw_iso_en_line(
        self,
        en: float,
        db_range: Union[list[float], NDArray[np.float64]] = [-10, 70],
        **kwargs,
    ):
        """
        Draw a line of constant specific enthalpy on the psychrometric chart.

        Args:
            en: Specific enthalpy (kJ/kg) as a float.
            db_range: Range of dry bulb temperatures (degC) for which to draw the line.
            **kwargs: Additional keyword arguments to be passed to plotly's go.Scatter.
        """
        dbs = np.array(db_range)
        hrs = self._pc.get_hr_from_db_en(dbs, en)
        ens = np.ones_like(hrs) * en
        x, y = self._skew_transform(ens, hrs)
        if "name" not in kwargs:
            kwargs["name"] = f"EN={en:.0f}kJ/kg"
        self._draw_line_from_xy(x, y, **kwargs)

    def _add_iso_rh_annotation(self):
        """Add annotations to the iso RH lines."""
        db = 48.0
        rhs = np.arange(10.0, 101.0, 10.0)
        hrs = self._pc.get_hr_from_db_rh(db, rhs)
        ens = self._pc.get_en_from_db_hr(db, hrs)
        x, y = self._skew_transform(ens, hrs)
        texts = np.array([f"RH={rh:.0f}%" for rh in rhs])
        self._add_annotation_from_xy(x, y, texts)

    def _add_iso_en_annotation(self):
        """Add annotations to the iso enthalpy lines."""
        ens = np.array([20.0, 40.0, 60.0, 80.0, 100.0])  # kJ.kg-1
        hrs = np.array([8.0, 13.0, 18.0, 23.0, 28.0])  # g.kg-1
        x, y = self._skew_transform(ens, hrs)
        texts = np.array([f"EN={en:.0f}kJ/kg<sub>air</sub>" for en in ens])
        self._add_annotation_from_xy(x, y, texts)

    def _draw_line_from_xy(
        self,
        x: NDArray,
        y: NDArray,
        **kwargs,
    ):
        """Draw a line from x and y."""
        if "mode" not in kwargs:
            kwargs["mode"] = "lines"
        if "hoverinfo" not in kwargs:
            kwargs["hoverinfo"] = "skip"
        self.add_trace(
            go.Scatter(
                x=x,
                y=y,
                **kwargs,
            )
        )

    def add_histogram_2d_contour(
        self,
        en: Union[NDArray[np.float64], float],
        hr: Union[NDArray[np.float64], float],
        **kwargs,
    ):
        """
        Add a 2D histogram contour to the psychrometric chart.

        Args:
            en: Moist air enthalpy (kJ/kg).
            hr: Humidity Ratio (g/kg).
            **kwargs: Additional keyword arguments to be passed to plotly's go.Histogram2dContour.
        """
        x, y = self._skew_transform(np.atleast_1d(en), np.atleast_1d(hr))

        self.add_trace(
            go.Histogram2dContour(
                x=x,
                y=y,
                **kwargs,
            )
        )
        # add text annotation to the contour peak point
        if "name" in kwargs:
            name = kwargs["name"]
            x_gravity, y_gravity = np.mean(x), np.mean(y)
            self.add_trace(
                go.Scatter(
                    x=[x_gravity],
                    y=[y_gravity],
                    mode="text",
                    text=[name],
                    textfont=dict(size=8),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    def _add_annotation_from_xy(self, x: NDArray, y: NDArray, text: NDArray):
        """Add annotations to the figure."""
        self.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="text",
                text=text,
                textposition="top center",
                textfont=dict(size=8, color="#BDBDBD"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    def _xtick_label_to_db(self):
        """Set xtick labels as the dry-bulb temperature values."""
        dbs = np.arange(-10, 51, 10)
        ens = self._db_to_en_at_hr0(dbs)
        self.update_xaxes(tickvals=ens, ticktext=dbs)

    def _skew_transform(
        self, en: NDArray[np.float64], hr: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Transform the moist air enthalpy and humidity ratio to skew coordinates.

        Args:
            en: Moist air enthalpy (kJ/kg).
            hr: Humidity ratio (g/kg).

        Returns:
            A tuple of numpy arrays representing the x and y coordinates in the skew coordinate system.
        """
        affine_skew_matrix = np.array(
            [
                [1.0, 1 / self._slope],
                [0.0, 1.0],
            ],
        )
        vector = np.array(
            [
                en,
                hr,
            ]
        )
        orthogonal_points = affine_skew_matrix @ vector
        x = orthogonal_points[0, :]
        y = orthogonal_points[1, :]
        return x, y

    def _calc_skew_slope(self) -> float:
        """
        Calculate the slope of the skew lines in the psychrometric chart.

        This method uses fixed values for humidity ratio (30.0 g/kg) and dry bulb temperature (50.0 degC)
        to calculate the slope of the skew lines in the psychrometric chart.

        Returns:
            The slope of the skew lines as a float.
        """
        db = 50.0
        hrs = np.array([0.0, 30.0])
        ens = self._pc.get_en_from_db_hr(db, hrs)
        slope = (hrs[1] - hrs[0]) / (ens[0] - ens[1])
        return slope

    def _db_to_en_at_hr0(self, db: ArrayLike):
        """Calculate enthalpy at HR=0."""
        return self._pc.get_en_from_db_hr(db, 0.0)
