# shimeri

A Python package for calculating psychrometric properties of moist air and plotting psychrometric charts based on plotly.

*shimeri* means "moisture" or "humidity" in Japanese.

## Installation

``` sh
pip install shimeri
```

## Quick Usage

Usage example below.

``` py title="sample.py"
import shimeri as sh
import numpy as np

if __name__ == "__main__":
    # instantiate psychrometric calculator
    pc = sh.PsychrometricCalculator()

    # calculate psychrometric properties at 25degC dry-bulb temperature and 50% relative humidity
    db, wb, rh, hr, en = pc.get_all(db=25, rh=50)
    print(
        f"DB={db:.1f}degC, WB={wb:.1f}degC, RH={rh:.1f}%, HR={hr:.1f}g/kg, EN={en:.1f}kJ/kg"
    )

    # initialize a psychrometric chart
    fig = sh.PsychrometricChart()

    # plot random points
    rng = np.random.default_rng()
    dbs = rng.normal(25, 5, 30)
    rhs = rng.normal(50, 5, 30)
    dbs, wbs, rhs, hrs, ens = pc.get_all(db=dbs, rh=rhs)
    fig.add_points(
        en=ens,
        hr=hrs,
        name="random points",
        mode="markers",
    )

    # add a line from points
    dbs = np.array([26.0, 35.0])
    rhs = np.array([50.0, 60.0])
    hrs = pc.get_hr_from_db_rh(dbs, rhs)
    ens = pc.get_en_from_db_hr(dbs, hrs)
    fig.add_points(
        en=ens,
        hr=hrs,
        name="line",
        mode="lines",
    )

    # draw constant humidity ratio line from half-mixed point to rh=90%
    hr_mixed = (hrs[0] + hrs[1]) * 0.5
    db_mixed = (dbs[0] + dbs[1]) * 0.5

    db_90, wb_90, rh_90, hr_90, en_90 = pc.get_all(hr=hr_mixed, rh=90)

    fig.draw_iso_hr_line(
        hr=hr_mixed,
        db_range=np.array([db_mixed, db_90]),
        mode="lines+markers",
    )

    # draw a line of constant relative humidity
    fig.draw_iso_rh_line(
        rh=90,
        db_range=np.array([db_90, 15.0]),
        mode="lines",
    )

    fig.show()
```

Sample Result:  
![Sample Result](https://github.com/yutaka-shoji/shimeri/blob/main/sample.png?raw=true)

## Attention
The `PsychrometricCalculator.get_all()` method uses convergence calculations. Especially when calculating from wet-bulb temperature and enthalpy, convergence can be poor, potentially leading to inaccurate results. (The poor convergence can be understood from the fact that the slopes of wet-bulb temperature and enthalpy lines are similar on the psychrometric chart.

