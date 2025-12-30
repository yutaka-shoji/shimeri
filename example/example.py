import numpy as np
import pandas as pd

import shimeri as sh

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
        name="a line",
        mode="lines",
    )

    # draw constant humidity ratio line from half-mixed point to rh=90%
    hr_mixed = np.mean(hrs)
    en_mixed = np.mean(ens)

    # cooling process to RH=90%, DB=15degC
    en_90 = pc.get_en_from_db_rh(15.0, 90.0)
    hr_90 = pc.get_hr_from_db_rh(15.0, 90.0)
    fig.add_points(
        en=[en_mixed, en_90],
        hr=[hr_mixed, hr_90],
        name="cooling to RH=90%",
        mode="lines+markers",
    )

    # db_90, wb_90, rh_90, hr_90, en_90 = pc.get_all(hr=hr_mixed, rh=90)

    # draw a line of constant relative humidity
    fig.draw_iso_rh_line(
        rh=90,
        db_range=np.array([25.0, 10.0]),
        mode="lines",
    )

    # draw a line of constant humidity ratio
    fig.draw_iso_hr_line(
        hr=float(hr_90),
        db_range=np.array([15.0, 25.0]),
        mode="lines+markers",
    )

    # density plot
    df = pd.read_csv("db_rh_tokyo_2023.csv", parse_dates=True, index_col=0)
    dbs = df.loc["2023-07":"2023-08", "db"].to_numpy()
    rhs = df.loc["2023-07":"2023-08", "rh"].to_numpy()
    # hrs = pc.get_hr_from_db_rh(dbs, rhs)
    # ens = pc.get_en_from_db_hr(dbs, hrs)
    dbs, wbs, rhs, hrs, ens = pc.get_all(db=dbs, rh=rhs)
    fig.add_histogram_2d_contour(
        en=ens,
        hr=hrs,
        name="tokyo summer 2023",
        colorscale=[[0, "rgba(255,255,255,0)"], [1, "rgba(255,0,0,255)"]],
        contours_showlines=False,
        showscale=False,
    )
    dbs = df.loc["2023-01":"2023-02", "db"].to_numpy()
    rhs = df.loc["2023-01":"2023-02", "rh"].to_numpy()
    hrs = pc.get_hr_from_db_rh(dbs, rhs)
    ens = pc.get_en_from_db_hr(dbs, hrs)
    fig.add_histogram_2d_contour(
        en=ens,
        hr=hrs,
        name="tokyo winter 2023",
        colorscale=[[0, "rgba(255,255,255,0)"], [1, "rgba(0,0,255,255)"]],
        contours_showlines=False,
        showscale=False,
    )

    # you can modify layout as a plotly figure
    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bordercolor="Black",
            borderwidth=1,
        ),
        margin=dict(l=20, r=20, t=20, b=20),
    )

    fig.show()
    fig.write_image("example.png", scale=4)
