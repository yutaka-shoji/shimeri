

# shimeri
![PyPI Downloads](https://static.pepy.tech/badge/shimeri)

A Python package for calculating psychrometric properties of moist air and plotting psychrometric charts based on plotly.

*shimeri* means "moisture" or "humidity" in Japanese.

## Installation

``` sh
pip install shimeri
```

## Example

Usage example: [example.py](https://github.com/yutaka-shoji/shimeri/blob/main/example/example.py?raw=true)

Example Result:  
![Example Result](https://github.com/yutaka-shoji/shimeri/blob/main/example/example.png?raw=true)


## Web App

*shimeri on web* is available at [shimeri-web.vercel.app](https://shimeri-web.vercel.app/).

## Attention
The `PsychrometricCalculator.get_all()` method uses convergence calculations. Especially when calculating from wet-bulb temperature and enthalpy, convergence can be poor, potentially leading to inaccurate results. (The poor convergence can be understood from the fact that the slopes of wet-bulb temperature and enthalpy lines are similar on the psychrometric chart.

