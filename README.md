# shimeri

## Overview

A Python package for calculating psychrometric properties of moist air and plotting psychrometric charts based on plotly.

## Motivation
Psychrometric charts are drawn on an oblique coordinate system with enthalpy and humidity ratio as axes.  
As a result, dry-bulb temperatures are not precisely equidistant.  
This tool managed this characteristic.


## Installation

```bash
pip install shimeri
```

## Usage
[See sample.py](/sample.py)

Sample Result:  
![Sample Result](https://github.com/yutaka-shoji/shimeri/blob/main/sample.png?raw=true)

## Attention
The `PsychrometricCalculator.get_all()` method uses convergence calculations. Especially when calculating from wet-bulb temperature and enthalpy, convergence can be poor, potentially leading to inaccurate results. (The poor convergence can be understood from the fact that the slopes of wet-bulb temperature and enthalpy lines are similar on the psychrometric chart.

