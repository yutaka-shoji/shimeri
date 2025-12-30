# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

shimeri is a Python package for calculating psychrometric properties of moist air and plotting psychrometric charts using plotly.

## Development Commands

```sh
# Install dependencies
uv sync

# Linting
uv run ruff check .
uv run ruff format .

# Type checking
uv run pyright

# Serve documentation locally
uv run mkdocs serve
```

## Architecture

The package has two main modules:

- **psychrometrics.py** (`PsychrometricCalculator`): Core calculation engine for psychrometric properties (dry bulb temp, wet bulb temp, relative humidity, humidity ratio, enthalpy). Uses CoolProp's `HAPropsSI` function for accurate thermodynamic calculations given any 2 of 5 variables.

- **psychrometricchart.py** (`PsychrometricChart`): Visualization class extending `plotly.graph_objects.Figure`. Uses a skew transformation to display enthalpy lines vertically. All points are plotted using enthalpy (en) and humidity ratio (hr) as coordinates.

## Key Concepts

- Psychrometric variables: db (dry bulb °C), wb (wet bulb °C), rh (relative humidity %), hr (humidity ratio g/kg), en (specific enthalpy kJ/kg)
- Default atmospheric pressure: 101.325 kPa
- Thermodynamic properties calculated via CoolProp library
