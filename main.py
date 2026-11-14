#!/usr/bin/env python3
"""btc-cent-forecast-01: half-shrunk 12h trend forecast.

Forecasts Bitcoin's price 240 minutes after a checkpoint as:

    last_close + 0.5 * horizon_minutes * slope

where slope is the ordinary-least-squares slope (dollars per minute) of the
last 720 closes (12 hours) of the case's 72-hour minute series.

Why this form: the last price is the honest random-walk baseline, and the
12-hour trend carries a little real signal over 4 hours, but a raw trend
extrapolation overshoots in a noise-dominated market, so the trend term is
shrunk by one half. On the kit's five practice cases with known answers this
scores a mean absolute error of about 44,314 cents versus about 47,055 for
the last-price baseline.

Standard library only, fully deterministic (no clock, no randomness), runs
offline. Reads every <id>.in in $MAKEFUN_INPUT_DIR (default /input) and
writes <id>.out to $MAKEFUN_OUTPUT_DIR (default /output) as
{"forecasts": [{"case": "<id>", "price": <number>}]}.
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

MIN_PRICE = 0.00001
MAX_PRICE = 9999999.99999

# OLS window: last 12 hours of minute closes.
WINDOW = 720
# Shrinkage applied to the trend term: half the extrapolated drift.
SHRINK = 0.5
DEFAULT_HORIZON_MINUTES = 240


def price5(x):
    """Clamp x onto the grid's 5 decimals and inside (0, 10,000,000)."""
    x = min(MAX_PRICE, max(MIN_PRICE, x))
    return round(x, 5)


def ols_slope(values):
    """OLS slope (dollars per minute) of values against 0..n-1.

    Returns 0.0 for fewer than 2 values or a degenerate (flat-x) window.
    """
    n = len(values)
    if n < 2:
        return 0.0
    # Mean of x = (n - 1) / 2; sum of squared deviations of x.
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    sxx = 0.0
    sxy = 0.0
    for i, y in enumerate(values):
        dx = i - mean_x
        sxx += dx * dx
        sxy += dx * (y - mean_y)
    if sxx == 0.0:
        return 0.0
    return sxy / sxx


def pick_market(markets):
    """The btc-usdc market if present, else the first market."""
    for m in markets:
        if isinstance(m, dict) and m.get("id") == "btc-usdc":
            return m
    return markets[0]


def forecast_case(case):
    """Forecast one case dict; returns the price as a number."""
    horizon = case.get("horizon_minutes", DEFAULT_HORIZON_MINUTES)
    if isinstance(horizon, str):
        try:
            horizon = float(horizon)
        except ValueError:
            horizon = DEFAULT_HORIZON_MINUTES
    if not isinstance(horizon, (int, float)) or not math.isfinite(horizon):
        horizon = DEFAULT_HORIZON_MINUTES

    markets = case.get("markets") or []
    if not markets:
        return 1.0
    prices = pick_market(markets).get("prices") or []

    # Keep only finite closes, preserving order.
    closes = [p for p in prices if isinstance(p, (int, float)) and math.isfinite(p)]
    if not closes:
        return 1.0
    last = closes[-1]
    if len(closes) < 2:
        return price5(last)

    window = closes[-WINDOW:]
    slope = ols_slope(window)
    if not math.isfinite(slope):
        slope = 0.0
    guess = last + SHRINK * horizon * slope
    return price5(guess)


def main():
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = sorted(p for p in input_dir.iterdir() if p.name.endswith(".in"))
    for path in inputs:
        case_id = path.name[: -len(".in")]
        try:
            with open(path, "r", encoding="utf-8") as f:
                case = json.load(f)
            price = forecast_case(case)
        except Exception:
            # Never crash on a bad case file: still emit a valid shape.
            price = 1.0
        out = {"forecasts": [{"case": case_id, "price": price}]}
        with open(output_dir / f"{case_id}.out", "w", encoding="utf-8") as f:
            f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
