#!/usr/bin/env python3
"""btc-cent-forecast-01 entry: half-shrunk 12-hour trend forecast.

Forecasts Bitcoin's price 240 minutes after the checkpoint.

Model (chosen on the kit's practice data only, 5 known practice cases
plus 72 rolling pseudo-cases cut inside the practice series):
  guess = last_price + 0.5 * 240 * slope
where slope is the OLS slope (dollars per minute) of the last 720
closes (12 hours). The 0.5 shrinkage is deliberate: 4-hour BTC moves
are noise-dominated, so the raw trend extrapolation is halved. On the
practice data this beats the last-price (persistence) baseline by
~6% in mean absolute error on both the 5 real practice cases and the
72 pseudo-cases, and wins in both halves of the pseudo set; adding
mean-reversion terms or volatility caps did not improve it clearly,
so the simplest winner is kept.

Fully deterministic (no clock, no randomness), O(n) in the 4,320
prices, robust to short/flat series and spikes (slope is bounded by
the 720-minute window and the 0.5 shrink; output is clamped to the
scorer's price bounds on the 5-decimal grid).

I/O per the kit's RULES.txt: reads every <id>.in in
$MAKEFUN_INPUT_DIR (default /input), writes <id>.out to
$MAKEFUN_OUTPUT_DIR (default /output) as
{"forecasts": [{"case": "<id>", "price": <number>}]}.
"""

import json
import math
import os
import sys

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

MIN_PRICE = 0.00001
MAX_PRICE = 9_999_999.99999
HORIZON_MINUTES = 240
TREND_WINDOW = 720          # minutes of closes for the OLS slope
SHRINK = 0.5                # shrinkage applied to the extrapolated trend


def clamp_price(x):
    """Keep a guess on the price grid's 5 decimals, inside (0, 10_000_000)."""
    if not isinstance(x, float) and not isinstance(x, int):
        return None
    if not math.isfinite(x):
        return None
    x = min(MAX_PRICE, max(MIN_PRICE, x))
    return round(x * 100000) / 100000


def ols_slope(xs):
    """OLS slope (dollars per minute) of a list of closes, oldest first."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = sum(xs) / n
    num = 0.0
    den = 0.0
    for i, y in enumerate(xs):
        dx = i - mean_x
        num += dx * (y - mean_y)
        den += dx * dx
    if den == 0.0:
        return 0.0
    slope = num / den
    return slope if math.isfinite(slope) else 0.0


def forecast(prices, horizon_minutes=HORIZON_MINUTES):
    """One forecast from one case's closes (oldest first, last = checkpoint)."""
    clean = [p for p in prices if isinstance(p, (int, float)) and math.isfinite(p)]
    if not clean:
        raise ValueError("market has no finite prices")
    last = clean[-1]
    window = clean[-TREND_WINDOW:]
    slope = ols_slope(window)
    try:
        horizon = float(horizon_minutes)
    except (TypeError, ValueError):
        horizon = float(HORIZON_MINUTES)
    if not math.isfinite(horizon) or horizon <= 0:
        horizon = float(HORIZON_MINUTES)
    guess = last + SHRINK * horizon * slope
    clamped = clamp_price(guess)
    if clamped is None:
        clamped = clamp_price(last)
    return clamped


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    names = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    if not names:
        raise SystemExit("no .in files in %s" % INPUT_DIR)
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        markets = case.get("markets")
        if not isinstance(markets, list) or len(markets) != 1:
            raise ValueError("%s does not hold exactly one market" % name)
        prices = markets[0].get("prices")
        if not isinstance(prices, list) or not prices:
            raise ValueError("%s market has no prices" % name)
        price = forecast(prices, case.get("horizon_minutes", HORIZON_MINUTES))
        out = {"forecasts": [{"case": case_id, "price": price}]}
        with open(os.path.join(OUTPUT_DIR, case_id + ".out"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
