"""Dispatch-time context: time of day, weather, driver cohort.

A segment's structural risk is a baseline. What a dispatcher actually needs is
the risk of *this* trip, leaving at *this* hour, in *today's* weather. These
multipliers are applied on top of the posterior rate at scoring time, so the
nightly batch never has to be re-run to answer a question about tonight.

Calibration note: the multipliers below are order-of-magnitude values grounded
in published Indian road-safety statistics (MoRTH annual reports consistently
show roughly a third of fatal crashes in the night window despite a much
smaller traffic share, and a pronounced wet-weather and fog penalty). They are
deliberately exposed as a single editable table rather than buried in code,
because the first thing a real deployment must do is re-fit them on the
operator's own data. :func:`calibrate_from_events` does exactly that.
"""
from __future__ import annotations

import numpy as np

# Hour-of-day multiplier. The 22:00-05:00 band is the fatigue window that
# dominates commercial-fleet night running in India.
HOUR_MULTIPLIER = {
    **{h: 1.55 for h in (22, 23, 0, 1, 2, 3, 4)},
    **{h: 1.20 for h in (5, 21)},
    **{h: 1.00 for h in (6, 7, 10, 11, 12, 13, 14, 15, 16, 20)},
    **{h: 1.15 for h in (8, 9, 17, 18)},     # peak-hour conflict exposure
    19: 1.25,                                 # dusk + peak together
}

WEATHER_MULTIPLIER = {
    "clear": 1.00,
    "cloudy": 1.02,
    "rain": 1.30,
    "heavy_rain": 1.55,
    "fog": 1.60,
    "thunderstorm": 1.45,
}

COHORT_MULTIPLIER = {
    "tenure_lt_1y": 1.45,
    "tenure_1_3y": 1.00,
    "tenure_gt_3y": 0.78,
}

# Open-Meteo WMO weather codes -> our buckets.
WMO_TO_BUCKET = {
    0: "clear", 1: "clear", 2: "cloudy", 3: "cloudy",
    45: "fog", 48: "fog",
    51: "rain", 53: "rain", 55: "rain", 56: "rain", 57: "rain",
    61: "rain", 63: "rain", 65: "heavy_rain",
    66: "rain", 67: "heavy_rain",
    80: "rain", 81: "rain", 82: "heavy_rain",
    95: "thunderstorm", 96: "thunderstorm", 99: "thunderstorm",
}


def context_multiplier(hour: int = 12, weather: str = "clear",
                       cohort: str | None = None) -> float:
    """Combined multiplier applied to a segment's posterior rate."""
    m = HOUR_MULTIPLIER.get(int(hour) % 24, 1.0)
    m *= WEATHER_MULTIPLIER.get(weather, 1.0)
    if cohort:
        m *= COHORT_MULTIPLIER.get(cohort, 1.0)
    return float(m)


def context_breakdown(hour: int = 12, weather: str = "clear",
                      cohort: str | None = None) -> list[dict]:
    """Itemised multipliers for the UI, so the number is never a black box."""
    out = [{"factor": "Time of day",
            "detail": f"{int(hour):02d}:00",
            "multiplier": HOUR_MULTIPLIER.get(int(hour) % 24, 1.0)},
           {"factor": "Weather",
            "detail": weather.replace("_", " "),
            "multiplier": WEATHER_MULTIPLIER.get(weather, 1.0)}]
    if cohort:
        out.append({"factor": "Driver cohort", "detail": cohort,
                    "multiplier": COHORT_MULTIPLIER.get(cohort, 1.0)})
    return out


def calibrate_from_events(events, exposure_share_by_hour: np.ndarray | None = None
                          ) -> dict[int, float]:
    """Re-fit hour multipliers from observed events.

    multiplier(h) = (event share in hour h) / (exposure share in hour h),
    normalised so the exposure-weighted mean is 1. With no exposure telemetry
    we fall back to a flat prior, which is a documented limitation rather than
    a hidden assumption.
    """
    hours = np.clip(events.hour.astype(int), 0, 23)
    w = events.severity
    ev_share = np.array([w[hours == h].sum() for h in range(24)])
    ev_share = ev_share / max(ev_share.sum(), 1e-9)
    if exposure_share_by_hour is None:
        exposure_share_by_hour = np.full(24, 1 / 24)
    exp_share = exposure_share_by_hour / exposure_share_by_hour.sum()
    mult = ev_share / np.maximum(exp_share, 1e-9)
    mult = mult / max(float(np.sum(mult * exp_share)), 1e-9)
    return {h: float(mult[h]) for h in range(24)}


# --------------------------------------------------------------------------
# weather feed
# --------------------------------------------------------------------------
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"


def fetch_weather(lat: float, lon: float, timeout: int = 20) -> dict:  # pragma: no cover
    """Current conditions from Open-Meteo (free, no key, CC-BY 4.0).

    Falls back to a climatology bucket when the network is unavailable, so a
    dispatcher never sees a blank score because a third-party API blinked.
    """
    import json
    import urllib.parse
    import urllib.request

    try:
        q = urllib.parse.urlencode({
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,precipitation,weather_code,visibility",
            "timezone": "Asia/Kolkata",
        })
        with urllib.request.urlopen(f"{OPEN_METEO}?{q}", timeout=timeout) as r:
            payload = json.loads(r.read().decode())
        cur = payload.get("current", {})
        bucket = WMO_TO_BUCKET.get(int(cur.get("weather_code", 0)), "clear")
        return {"bucket": bucket, "source": "open-meteo",
                "weather_code": cur.get("weather_code"),
                "precipitation_mm": cur.get("precipitation"),
                "temperature_c": cur.get("temperature_2m")}
    except Exception as exc:
        return {"bucket": climatology_bucket(), "source": "climatology-fallback",
                "error": str(exc)}


def climatology_bucket(month: int | None = None) -> str:
    """Offline fallback: the modal condition for a Deccan-plateau month."""
    import datetime as dt
    m = month or dt.date.today().month
    if m in (6, 7, 8, 9):
        return "rain"           # south-west monsoon
    if m in (10, 11):
        return "cloudy"         # retreating monsoon
    if m in (12, 1):
        return "fog"            # early-morning fog window
    return "clear"
