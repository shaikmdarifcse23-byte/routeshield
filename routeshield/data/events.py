"""Telematics events: canonical schema, CSV loader, and a labelled synthetic generator.

The generator exists because fleet telematics is proprietary — no public Indian
dataset of harsh-braking / near-miss events with GPS traces exists. It is
seeded from *real* structure (road features from OSM, MoRTH severity mix,
India-specific night / monsoon / fog exposure) so that offline evaluation is a
meaningful test of the pipeline, and it is labelled ``SYNTHETIC`` wherever it
surfaces.

Real data drops in through :func:`load_events_csv` with no other change.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass

import numpy as np

# Severity weights. Ratios follow the MoRTH "Road Accidents in India" severity
# split (fatal : grievous : minor) carried down to proxy telematics events;
# a logged accident counts for an order of magnitude more than a harsh brake.
SEVERITY_WEIGHT: dict[str, float] = {
    "harsh_accel": 0.8,
    "harsh_brake": 1.0,
    "harsh_turn": 1.0,
    "overspeed": 1.2,
    "near_miss": 3.0,
    "accident": 10.0,
}
EVENT_TYPES = list(SEVERITY_WEIGHT)

# Observed mix in commercial-fleet telematics feeds.
EVENT_MIX = np.array([0.15, 0.36, 0.16, 0.12, 0.17, 0.04])

MEAN_SEVERITY = float(np.dot(list(SEVERITY_WEIGHT.values()), EVENT_MIX))

DRIVER_COHORTS = ("tenure_lt_1y", "tenure_1_3y", "tenure_gt_3y")
COHORT_MULTIPLIER = {"tenure_lt_1y": 1.45, "tenure_1_3y": 1.0, "tenure_gt_3y": 0.78}

# India-specific exposure context -------------------------------------------
# Night driving share and its risk multiplier are the single largest
# behavioural factor in Indian commercial-fleet crash data.
NIGHT_SHARE = 0.28
NIGHT_MULTIPLIER = 1.55
# Monsoon months (Jun-Sep) over the Deccan plateau.
MONSOON_MONTHS = {6, 7, 8, 9}
RAIN_MULTIPLIER = 1.30
# North-Indian fog season; mild in Bengaluru but kept for corridor portability.
FOG_MONTHS = {12, 1}
FOG_MULTIPLIER = 1.60

EVENT_COLUMNS = (
    "event_id", "ts", "lat", "lon", "type", "severity",
    "vehicle_id", "driver_cohort", "speed_kmh", "trip_id",
)


@dataclass
class EventTable:
    """Columnar event store. ``edge`` is filled in by map-matching."""

    lat: np.ndarray
    lon: np.ndarray
    month: np.ndarray            # float months since t0
    type_idx: np.ndarray
    severity: np.ndarray
    cohort: np.ndarray           # index into DRIVER_COHORTS
    hour: np.ndarray
    edge: np.ndarray | None = None
    true_edge: np.ndarray | None = None   # generating segment (synthetic only)
    source: str = "SYNTHETIC"
    # Average exposure-context multiplier (night share x season) applied when
    # these events were generated. The fitted rate estimates theta * this, so
    # evaluation against the latent theta must divide it out.
    context_factor: float = 1.0

    def __len__(self) -> int:
        return len(self.lat)

    @property
    def type_name(self) -> list[str]:
        return [EVENT_TYPES[i] for i in self.type_idx]


# --------------------------------------------------------------------------
# real-data loader
# --------------------------------------------------------------------------
def load_events_csv(path: str, t0_iso: str | None = None) -> EventTable:
    """Load a real telematics export. See docs/EVENT_SCHEMA.md for columns."""
    import datetime as dt

    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    if not rows:
        raise ValueError(f"{path} contains no rows")
    stamps = [dt.datetime.fromisoformat(r["ts"].replace("Z", "+00:00")) for r in rows]
    t0 = dt.datetime.fromisoformat(t0_iso) if t0_iso else min(stamps)
    month = np.array([(s - t0).total_seconds() / (30.44 * 86400) for s in stamps])
    type_idx = np.array([EVENT_TYPES.index(r["type"]) for r in rows])
    sev = np.array([
        float(r["severity"]) * 10 if r.get("severity") else SEVERITY_WEIGHT[r["type"]]
        for r in rows
    ])
    cohort = np.array([
        DRIVER_COHORTS.index(r["driver_cohort"]) if r.get("driver_cohort") in DRIVER_COHORTS else 1
        for r in rows
    ])
    return EventTable(
        lat=np.array([float(r["lat"]) for r in rows]),
        lon=np.array([float(r["lon"]) for r in rows]),
        month=month,
        type_idx=type_idx,
        severity=sev,
        cohort=cohort,
        hour=np.array([s.hour for s in stamps], dtype=float),
        source="REAL_FLEET",
    )


# --------------------------------------------------------------------------
# synthetic generator
# --------------------------------------------------------------------------
def severity_tilt(net: dict) -> np.ndarray:
    """Per-segment tilt toward severe outcomes.

    Severity is not uniform across a road network, and modelling it as if it
    were would make the severity half of the model estimate pure noise. Indian
    crash statistics are unambiguous on the direction: fatality and
    grievous-injury rates are far higher on high-speed national highways, on
    unlit stretches and on gradients than on congested urban arterials, where
    crashes are frequent but slow. So the probability of a near-miss or a
    logged accident (as opposed to a harsh brake) rises with speed limit,
    absence of lighting and gradient.
    """
    return np.exp(0.55 * net["unlit"]
                  + 0.90 * (net["maxspeed"] / 100.0 - 0.60)
                  + 0.12 * net["gradient"])


def severe_share(net: dict) -> np.ndarray:
    """Share of events on each segment that are near-miss or accident."""
    base = EVENT_MIX[EVENT_TYPES.index("near_miss")] + EVENT_MIX[EVENT_TYPES.index("accident")]
    return np.clip(base * severity_tilt(net), 0.04, 0.62)


def expected_severity(net: dict) -> np.ndarray:
    """Mean severity weight of an event on each segment (ground truth)."""
    i_nm, i_ac = EVENT_TYPES.index("near_miss"), EVENT_TYPES.index("accident")
    base = EVENT_MIX.copy()
    base_sev = base[i_nm] + base[i_ac]
    out = np.empty(len(net["u"]))
    tgt = severe_share(net)
    for e in range(len(out)):
        mix = base.copy()
        mix[i_nm] = base[i_nm] / base_sev * tgt[e]
        mix[i_ac] = base[i_ac] / base_sev * tgt[e]
        rest = [i for i in range(len(mix)) if i not in (i_nm, i_ac)]
        mix[rest] = base[rest] / base[rest].sum() * (1 - tgt[e])
        out[e] = float(np.dot(mix, list(SEVERITY_WEIGHT.values())))
    return out


def latent_rate(net: dict, seed: int = 11) -> np.ndarray:
    """Hidden per-vehicle-km event rate the model must recover.

    Built from the *real* road features plus unobserved local effects, so the
    model cannot simply invert its own feature set.
    """
    rng = np.random.default_rng(seed)
    curv = net["curvature"]
    junc = net["junction_density"]
    unlit = net["unlit"]
    grad = net["gradient"]
    speed = net["maxspeed"] / 100.0

    lin = (
        -5.40
        + 1.90 * curv
        + 0.115 * junc
        + 0.42 * unlit
        + 0.14 * grad
        + 0.55 * speed
        + 0.30 * curv * speed          # fast + curvy interacts (model must learn it)
    )
    # Unobserved local effects (surface quality, enforcement, encroachment) are
    # a property of the physical road, so both directions share the same draw.
    rid = net["road_id"]
    noise = rng.normal(0, 0.42, int(rid.max()) + 1)[rid]
    return np.exp(lin + noise)


def sample_events(
    net: dict,
    theta: np.ndarray,
    months: float,
    t0: float = 0.0,
    seed: int = 12,
    start_calendar_month: int = 1,
    gps_noise_m: float = 18.0,
) -> EventTable:
    """Poisson-sample events, place them on real segment geometry with GPS
    noise, and stamp them with realistic time-of-day and cohort."""
    rng = np.random.default_rng(seed)
    n_edges = len(theta)

    # Context multiplier averaged over the simulated window, per calendar month.
    cal_months = [(start_calendar_month - 1 + int(k)) % 12 + 1 for k in range(max(1, int(months)))]
    ctx = float(np.mean([
        (RAIN_MULTIPLIER if cm in MONSOON_MONTHS else 1.0)
        * (FOG_MULTIPLIER if cm in FOG_MONTHS else 1.0)
        for cm in cal_months
    ]))
    night_factor = (1 - NIGHT_SHARE) + NIGHT_SHARE * NIGHT_MULTIPLIER

    lam = theta * net["exposure"] * months * ctx * night_factor
    counts = rng.poisson(lam)
    total = int(counts.sum())
    if total == 0:
        raise ValueError("No events sampled; raise exposure or months.")

    edge = np.repeat(np.arange(n_edges), counts)

    # Event type depends on where it happened: the same collision energy that
    # makes a highway crash severe makes an urban one a harsh brake.
    i_nm, i_ac = EVENT_TYPES.index("near_miss"), EVENT_TYPES.index("accident")
    base = EVENT_MIX.copy()
    base_sev = base[i_nm] + base[i_ac]
    rest_idx = [i for i in range(len(base)) if i not in (i_nm, i_ac)]
    tgt = severe_share(net)
    type_idx = np.empty(total, dtype=int)
    for e in np.unique(edge):
        sel = edge == e
        mix = base.copy()
        mix[i_nm] = base[i_nm] / base_sev * tgt[e]
        mix[i_ac] = base[i_ac] / base_sev * tgt[e]
        mix[rest_idx] = base[rest_idx] / base[rest_idx].sum() * (1 - tgt[e])
        type_idx[sel] = rng.choice(len(EVENT_TYPES), int(sel.sum()), p=mix / mix.sum())
    severity = np.array([SEVERITY_WEIGHT[EVENT_TYPES[i]] for i in type_idx])
    cohort = rng.choice(len(DRIVER_COHORTS), total, p=[0.3, 0.42, 0.28])
    month = t0 + rng.uniform(0, months, total)

    # Time of day. Night driving is NIGHT_SHARE of *exposure* and is
    # NIGHT_MULTIPLIER times riskier, so its share of *events* is higher than
    # its share of hours:
    #
    #     p(night event) = s*m / (s*m + (1-s))
    #
    # Sampling events at the exposure share instead would quietly encode
    # "night is exactly as safe as day", contradicting the multiplier the rest
    # of the system is built on — and the hour-of-day report would show a flat
    # line while the model claimed a 1.55x night penalty.
    night_event_share = (NIGHT_SHARE * NIGHT_MULTIPLIER) / (
        NIGHT_SHARE * NIGHT_MULTIPLIER + (1 - NIGHT_SHARE))
    is_night = rng.random(total) < night_event_share
    hour = np.where(is_night,
                    (rng.uniform(22, 29, total)) % 24,
                    rng.uniform(5, 22, total))

    # Drop each event somewhere along its segment polyline, then add GPS error,
    # so the map-matching stage has real work to do.
    lat = np.empty(total); lon = np.empty(total)
    deg_per_m = 1.0 / 111_320.0
    for i, e in enumerate(edge):
        pts = net["geometry"][e]
        t = rng.random() * (len(pts) - 1)
        k = int(t); f = t - k
        a, b = pts[k], pts[min(k + 1, len(pts) - 1)]
        la = a[0] + (b[0] - a[0]) * f
        lo = a[1] + (b[1] - a[1]) * f
        lat[i] = la + rng.normal(0, gps_noise_m * deg_per_m)
        lon[i] = lo + rng.normal(0, gps_noise_m * deg_per_m / math.cos(math.radians(la)))

    return EventTable(lat=lat, lon=lon, month=month, type_idx=type_idx,
                      severity=severity, cohort=cohort, hour=hour,
                      edge=None, true_edge=edge, source="SYNTHETIC",
                      context_factor=float(ctx * night_factor))
