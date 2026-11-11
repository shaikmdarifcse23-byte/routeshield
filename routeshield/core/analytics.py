"""Analytics and reporting: risk trends by road, region, cohort and period.

Task 7 of the brief. These are the reports a fleet safety manager reads on a
Monday morning, and the reason the trip log in :mod:`routeshield.core.store`
records the predicted score alongside the chosen route: without that pairing
there is no way to ask "were we right?" after the fact.

Privacy: cohort-level only. Driver identity never enters a report, and vehicle
identifiers are hashed at ingest.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..data.events import DRIVER_COHORTS, EVENT_TYPES, MEAN_SEVERITY


def _weighted(events, mask, now, half_life):
    age = np.maximum(0.0, now - events.month[mask])
    return float((events.severity[mask] * 0.5 ** (age / half_life)).sum() / MEAN_SEVERITY)


def risk_by_road(model, net: dict, top_k: int = 15) -> list[dict]:
    """Aggregate segment risk up to named corridors."""
    risk = model.segment_risk(net)
    conf = model.confidence()
    agg: dict[str, dict] = defaultdict(
        lambda: {"risk": 0.0, "km": 0.0, "segments": 0, "events": 0.0, "low_conf": 0})
    for e, name in enumerate(net["name"]):
        a = agg[str(name)]
        a["risk"] += float(risk[e])
        a["km"] += float(net["length"][e])
        a["segments"] += 1
        a["events"] += float(model.y[e])
        a["low_conf"] += int(conf[e] == "Low")

    rows = []
    for name, a in agg.items():
        km = max(a["km"], 1e-9)
        rows.append({
            "road": name,
            "segments": a["segments"],
            "length_km": round(a["km"], 1),
            "risk_per_100km": round(a["risk"] / km * 100, 3),
            "weighted_events": round(a["events"], 1),
            "low_confidence_segments": a["low_conf"],
        })
    rows.sort(key=lambda r: -r["risk_per_100km"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows[:top_k]


def risk_by_cohort(events, now: float, half_life: float = 6.0) -> list[dict]:
    """Severity-weighted event share by driver tenure band."""
    ok = events.edge >= 0
    rows = []
    for i, name in enumerate(DRIVER_COHORTS):
        mask = ok & (events.cohort == i)
        n = int(mask.sum())
        rows.append({
            "cohort": name,
            "events": n,
            "weighted_events": round(_weighted(events, mask, now, half_life), 1),
            "share_pct": round(100 * n / max(int(ok.sum()), 1), 1),
            "severe_share_pct": round(
                100 * float(np.mean(np.isin(events.type_idx[mask],
                                            [EVENT_TYPES.index("near_miss"),
                                             EVENT_TYPES.index("accident")])))
                if n else 0.0, 1),
        })
    return rows


def risk_by_hour(events, half_life: float = 6.0,
                 exposure_share_by_hour: np.ndarray | None = None) -> list[dict]:
    """Event load by hour of day, as a RATE index rather than a raw share.

    A raw share by hour is close to useless: a flat line can mean either
    "every hour is equally dangerous" or "nobody drives at 3 a.m. but the few
    who do crash constantly". What a safety team needs is events relative to
    exposure. Without hour-level telemetry we use the documented night-driving
    share as the exposure denominator, and say so.
    """
    from ..data.events import NIGHT_SHARE

    ok = events.edge >= 0
    hours = np.clip(events.hour[ok].astype(int), 0, 23)
    w = events.severity[ok]
    total = max(float(w.sum()), 1e-9)

    if exposure_share_by_hour is None:
        night = np.array([1 if (h >= 22 or h < 5) else 0 for h in range(24)])
        n_night = int(night.sum())
        exposure_share_by_hour = np.where(
            night == 1, NIGHT_SHARE / n_night, (1 - NIGHT_SHARE) / (24 - n_night))
    exp_share = exposure_share_by_hour / exposure_share_by_hour.sum()

    rows = []
    for h in range(24):
        share = float(w[hours == h].sum()) / total
        rows.append({
            "hour": int(h),
            "events": int((hours == h).sum()),
            "weighted_share_pct": round(100 * share, 2),
            "exposure_share_pct": round(100 * float(exp_share[h]), 2),
            "rate_index": round(share / max(float(exp_share[h]), 1e-9), 3),
            "night_window": bool(h >= 22 or h < 5),
        })
    return rows


def monthly_trend(events, months: int = 12, half_life: float = 6.0) -> list[dict]:
    """Severity-weighted event volume by month, with a severe-event split."""
    ok = events.edge >= 0
    m = events.month[ok].astype(int)
    sev_idx = [EVENT_TYPES.index("near_miss"), EVENT_TYPES.index("accident")]
    rows = []
    for k in range(int(m.min()), int(m.max()) + 1):
        sel = m == k
        if not sel.any():
            continue
        rows.append({
            "month": int(k),
            "events": int(sel.sum()),
            "weighted_events": round(float(events.severity[ok][sel].sum() / MEAN_SEVERITY), 1),
            "severe_events": int(np.isin(events.type_idx[ok][sel], sev_idx).sum()),
        })
    return rows


def recurring_hotspots(hotspots: list[dict], min_score: float = 0.0) -> list[dict]:
    """Searchable register of persistent black spots, worsening first."""
    rows = [h for h in hotspots if h["weighted_score"] >= min_score]
    return sorted(rows, key=lambda h: (h["trend_label"] != "worsening",
                                       -h["weighted_score"]))


def fleet_summary(model, net: dict, events, hotspots: list[dict], now: float) -> dict:
    risk = model.segment_risk(net)
    conf = model.confidence()
    ok = int((events.edge >= 0).sum())
    return {
        "segments": int(len(net["u"])),
        "network_km": round(float(net["length"].sum()), 1),
        "matched_events": ok,
        "weighted_events": round(float(model.y.sum()), 1),
        "hotspots": len(hotspots),
        "worsening_hotspots": sum(1 for h in hotspots if h["trend_label"] == "worsening"),
        "high_confidence_pct": round(100 * float(np.mean(conf == "High")), 1),
        "low_confidence_pct": round(100 * float(np.mean(conf == "Low")), 1),
        "median_risk_per_100km": round(float(np.median(
            risk / np.maximum(net["length"], 1e-9) * 100)), 3),
        "p90_risk_per_100km": round(float(np.percentile(
            risk / np.maximum(net["length"], 1e-9) * 100, 90)), 3),
    }
