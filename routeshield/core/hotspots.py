"""Risk-hotspot detection: density clustering of severity-weighted events.

Segment scores answer "which road is risky". Dispatchers and safety managers
also need "*where exactly*, and is it getting worse" — a specific junction, a
specific curve. That is a spatial clustering problem, not a per-segment one,
because a black spot routinely straddles two or three segment boundaries.

HDBSCAN is used rather than DBSCAN or k-means: it finds clusters of varying
density (a dense urban junction and a sparse rural curve in the same pass),
needs no radius parameter guessed up front, and labels genuinely isolated
events as noise instead of forcing them into a cluster.

Events are weighted by severity x recency before clustering, so a cluster's
score reflects what actually happened there and when.
"""
from __future__ import annotations

import numpy as np

from ..data.events import EVENT_TYPES, MEAN_SEVERITY
from .matching import _local_xy


def detect_hotspots(net: dict, events, now: float,
                    half_life_months: float = 6.0,
                    min_cluster_size: int | None = None,
                    min_samples: int = 8,
                    max_radius_m: float = 400.0) -> list[dict]:
    """Cluster matched events in metric space; return ranked hotspots.

    ``min_cluster_size`` defaults to a fraction of the event volume. A fixed
    value makes the output meaningless as the fleet grows: with a few hundred
    events it finds nothing, with forty thousand it merges whole corridors
    into one blob and calls it a black spot.

    ``cluster_selection_method="leaf"`` is deliberate. The default ("eom")
    prefers large, stable clusters, which is the opposite of what is wanted
    here — a black spot is a junction or a curve, not a 20 km stretch of
    highway. Leaf selection returns the finest clusters in the hierarchy.
    Anything still wider than ``max_radius_m`` is a corridor, not a spot, and
    is dropped.
    """
    from sklearn.cluster import HDBSCAN

    ok = events.edge >= 0
    n_ok = int(ok.sum())
    if min_cluster_size is None:
        min_cluster_size = int(max(15, min(120, n_ok / 400)))
    if n_ok < min_cluster_size:
        return []

    lat, lon = events.lat[ok], events.lon[ok]
    lat0, lon0 = float(np.mean(net["node_lat"])), float(np.mean(net["node_lon"]))
    x, y = _local_xy(lat, lon, lat0, lon0)
    XY = np.column_stack([x, y])

    age = np.maximum(0.0, now - events.month[ok])
    sev_raw = events.severity[ok]                 # severity, no recency decay
    w = sev_raw * 0.5 ** (age / half_life_months)  # score weight: severity x recency

    labels = HDBSCAN(min_cluster_size=min_cluster_size,
                     min_samples=min_samples,
                     cluster_selection_method="leaf",
                     copy=True).fit_predict(XY)

    edges = events.edge[ok]
    types = events.type_idx[ok]
    months = events.month[ok]

    out = []
    for lab in sorted(set(labels)):
        if lab < 0:                                  # HDBSCAN noise label
            continue
        sel = labels == lab
        n = int(sel.sum())
        score = float(w[sel].sum() / MEAN_SEVERITY)
        seg_ids, seg_counts = np.unique(edges[sel], return_counts=True)
        order = np.argsort(-seg_counts)
        type_ids, type_counts = np.unique(types[sel], return_counts=True)

        # Trend: severity-weighted volume in the recent half of the window
        # against the earlier half.
        #
        # Two traps here, both of which make everything look like it is
        # getting worse. First, the halves must be the SAME length —
        # splitting at "now minus one half-life" compares 6 months against 3.
        # Second, the trend must use UNDECAYED weights: `w` already applies
        # exponential recency decay, so using it would count recency twice and
        # label almost every cluster "worsening" by construction.
        t_lo = float(months.min())
        cut = t_lo + (now - t_lo) / 2.0
        recent = float(sev_raw[sel][months[sel] >= cut].sum())
        earlier = float(sev_raw[sel][months[sel] < cut].sum())
        trend = (recent - earlier) / max(recent + earlier, 1e-9)

        radius = float(np.percentile(
            np.hypot(x[sel] - x[sel].mean(), y[sel] - y[sel].mean()), 90))
        if radius > max_radius_m:
            continue                      # a corridor, not a black spot
        out.append({
            "hotspot_id": int(lab),
            "lat": float(np.mean(lat[sel])),
            "lon": float(np.mean(lon[sel])),
            "radius_m": radius,
            "n_events": n,
            "weighted_score": round(score, 2),
            "trend": round(float(trend), 3),
            "trend_label": ("worsening" if trend > 0.15
                            else "improving" if trend < -0.15 else "stable"),
            "segments": [int(s) for s in seg_ids[order][:5]],
            "road_names": sorted({str(net["name"][s]) for s in seg_ids[order][:5]}),
            "dominant_event": EVENT_TYPES[int(type_ids[np.argmax(type_counts)])],
            "event_mix": {EVENT_TYPES[int(t)]: int(c)
                          for t, c in zip(type_ids, type_counts)},
        })

    out.sort(key=lambda h: -h["weighted_score"])
    for rank, h in enumerate(out, 1):
        h["rank"] = rank
    return out


def hotspots_on_route(hotspots: list[dict], route_edges: list[int],
                      top_k: int = 5) -> list[dict]:
    """Hotspots touching a given route, worst first."""
    rs = set(int(e) for e in route_edges)
    hits = [h for h in hotspots if rs & set(h["segments"])]
    return sorted(hits, key=lambda h: -h["weighted_score"])[:top_k]
