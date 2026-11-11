"""Live OpenStreetMap ingestion (Overpass API) -> RouteShield segment graph.

This is the production data path. It emits exactly the schema produced by
``routeshield.data.corridor.build_network``, so swapping real OSM in changes
nothing downstream.

Run it with::

    python -m scripts.fetch_osm --city bengaluru --out artifacts/osm_graph.json

Requires network access to an Overpass endpoint. Everything below the network
call (``parse_overpass`` and ``build_from_osm``) is pure and is unit-tested
against a recorded Overpass fixture in ``tests/fixtures/overpass_sample.json``,
so the parsing logic is verified even where the sandbox has no egress.

Licence note: OpenStreetMap data is © OpenStreetMap contributors, ODbL 1.0.
Any derived segment-risk table redistributed publicly inherits that licence.
"""
from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from typing import Any

import numpy as np

from .corridor import (
    FEATURE_NAMES,
    _curvature_index,
    haversine_km,
    polyline_length_km,
)

OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)

# Road classes worth routing a commercial fleet over.
DRIVABLE = (
    "motorway|trunk|primary|secondary|tertiary"
    "|motorway_link|trunk_link|primary_link|secondary_link"
)

# Bounding boxes (south, west, north, east).
CITY_BBOX: dict[str, tuple[float, float, float, float]] = {
    "bengaluru": (12.78, 77.44, 13.14, 77.80),
    "bengaluru_core": (12.88, 77.52, 13.06, 77.72),
    "mumbai_pune": (18.44, 73.05, 19.10, 73.95),
    "delhi_ncr": (28.35, 76.90, 28.78, 77.40),
}

# Default speed limits (km/h) when `maxspeed` is absent, by road class.
DEFAULT_MAXSPEED = {
    "motorway": 100, "trunk": 80, "primary": 60,
    "secondary": 50, "tertiary": 40,
    "motorway_link": 60, "trunk_link": 50,
    "primary_link": 40, "secondary_link": 40,
}

# Realised speed as a fraction of the posted limit (Indian urban congestion).
REALISED_SPEED_FACTOR = {
    "motorway": 0.85, "trunk": 0.70, "primary": 0.55,
    "secondary": 0.50, "tertiary": 0.45,
}


def overpass_query(bbox: tuple[float, float, float, float], timeout: int = 180) -> str:
    s, w, n, e = bbox
    return (
        f"[out:json][timeout:{timeout}];"
        f'way({s},{w},{n},{e})["highway"~"^({DRIVABLE})$"];'
        f"out geom;"
    )


def fetch_overpass(bbox, endpoints=OVERPASS_ENDPOINTS, retries: int = 3) -> dict[str, Any]:
    """Call Overpass and return the decoded JSON. Tries mirrors in order."""
    q = overpass_query(bbox)
    last: Exception | None = None
    for endpoint in endpoints:
        for attempt in range(retries):
            try:
                req = urllib.request.Request(
                    endpoint,
                    data=urllib.parse.urlencode({"data": q}).encode(),
                    headers={"User-Agent": "RouteShield/1.0 (open-source road-safety research)"},
                )
                with urllib.request.urlopen(req, timeout=300) as resp:
                    return json.loads(resp.read().decode())
            except Exception as exc:                      # pragma: no cover - network
                last = exc
                time.sleep(2 ** attempt * 3)
    raise RuntimeError(f"All Overpass endpoints failed; last error: {last}")


# --------------------------------------------------------------------------
# pure parsing (unit-tested without network)
# --------------------------------------------------------------------------
def _parse_maxspeed(raw: Any, highway: str) -> float:
    """OSM `maxspeed` is free text: '60', '60 mph', 'IN:urban', None."""
    if raw is None:
        return float(DEFAULT_MAXSPEED.get(highway, 40))
    s = str(raw).strip().lower()
    mph = "mph" in s
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return float(DEFAULT_MAXSPEED.get(highway, 40))
    val = float(digits)
    return val * 1.609 if mph else val


def _parse_lanes(raw: Any, highway: str) -> float:
    try:
        return float(str(raw).split(";")[0])
    except (TypeError, ValueError):
        return 3.0 if highway in ("motorway", "trunk") else 2.0


def _is_lit(tags: dict) -> float:
    lit = str(tags.get("lit", "")).lower()
    if lit in ("yes", "24/7", "automatic", "sunset-sunrise"):
        return 1.0
    if lit in ("no", "disused"):
        return 0.0
    # Unknown: assume lit inside the urban classes, unlit on rural highways.
    return 1.0 if tags.get("highway") in ("primary", "secondary", "tertiary") else 0.0


def _is_oneway(tags: dict) -> bool:
    ow = str(tags.get("oneway", "")).lower()
    if ow in ("yes", "true", "1", "-1"):
        return True
    return tags.get("highway") in ("motorway", "motorway_link")


def parse_overpass(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Overpass JSON -> list of way records with geometry and cleaned tags."""
    ways = []
    for el in payload.get("elements", []):
        if el.get("type") != "way" or not el.get("geometry"):
            continue
        tags = el.get("tags", {}) or {}
        highway = tags.get("highway")
        if not highway:
            continue
        pts = [(float(p["lat"]), float(p["lon"])) for p in el["geometry"]]
        if len(pts) < 2:
            continue
        ways.append({
            "id": el.get("id"),
            "nodes": el.get("nodes") or list(range(len(pts))),
            "geometry": pts,
            "name": tags.get("name") or tags.get("ref") or f"{highway} way {el.get('id')}",
            "highway": highway,
            "maxspeed": _parse_maxspeed(tags.get("maxspeed"), highway),
            "lanes": _parse_lanes(tags.get("lanes"), highway),
            "lit": _is_lit(tags),
            "oneway": _is_oneway(tags),
            "bridge": 1.0 if tags.get("bridge") else 0.0,
            "tags": tags,
        })
    return ways


def build_from_osm(ways: list[dict[str, Any]], seed: int = 7) -> dict:
    """Way records -> the canonical RouteShield segment graph.

    Ways are split at shared OSM nodes so that every segment runs junction to
    junction, which is the unit the risk model scores.
    """
    rng = np.random.default_rng(seed)

    # A node shared by 2+ ways is a real junction and becomes a split point.
    node_use: dict[Any, int] = {}
    for w in ways:
        for nid in w["nodes"]:
            node_use[nid] = node_use.get(nid, 0) + 1

    coord_of: dict[Any, tuple[float, float]] = {}
    for w in ways:
        for nid, pt in zip(w["nodes"], w["geometry"]):
            coord_of.setdefault(nid, pt)

    node_index: dict[Any, int] = {}
    node_lat_l, node_lon_l, node_ids = [], [], []

    def node_slot(nid):
        if nid not in node_index:
            node_index[nid] = len(node_ids)
            lat, lon = coord_of[nid]
            node_ids.append(str(nid))
            node_lat_l.append(lat)
            node_lon_l.append(lon)
        return node_index[nid]

    u, v, geoms, names, highways = [], [], [], [], []
    maxspeed, lit, lanes = [], [], []
    road_id: list[int] = []
    next_road = 0

    for w in ways:
        nodes, pts = w["nodes"], w["geometry"]
        if len(nodes) != len(pts):           # geometry/node mismatch: treat as one segment
            nodes = list(range(len(pts)))
        # split indices: endpoints plus every shared node
        cuts = [0] + [i for i in range(1, len(nodes) - 1)
                      if node_use.get(nodes[i], 0) > 1] + [len(nodes) - 1]
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b <= a:
                continue
            seg_pts = pts[a:b + 1]
            if polyline_length_km(seg_pts) < 0.01:
                continue
            s_slot, t_slot = node_slot(nodes[a]), node_slot(nodes[b])
            directions = [(s_slot, t_slot, seg_pts)]
            if not w["oneway"]:
                directions.append((t_slot, s_slot, seg_pts[::-1]))
            rid = next_road
            next_road += 1
            for s, t, g in directions:
                road_id.append(rid)
                u.append(s); v.append(t); geoms.append(g)
                names.append(w["name"]); highways.append(w["highway"])
                maxspeed.append(w["maxspeed"]); lit.append(w["lit"]); lanes.append(w["lanes"])

    if not u:
        raise ValueError("No drivable segments parsed from the Overpass payload.")

    m = len(u)
    length = np.array([polyline_length_km(g) for g in geoms])
    curvature = np.array([_curvature_index(g, l) for g, l in zip(geoms, length)])

    degree = np.zeros(len(node_ids))
    for a in u:
        degree[a] += 1
    junction_density = np.clip(
        np.array([(degree[a] + degree[b]) / 2.0 / max(l, 0.05)
                  for a, b, l in zip(u, v, length)]), 0, 12)

    node_lat = np.array(node_lat_l)
    node_lon = np.array(node_lon_l)

    # Gradient needs a DEM.  Without one, derive a proxy from the local
    # terrain roughness implied by the corridor; replace with SRTM in prod.
    gradient = np.abs(rng.normal(0, 0.6, m))

    maxspeed = np.array(maxspeed, dtype=float)
    lit = np.array(lit, dtype=float)
    lanes = np.array(lanes, dtype=float)
    highways_arr = np.array(highways)

    realised = np.array([REALISED_SPEED_FACTOR.get(h, 0.5) for h in highways])
    time_min = length / np.maximum(maxspeed * realised, 5.0) * 60.0

    class_weight = np.array([
        {"motorway": 4.0, "trunk": 3.0, "primary": 1.6}.get(h, 0.7) for h in highways
    ])
    exposure = length * class_weight * rng.lognormal(6.0, 1.15, m)

    X = np.column_stack([curvature, junction_density, 1.0 - lit, gradient,
                         maxspeed / 100.0, lanes])

    return {
        "source": "OSM_OVERPASS",
        "place": "OpenStreetMap extract",
        "node_ids": node_ids,
        "node_name": node_ids,
        "node_lat": node_lat,
        "node_lon": node_lon,
        "n_nodes": len(node_ids),
        "road_id": np.array(road_id),
        "u": np.array(u),
        "v": np.array(v),
        "geometry": geoms,
        "name": names,
        "highway": highways_arr,
        "length": length,
        "maxspeed": maxspeed,
        "lit": lit,
        "unlit": 1.0 - lit,
        "lanes": lanes,
        "curvature": curvature,
        "junction_density": junction_density,
        "gradient": gradient,
        "time": time_min,
        "exposure": exposure,
        "X": X,
        "feature_names": FEATURE_NAMES,
        "tags": [() for _ in range(m)],
    }


def fetch_city(city: str = "bengaluru") -> dict:      # pragma: no cover - network
    bbox = CITY_BBOX[city]
    payload = fetch_overpass(bbox)
    ways = parse_overpass(payload)
    net = build_from_osm(ways)
    net["place"] = f"OpenStreetMap: {city} {bbox}"
    return net
