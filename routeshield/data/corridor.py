"""Bengaluru ORR + Hosur Road corridor network.

PROVENANCE — read this before quoting any number from this module.

* Junction **node coordinates** are real-world locations of named Bengaluru
  junctions (WGS84), accurate to roughly a hundred metres.
* Inter-junction **polyline geometry** is interpolated (great-circle with a
  configurable bow for ring roads), *not* true OpenStreetMap way geometry.
* Road attributes (class, speed limit, lighting) are assigned from the
  corridor's real-world character, not read from OSM tags.

This bundled network exists so the whole system runs offline, deterministically,
in about a second, with zero network access. It is explicitly labelled
``GEO_ANCHORED_SYNTHETIC`` everywhere it surfaces.

To replace it with genuine OpenStreetMap geometry and tags::

    python -m scripts.fetch_osm --city bengaluru --out artifacts/osm_graph.json

``routeshield.data.osm`` implements that path against the live Overpass API and
emits the identical schema, so every downstream module is unchanged.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

SOURCE_TAG = "GEO_ANCHORED_SYNTHETIC"
EARTH_R_KM = 6371.0088

# --------------------------------------------------------------------------
# Real Bengaluru junctions.  (lat, lon, display name)
# --------------------------------------------------------------------------
JUNCTIONS: dict[str, tuple[float, float, str]] = {
    # --- Outer Ring Road, clockwise from Hebbal ---
    "hebbal":        (13.0358, 77.5970, "Hebbal Flyover"),
    "nagawara":      (13.0430, 77.6200, "Nagawara Junction"),
    "hennur":        (13.0310, 77.6420, "Hennur Cross"),
    "tin_factory":   (13.0030, 77.6600, "Tin Factory"),
    "krpuram":       (13.0070, 77.6960, "K R Puram"),
    "mahadevapura":  (12.9890, 77.6950, "Mahadevapura"),
    "marathahalli":  (12.9591, 77.6974, "Marathahalli Bridge"),
    "bellandur":     (12.9300, 77.6790, "Bellandur / Iblur"),
    "agara":         (12.9220, 77.6450, "Agara Junction"),
    "silk_board":    (12.9166, 77.6226, "Central Silk Board"),
    "btm":           (12.9160, 77.6050, "BTM Layout"),
    "bannerghatta":  (12.9080, 77.5950, "Bannerghatta Road ORR"),
    "jp_nagar":      (12.9070, 77.5750, "J P Nagar"),
    "kanakapura":    (12.9050, 77.5540, "Kanakapura Road ORR"),
    "nayandahalli":  (12.9400, 77.5250, "Nayandahalli / Mysore Rd"),
    "kengeri_orr":   (12.9180, 77.5080, "Kengeri ORR"),
    "sumanahalli":   (12.9880, 77.5200, "Sumanahalli"),
    "peenya":        (13.0280, 77.5200, "Peenya Industrial Area"),
    "yeshwanthpur":  (13.0220, 77.5500, "Yeshwanthpur"),
    # --- radial corridors ---
    "bommanahalli":  (12.9010, 77.6180, "Bommanahalli"),
    "singasandra":   (12.8790, 77.6350, "Singasandra"),
    "electronic_city": (12.8452, 77.6602, "Electronic City"),
    "attibele":      (12.7830, 77.7730, "Attibele (NH44)"),
    "kaikondrahalli": (12.9100, 77.6800, "Kaikondrahalli"),
    "sarjapur":      (12.8600, 77.7800, "Sarjapur"),
    "whitefield":    (12.9698, 77.7500, "Whitefield"),
    "hoskote":       (13.0707, 77.7980, "Hoskote (NH75)"),
    "devanahalli":   (13.1989, 77.7068, "Devanahalli / KIA"),
    "nelamangala":   (13.0990, 77.4420, "Nelamangala (NH48)"),
    "bidadi":        (12.7980, 77.3860, "Bidadi (Mysore Rd)"),
    "anekal":        (12.7110, 77.6960, "Anekal"),
    # --- inner city ---
    "majestic":      (12.9767, 77.5713, "Majestic / KSR"),
    "mg_road":       (12.9750, 77.6060, "M G Road"),
    "dairy_circle":  (12.9330, 77.5980, "Dairy Circle"),
    "koramangala":   (12.9350, 77.6240, "Koramangala"),
    "indiranagar":   (12.9719, 77.6412, "Indiranagar"),
    "domlur":        (12.9610, 77.6380, "Domlur"),
    "hosur_rd_bda":  (12.9430, 77.6080, "Hosur Rd / Lalbagh"),
}

# Centre used to bow Outer Ring Road segments outward into a true ring.
_ORR_CENTRE = (12.9690, 77.6050)


@dataclass(frozen=True)
class Corridor:
    """One named road corridor: an ordered walk through junctions."""

    name: str
    nodes: tuple[str, ...]
    highway: str                 # motorway | trunk | primary | secondary
    maxspeed: int                # km/h
    lit: int                     # 1 = street lighting present
    lanes: int
    ring: bool = False           # bow the geometry outward (ORR)
    gradient: float = 0.0        # mean |gradient| %, ghat sections are higher
    curviness: float = 0.15      # corridor-level curvature character, 0..1
    tags: tuple[str, ...] = field(default=())


CORRIDORS: tuple[Corridor, ...] = (
    # ---------------- Outer Ring Road (both arcs) ----------------
    Corridor("Outer Ring Road (East)",
             ("hebbal", "nagawara", "hennur", "tin_factory", "krpuram",
              "mahadevapura", "marathahalli", "bellandur", "agara", "silk_board"),
             "trunk", 60, 1, 4, ring=True, curviness=0.22, tags=("orr", "high_volume")),
    Corridor("Outer Ring Road (South-West)",
             ("silk_board", "btm", "bannerghatta", "jp_nagar", "kanakapura",
              "kengeri_orr", "nayandahalli"),
             "trunk", 60, 1, 3, ring=True, curviness=0.20, tags=("orr",)),
    Corridor("Outer Ring Road (West-North)",
             ("nayandahalli", "sumanahalli", "peenya", "yeshwanthpur", "hebbal"),
             "trunk", 60, 1, 3, ring=True, curviness=0.20, tags=("orr",)),
    # ---------------- National highways / radials ----------------
    Corridor("Hosur Road (NH44)",
             ("silk_board", "bommanahalli", "singasandra", "electronic_city", "attibele"),
             "motorway", 80, 1, 3, curviness=0.06, tags=("nh44", "freight", "elevated")),
    Corridor("NH44 South (Attibele–Anekal link)",
             ("attibele", "anekal"),
             "primary", 60, 0, 2, gradient=1.8, curviness=0.46, tags=("nh44", "unlit")),
    Corridor("Sarjapur Road",
             ("agara", "kaikondrahalli", "sarjapur"),
             "primary", 50, 0, 2, curviness=0.38, tags=("tech_corridor", "unlit_stretch")),
    Corridor("Varthur / Whitefield Road",
             ("marathahalli", "whitefield"),
             "primary", 50, 1, 2, curviness=0.33, tags=("tech_corridor",)),
    Corridor("Old Madras Road (NH75)",
             ("krpuram", "hoskote"),
             "trunk", 80, 0, 3, curviness=0.14, tags=("nh75", "freight")),
    Corridor("Bellary Road (NH44 North / Airport)",
             ("hebbal", "devanahalli"),
             "motorway", 100, 1, 3, curviness=0.05, tags=("nh44", "airport", "high_speed")),
    Corridor("Tumkur Road (NH48)",
             ("peenya", "nelamangala"),
             "motorway", 90, 1, 3, curviness=0.07, tags=("nh48", "freight")),
    Corridor("Mysore Road (NH275)",
             ("nayandahalli", "bidadi"),
             "trunk", 80, 0, 2, gradient=1.2, curviness=0.18, tags=("nh275", "freight")),
    Corridor("Sarjapur–Attibele Link",
             ("sarjapur", "attibele"),
             "secondary", 50, 0, 2, gradient=2.4, curviness=0.62, tags=("unlit", "rural")),
    # ---------------- Inner-city arterials ----------------
    Corridor("Hosur Road (inner)",
             ("silk_board", "koramangala", "hosur_rd_bda", "dairy_circle"),
             "primary", 50, 1, 3, curviness=0.30, tags=("urban",)),
    Corridor("Bannerghatta Road (inner)",
             ("bannerghatta", "dairy_circle"),
             "primary", 50, 1, 2, curviness=0.32, tags=("urban",)),
    Corridor("Inner Ring Road",
             ("dairy_circle", "koramangala", "domlur", "indiranagar", "mg_road"),
             "primary", 50, 1, 2, curviness=0.32, tags=("urban",)),
    Corridor("Old Airport Road",
             ("domlur", "marathahalli"),
             "primary", 50, 1, 3, curviness=0.28, tags=("urban", "high_volume")),
    Corridor("M G Road – Majestic",
             ("mg_road", "majestic"),
             "primary", 40, 1, 2, curviness=0.40, tags=("urban", "cbd")),
    Corridor("Majestic – Yeshwanthpur",
             ("majestic", "yeshwanthpur"),
             "primary", 50, 1, 2, curviness=0.32, tags=("urban",)),
    Corridor("Indiranagar – Tin Factory",
             ("indiranagar", "tin_factory"),
             "secondary", 40, 1, 2, curviness=0.42, tags=("urban",)),
    Corridor("Koramangala – Agara",
             ("koramangala", "agara"),
             "secondary", 40, 1, 2, curviness=0.42, tags=("urban",)),
    Corridor("Bommanahalli – Bannerghatta link",
             ("bommanahalli", "bannerghatta"),
             "secondary", 40, 0, 2, curviness=0.44, tags=("urban", "unlit_stretch")),
    Corridor("Electronic City – Anekal",
             ("electronic_city", "anekal"),
             "secondary", 50, 0, 2, gradient=1.6, curviness=0.55, tags=("rural", "unlit")),
    Corridor("Kaikondrahalli – Bellandur",
             ("kaikondrahalli", "bellandur"),
             "secondary", 40, 1, 2, curviness=0.36, tags=("tech_corridor",)),
    Corridor("Whitefield – Hoskote",
             ("whitefield", "hoskote"),
             "secondary", 60, 0, 2, gradient=1.4, curviness=0.50, tags=("rural", "unlit")),
    # ---------------- cross-links (real roads that create route choice) ----
    Corridor("Hennur-Whitefield Link",
             ("hennur", "whitefield"),
             "secondary", 50, 0, 2, curviness=0.48, tags=("rural", "unlit")),
    Corridor("NH648 Hoskote-Devanahalli",
             ("hoskote", "devanahalli"),
             "trunk", 80, 0, 2, curviness=0.16, tags=("nh648", "freight")),
    Corridor("NH648 Devanahalli-Nelamangala",
             ("devanahalli", "nelamangala"),
             "trunk", 80, 0, 2, curviness=0.18, tags=("nh648", "freight")),
    Corridor("Electronic City-Sarjapur Link",
             ("electronic_city", "kaikondrahalli"),
             "secondary", 50, 0, 2, gradient=1.1, curviness=0.52, tags=("rural", "unlit")),
    Corridor("Kanakapura-JP Nagar Link",
             ("kanakapura", "jp_nagar"),
             "secondary", 40, 1, 2, curviness=0.40, tags=("urban",)),
    Corridor("Majestic-Hebbal (Bellary Rd inner)",
             ("majestic", "hebbal"),
             "primary", 50, 1, 3, curviness=0.26, tags=("urban", "high_volume")),
    Corridor("Indiranagar-Koramangala",
             ("indiranagar", "koramangala"),
             "secondary", 40, 1, 2, curviness=0.38, tags=("urban",)),
    Corridor("BTM-Koramangala",
             ("btm", "koramangala"),
             "secondary", 40, 1, 2, curviness=0.40, tags=("urban",)),
    Corridor("Mahadevapura-Whitefield",
             ("mahadevapura", "whitefield"),
             "primary", 50, 1, 2, curviness=0.30, tags=("tech_corridor",)),
    Corridor("Nagawara-Indiranagar",
             ("nagawara", "indiranagar"),
             "secondary", 40, 1, 2, curviness=0.42, tags=("urban",)),
    Corridor("Anekal-Sarjapur",
             ("anekal", "sarjapur"),
             "secondary", 50, 0, 2, gradient=1.9, curviness=0.58, tags=("rural", "unlit")),
    Corridor("Bidadi-Kengeri ORR",
             ("bidadi", "kengeri_orr"),
             "trunk", 70, 0, 2, curviness=0.20, tags=("nh275",)),
    Corridor("Sumanahalli-Majestic",
             ("sumanahalli", "majestic"),
             "primary", 50, 1, 2, curviness=0.34, tags=("urban",)),
    Corridor("Dairy Circle-Majestic",
             ("dairy_circle", "majestic"),
             "primary", 40, 1, 2, curviness=0.38, tags=("urban", "cbd")),
    Corridor("Tin Factory-Mahadevapura",
             ("tin_factory", "mahadevapura"),
             "secondary", 40, 1, 2, curviness=0.36, tags=("urban",)),
)


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_R_KM * math.asin(math.sqrt(h))


def polyline_length_km(pts: Iterable[tuple[float, float]]) -> float:
    pts = list(pts)
    return float(sum(haversine_km(pts[i], pts[i + 1]) for i in range(len(pts) - 1)))


def _interpolate(a, b, n, bow_centre=None, bow_frac=0.0, meander=0.0, seed=0):
    """n+1 points from a to b.

    ``bow_frac`` pushes midpoints radially outward from ``bow_centre`` as a
    fraction of the leg length, so a ring road curves like a ring instead of
    bulging into a flower on short legs. ``meander`` adds a gentle lateral
    waver, because a 30 km highway leg drawn as a perfectly straight line
    reads as a spider web, not a road.
    """
    out = []
    span = math.hypot(b[0] - a[0], b[1] - a[1]) or 1e-9
    bow = bow_frac * span
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * math.pi)
    waves = rng.uniform(1.2, 2.4)
    # unit normal to the leg, used for the meander offset
    nx, ny = -(b[1] - a[1]) / span, (b[0] - a[0]) / span
    for i in range(n + 1):
        t = i / n
        lat = a[0] + (b[0] - a[0]) * t
        lon = a[1] + (b[1] - a[1]) * t
        if bow_centre is not None and bow > 0:
            w = bow * 4 * t * (1 - t)          # zero at the ends, max mid-leg
            dlat, dlon = lat - bow_centre[0], lon - bow_centre[1]
            norm = math.hypot(dlat, dlon) or 1e-9
            lat += w * dlat / norm
            lon += w * dlon / norm
        if meander > 0:
            k = meander * span * math.sin(waves * math.pi * t + phase) * (4 * t * (1 - t))
            lat += k * nx
            lon += k * ny
        out.append((round(lat, 6), round(lon, 6)))
    return out


def _sinuosity_index(pts: list[tuple[float, float]], length_km: float) -> float:
    """Excess of path length over the straight chord, squashed to 0..1.

    Robust where a heading-change sum is not: it does not blow up on short
    polylines with a few jittery vertices.
    """
    if len(pts) < 2 or length_km <= 0:
        return 0.0
    chord = haversine_km(pts[0], pts[-1])
    if chord <= 1e-9:
        return 1.0
    return float(min(1.0, max(0.0, (length_km / chord - 1.0) * 8.0)))


def _curvature_index(pts: list[tuple[float, float]], length_km: float) -> float:
    """Total absolute heading change per km, squashed into 0..1.

    A straight expressway is ~0; a tight ghat hairpin approaches 1.
    """
    if len(pts) < 3 or length_km <= 0:
        return 0.0
    total = 0.0
    for i in range(1, len(pts) - 1):
        v1 = (pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        v2 = (pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        a1, a2 = math.atan2(v1[0], v1[1]), math.atan2(v2[0], v2[1])
        d = abs(math.degrees(a2 - a1))
        total += min(d, 360 - d)
    deg_per_km = total / max(length_km, 0.2)
    return float(min(1.0, deg_per_km / 60.0))


# --------------------------------------------------------------------------
# network construction
# --------------------------------------------------------------------------
def _split_polyline(pts: list[tuple[float, float]], target_km: float
                    ) -> list[list[tuple[float, float]]]:
    """Cut a polyline into consecutive chunks of roughly ``target_km``.

    Real OSM ways are split at junctions; long inter-junction stretches are
    still split further so that a 30 km highway leg is not scored as one
    undifferentiated block. Risk varies within a corridor and the model has to
    be able to say where.
    """
    total = polyline_length_km(pts)
    n_chunks = max(1, int(round(total / target_km)))
    if n_chunks == 1:
        return [pts]
    step = total / n_chunks

    chunks: list[list[tuple[float, float]]] = []
    cur = [pts[0]]
    acc = 0.0
    target = step
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        seg = haversine_km(a, b)
        if seg <= 0:
            continue
        pos = 0.0
        while acc + (seg - pos) >= target - 1e-12 and len(chunks) < n_chunks - 1:
            need = target - acc - pos
            f = (pos + need) / seg
            cut = (round(a[0] + (b[0] - a[0]) * f, 6),
                   round(a[1] + (b[1] - a[1]) * f, 6))
            cur.append(cut)
            chunks.append(cur)
            cur = [cut]
            pos += need
            acc = 0.0
            target = step
        acc += seg - pos
        cur.append(b)
    if len(cur) >= 2:
        chunks.append(cur)
    return chunks


def build_network(seed: int = 7, target_segment_km: float = 1.5) -> dict:
    """Build the directed segment graph. Returns the canonical network dict
    consumed by every other module (same schema as ``data.osm``)."""
    rng = np.random.default_rng(seed)

    node_ids: list[str] = []
    node_lat_l: list[float] = []
    node_lon_l: list[float] = []
    node_name: list[str] = []
    node_is_junction: list[int] = []
    node_index: dict[str, int] = {}

    for key, (lat, lon, name) in JUNCTIONS.items():
        node_index[key] = len(node_ids)
        node_ids.append(key)
        node_lat_l.append(lat)
        node_lon_l.append(lon)
        node_name.append(name)
        node_is_junction.append(1)

    def add_interior_node(lat: float, lon: float, label: str) -> int:
        key = f"pt_{len(node_ids)}"
        node_index[key] = len(node_ids)
        node_ids.append(key)
        node_lat_l.append(lat)
        node_lon_l.append(lon)
        node_name.append(label)
        node_is_junction.append(0)
        return node_index[key]

    u, v, geoms, names, highways = [], [], [], [], []
    maxspeed, lit, lanes, gradient, corr_tags, curviness = [], [], [], [], [], []
    road_id: list[int] = []
    next_road = 0

    for corr in CORRIDORS:
        for i in range(len(corr.nodes) - 1):
            a_key, b_key = corr.nodes[i], corr.nodes[i + 1]
            a = (JUNCTIONS[a_key][0], JUNCTIONS[a_key][1])
            b = (JUNCTIONS[b_key][0], JUNCTIONS[b_key][1])
            straight = haversine_km(a, b)
            n_pts = max(6, min(60, int(straight * 4)))
            centre, bow = (_ORR_CENTRE, 0.085) if corr.ring else (None, 0.0)
            # longer legs waver more; short urban links stay nearly straight
            meander = 0.0 if corr.ring else min(0.055, 0.004 * straight)
            # zlib.crc32, not hash(): Python randomises string hashing per
            # process, so hash() here would make the network non-reproducible
            # between runs — and every published metric with it.
            leg_seed = zlib.crc32(f"{corr.name}|{a_key}|{b_key}".encode())
            pts = _interpolate(a, b, n_pts, centre, bow, meander, seed=leg_seed)

            chunks = _split_polyline(pts, target_segment_km)
            # chain: a -> interior nodes -> b
            slots = [node_index[a_key]]
            for c in chunks[:-1]:
                end = c[-1]
                slots.append(add_interior_node(
                    end[0], end[1], f"{corr.name} km {polyline_length_km(pts[:1] + c):.1f}"))
            slots.append(node_index[b_key])

            for k, chunk in enumerate(chunks):
                s_slot, t_slot = slots[k], slots[k + 1]
                rid = next_road
                next_road += 1
                for s, t, g in ((s_slot, t_slot, chunk), (t_slot, s_slot, chunk[::-1])):
                    road_id.append(rid)
                    u.append(s)
                    v.append(t)
                    geoms.append(g)
                    names.append(corr.name)
                    highways.append(corr.highway)
                    maxspeed.append(corr.maxspeed)
                    lit.append(corr.lit)
                    lanes.append(corr.lanes)
                    gradient.append(corr.gradient)
                    curviness.append(corr.curviness)
                    corr_tags.append(corr.tags)

    node_lat = np.array(node_lat_l)
    node_lon = np.array(node_lon_l)
    m = len(u)
    length = np.array([polyline_length_km(g) for g in geoms])
    # Curvature combines the corridor's real-world character with the
    # sinuosity measured from the polyline. With a true OSM graph loaded the
    # measured term dominates; here the corridor term carries most of it,
    # which is stated plainly in the module docstring.
    sinuosity = np.array([_sinuosity_index(g, l) for g, l in zip(geoms, length)])
    curvature = np.clip(np.array(curviness) + 0.6 * sinuosity
                        + rng.normal(0, 0.03, len(geoms)), 0.0, 1.0)

    # Junction density: how many road ends meet at this segment's endpoints,
    # per km.  A real structural property of the graph — interior split points
    # score ~2, a six-arm junction like Silk Board scores much higher.
    degree = np.zeros(len(node_ids))
    for a, b in zip(u, v):
        degree[a] += 1
        degree[b] += 1
    junction_density = np.clip(
        np.array([(degree[a] + degree[b]) / 2.0 / max(l, 0.3)
                  for a, b, l in zip(u, v, length)]), 0, 12)

    # Gradient: corridor base value plus a small deterministic per-segment jitter.
    gradient = np.array(gradient) + np.abs(rng.normal(0, 0.35, m))

    maxspeed = np.array(maxspeed, dtype=float)
    lit = np.array(lit, dtype=float)
    lanes = np.array(lanes, dtype=float)
    unlit = 1.0 - lit

    # Free-flow travel time (min).  Urban arterials are congested, so apply a
    # class-dependent realisation factor rather than using the posted limit.
    realised = np.where(
        np.isin(highways, ["motorway"]), 0.85,
        np.where(np.isin(highways, ["trunk"]), 0.70,
                 np.where(np.isin(highways, ["primary"]), 0.55, 0.50)),
    )
    time_min = length / (maxspeed * realised) * 60.0

    # Fleet exposure (vehicle-km per month).  Deliberately heavy-tailed: a
    # real fleet concentrates on a few lanes, so most segments are data-sparse.
    # That sparse regime is exactly what the Bayesian shrinkage is built for,
    # and a demo on uniformly data-rich roads would prove nothing.
    class_weight = np.where(
        np.isin(highways, ["motorway"]), 4.0,
        np.where(np.isin(highways, ["trunk"]), 3.0,
                 np.where(np.isin(highways, ["primary"]), 1.6, 0.7)),
    )
    exposure = length * class_weight * rng.lognormal(2.3, 2.00, m)

    X = np.column_stack([curvature, junction_density, unlit, gradient,
                         maxspeed / 100.0, lanes])

    return {
        "source": SOURCE_TAG,
        "place": "Bengaluru ORR + Hosur Road corridor",
        "node_ids": node_ids,
        "node_name": node_name,
        "node_lat": node_lat,
        "node_lon": node_lon,
        "node_is_junction": np.array(node_is_junction),
        "road_id": np.array(road_id),
        "n_nodes": len(node_ids),
        "u": np.array(u),
        "v": np.array(v),
        "geometry": geoms,
        "name": names,
        "highway": np.array(highways),
        "length": length,
        "maxspeed": maxspeed,
        "lit": lit,
        "unlit": unlit,
        "lanes": lanes,
        "curvature": curvature,
        "junction_density": junction_density,
        "gradient": gradient,
        "time": time_min,
        "exposure": exposure,
        "X": X,
        "feature_names": FEATURE_NAMES,
        "tags": corr_tags,
    }


FEATURE_NAMES = [
    "curvature",
    "junction_density",
    "unlit",
    "gradient_pct",
    "maxspeed_100",
    "lanes",
]

FEATURE_LABELS = {
    "curvature": "Road curvature",
    "junction_density": "Junction density",
    "unlit": "No street lighting",
    "gradient_pct": "Gradient / ghat",
    "maxspeed_100": "Speed limit",
    "lanes": "Lane count",
}
