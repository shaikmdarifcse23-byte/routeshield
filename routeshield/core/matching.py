"""GPS -> road-segment map matching.

Telematics events arrive as raw (lat, lon) with 10-30 m of GPS error. Before
any risk model can be fitted they must be snapped to the segment the vehicle
was actually on. A naive nearest-vertex match smears events across parallel
carriageways and service roads, which is exactly how hotspot maps go wrong.

Implementation: a uniform spatial hash over polyline vertices gives candidate
segments in O(1); each candidate is scored by true perpendicular
point-to-polyline distance, with a gate that rejects matches beyond
``max_dist_m`` instead of forcing a wrong one.
"""
from __future__ import annotations

import math

import numpy as np

EARTH_R_M = 6_371_008.8


def _local_xy(lat, lon, lat0, lon0):
    """Equirectangular projection to metres — accurate well under 1 m over a city."""
    k = math.cos(math.radians(lat0))
    x = np.radians(np.asarray(lon) - lon0) * EARTH_R_M * k
    y = np.radians(np.asarray(lat) - lat0) * EARTH_R_M
    return x, y


def _point_seg_dist2(px, py, ax, ay, bx, by):
    """Squared distance from point to line segment, vectorised over segments."""
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    denom = vx * vx + vy * vy
    t = np.where(denom > 0, (wx * vx + wy * vy) / np.maximum(denom, 1e-12), 0.0)
    t = np.clip(t, 0.0, 1.0)
    cx, cy = ax + t * vx, ay + t * vy
    return (px - cx) ** 2 + (py - cy) ** 2


class MapMatcher:
    """Spatial index over segment polylines, built once per network."""

    def __init__(self, net: dict, cell_m: float = 120.0):
        self.net = net
        self.cell = cell_m
        self.lat0 = float(np.mean(net["node_lat"]))
        self.lon0 = float(np.mean(net["node_lon"]))

        # Flatten every polyline into (edge_id, ax, ay, bx, by) sub-segments.
        eids, axs, ays, bxs, bys = [], [], [], [], []
        for e, pts in enumerate(net["geometry"]):
            lats = np.array([p[0] for p in pts])
            lons = np.array([p[1] for p in pts])
            x, y = _local_xy(lats, lons, self.lat0, self.lon0)
            for i in range(len(pts) - 1):
                eids.append(e)
                axs.append(x[i]); ays.append(y[i])
                bxs.append(x[i + 1]); bys.append(y[i + 1])
        self.eid = np.array(eids)
        self.ax = np.array(axs); self.ay = np.array(ays)
        self.bx = np.array(bxs); self.by = np.array(bys)

        # Spatial hash: every cell touched by a sub-segment's bounding box.
        self.grid: dict[tuple[int, int], list[int]] = {}
        for k in range(len(self.eid)):
            x0, x1 = sorted((self.ax[k], self.bx[k]))
            y0, y1 = sorted((self.ay[k], self.by[k]))
            for cx in range(int(x0 // cell_m), int(x1 // cell_m) + 1):
                for cy in range(int(y0 // cell_m), int(y1 // cell_m) + 1):
                    self.grid.setdefault((cx, cy), []).append(k)

    def match(self, lats, lons, max_dist_m: float = 60.0, ring: int = 1,
              margin_m: float = 18.0, road_id=None):
        """Snap points to segments.

        Returns ``(edge_idx, dist_m)``; ``edge_idx == -1`` marks a point that
        was rejected and should be dropped, not force-matched.

        Two gates, and the second one matters more than it looks:

        * **Distance** — beyond ``max_dist_m`` there is no plausible road.
        * **Ambiguity** — if the best candidate on a *different* road is within
          ``margin_m`` of the best, the point is thrown away rather than
          guessed. Near a junction two roads are almost equidistant, and
          guessing there is how a quiet side road silently inherits a trunk
          road's harsh-braking events. Measured on this network, forcing those
          matches misattributes ~4.6% of events, which is enough to make a
          handful of low-traffic roads look like the worst in the city.
          Dropping them costs a little data and buys a map you can trust.
        """
        px, py = _local_xy(np.asarray(lats), np.asarray(lons), self.lat0, self.lon0)
        n = len(px)
        out = np.full(n, -1, dtype=int)
        dist = np.full(n, np.inf)
        rid = None if road_id is None else np.asarray(road_id)

        for i in range(n):
            cx, cy = int(px[i] // self.cell), int(py[i] // self.cell)
            cand: list[int] = []
            for dx in range(-ring, ring + 1):
                for dy in range(-ring, ring + 1):
                    cand.extend(self.grid.get((cx + dx, cy + dy), ()))
            if not cand:
                continue
            c = np.fromiter(set(cand), dtype=int)
            d2 = _point_seg_dist2(px[i], py[i], self.ax[c], self.ay[c],
                                  self.bx[c], self.by[c])
            order = np.argsort(d2)
            j = int(order[0])
            d = math.sqrt(float(d2[j]))
            if d > max_dist_m:
                continue
            best_edge = int(self.eid[c[j]])

            if margin_m > 0 and rid is not None:
                best_road = rid[best_edge]
                runner = None
                for k in order[1:]:
                    e2 = int(self.eid[c[int(k)]])
                    if rid[e2] != best_road:
                        runner = math.sqrt(float(d2[int(k)]))
                        break
                if runner is not None and (runner - d) < margin_m:
                    continue                      # ambiguous: refuse to guess

            out[i] = best_edge
            dist[i] = d
        return out, dist


def match_events(net: dict, events, max_dist_m: float = 60.0,
                 margin_m: float = 18.0):
    """Map-match an :class:`EventTable` in place; returns match statistics."""
    matcher = MapMatcher(net)
    edge, dist = matcher.match(events.lat, events.lon, max_dist_m,
                               margin_m=margin_m, road_id=net.get("road_id"))
    events.edge = edge
    ok = edge >= 0
    stats_extra = {}
    if getattr(events, "true_edge", None) is not None and net.get("road_id") is not None:
        rid = np.asarray(net["road_id"])
        if ok.any():
            stats_extra["road_match_accuracy"] = float(
                (rid[edge[ok]] == rid[events.true_edge[ok]]).mean())
    return {**stats_extra,
        "events_in": int(len(edge)),
        "matched": int(ok.sum()),
        "match_rate": float(ok.mean()),
        "median_snap_dist_m": float(np.median(dist[ok])) if ok.any() else None,
        "p95_snap_dist_m": float(np.percentile(dist[ok], 95)) if ok.any() else None,
        "max_dist_gate_m": max_dist_m,
        "ambiguity_margin_m": margin_m,
    }
