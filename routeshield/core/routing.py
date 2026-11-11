"""Route scoring and safer-alternative search.

Scoring a route is a lookup and a sum — segment risk is precomputed in the
nightly batch. The interesting part is *choice*: given a planned trip, which
other routes exist, what do they cost in time, and is the trade worth it?

Approach: sweep a penalty parameter ``lambda`` over cost = time + lambda x risk
and run Dijkstra at each setting. Sweeping lambda from 0 to large traces out
exactly the lower-left boundary of the achievable (time, risk) set, so the
distinct paths it returns are the Pareto-optimal ones. That is cheaper and
more complete than k-shortest-paths, which tends to return k near-identical
detours around the same corner.

The dispatcher then sees the frontier, not a single opaque recommendation.
"""
from __future__ import annotations

import heapq

import numpy as np

LAMBDA_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0)


def build_adj(net: dict) -> list[list[tuple[int, int]]]:
    adj: list[list[tuple[int, int]]] = [[] for _ in range(net["n_nodes"])]
    for e, (a, b) in enumerate(zip(net["u"], net["v"])):
        adj[int(a)].append((int(b), e))
    return adj


def dijkstra(adj, cost: np.ndarray, s: int, t: int) -> list[int] | None:
    """Shortest path by edge cost; returns the edge sequence, or None."""
    n = len(adj)
    dist = np.full(n, np.inf)
    prev: list[tuple[int, int] | None] = [None] * n
    dist[s] = 0.0
    pq = [(0.0, s)]
    while pq:
        d, x = heapq.heappop(pq)
        if d > dist[x]:
            continue
        if x == t:
            break
        for y, e in adj[x]:
            nd = d + cost[e]
            if nd < dist[y]:
                dist[y] = nd
                prev[y] = (x, e)
                heapq.heappush(pq, (nd, y))
    if not np.isfinite(dist[t]):
        return None
    path, x = [], t
    while prev[x] is not None:
        x, e = prev[x]
        path.append(e)
    return path[::-1]


def candidate_routes(net: dict, adj, edge_risk: np.ndarray, s: int, t: int,
                     lambdas=LAMBDA_GRID) -> list[list[int]]:
    """Distinct paths traced by sweeping the risk penalty."""
    mean_risk = max(float(np.mean(edge_risk)), 1e-12)
    scale = float(np.mean(net["time"])) / mean_risk      # put risk in time units
    seen: set[tuple[int, ...]] = set()
    out: list[list[int]] = []
    for lam in lambdas:
        r = dijkstra(adj, net["time"] + lam * scale * edge_risk, s, t)
        if r is None:
            continue
        key = tuple(r)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def pareto_frontier(routes: list[list[int]], net: dict,
                    edge_risk: np.ndarray) -> list[dict]:
    """Keep only routes not dominated on both time and risk."""
    rows = [{"edges": r,
             "time_min": float(net["time"][r].sum()),
             "risk": float(edge_risk[r].sum()),
             "distance_km": float(net["length"][r].sum())}
            for r in routes]
    keep = []
    for x in rows:
        dominated = any(
            (y["time_min"] <= x["time_min"] and y["risk"] <= x["risk"])
            and (y["time_min"] < x["time_min"] or y["risk"] < x["risk"])
            for y in rows)
        if not dominated:
            keep.append(x)
    return sorted(keep, key=lambda d: d["time_min"])


def recommend(front: list[dict], max_time_increase: float = 0.15,
              min_risk_cut: float = 0.10) -> tuple[dict, dict, str]:
    """Baseline = fastest route. Recommend the lowest-risk option inside the
    time budget, but only if the risk cut clears a materiality threshold."""
    base = front[0]
    affordable = [x for x in front if x["time_min"] <= base["time_min"] * (1 + max_time_increase)]
    best = min(affordable, key=lambda x: x["risk"])
    if best is base:
        return base, base, "fastest route is already the safest available"
    cut = 1 - best["risk"] / max(base["risk"], 1e-12)
    if cut >= min_risk_cut:
        dt = best["time_min"] - base["time_min"]
        return base, best, (f"{cut:.0%} lower risk for {dt:+.1f} min "
                            f"({dt / max(base['time_min'], 1e-9):+.0%} time)")
    return base, base, f"no alternative cut risk by at least {min_risk_cut:.0%} within the time budget"


# --------------------------------------------------------------------------
# full route scoring
# --------------------------------------------------------------------------
def score_route(model, net: dict, edges: list[int], multiplier: float = 1.0,
                n_samples: int = 4000, rng: np.random.Generator | None = None,
                level: float = 0.90) -> dict:
    """Risk of one route with a credible interval.

    The interval comes from Monte Carlo over the per-segment Gamma posteriors.
    Summing segment *means* is exact; summing segment *intervals* is not, so
    sampling is the honest way to get a route-level interval.
    """
    rng = rng or np.random.default_rng(0)
    edges = [int(e) for e in edges]
    shape = model.shape[edges]
    rate = model.rate[edges]
    # Severity-adjusted effective length: converts a sampled *count* rate into
    # average-severity-equivalent events for this segment.
    length = net["length"][edges] * model.severity_weight()[edges]

    draws = rng.gamma(shape[None, :], 1.0 / rate[None, :], size=(n_samples, len(edges)))
    totals = (draws * length[None, :]).sum(axis=1) * multiplier

    lo_q = (1 - level) / 2
    mean_risk = float(np.mean(totals))
    lo, hi = (float(np.quantile(totals, lo_q)), float(np.quantile(totals, 1 - lo_q)))

    seg_mean = model.mean()[edges] * length * multiplier
    worst = np.argsort(-seg_mean)[:5]

    # Route-level confidence: driven by how wide the interval is relative to
    # the mean, i.e. how much of the route rests on thin evidence.
    rel_width = (hi - lo) / max(mean_risk, 1e-12)
    confidence = "High" if rel_width < 0.5 else "Medium" if rel_width < 1.0 else "Low"

    conf_arr = model.confidence()[edges]
    return {
        "risk_mean": mean_risk,
        "risk_lo": lo,
        "risk_hi": hi,
        "confidence": confidence,
        "relative_interval_width": float(rel_width),
        "time_min": float(net["time"][edges].sum()),
        "distance_km": float(net["length"][edges].sum()),
        "n_segments": len(edges),
        "context_multiplier": float(multiplier),
        "low_confidence_segment_share": float(np.mean(conf_arr == "Low")),
        "worst_segments": [
            {
                "edge": int(edges[i]),
                "name": str(net["name"][edges[i]]),
                "risk": float(seg_mean[i]),
                "share_of_route": float(seg_mean[i] / max(seg_mean.sum(), 1e-12)),
                "length_km": float(net["length"][edges[i]]),
                "confidence": str(conf_arr[i]),
            }
            for i in worst
        ],
    }


def risk_per_100km(score: dict) -> float:
    """Normalised headline figure: expected weighted events per 100 vehicle-km."""
    return score["risk_mean"] / max(score["distance_km"], 1e-9) * 100.0
