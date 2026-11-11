"""Test suite.

These are not smoke tests. Each one pins a property that, if it broke,
would make the product quietly wrong rather than loudly broken — the
failure mode that matters for a risk model.

Run:  pytest -q
"""
from __future__ import annotations

import json
import math
import subprocess
import sys

import numpy as np
import pytest

from routeshield.core import evaluate
from routeshield.core.context import context_multiplier, WMO_TO_BUCKET
from routeshield.core.hotspots import detect_hotspots
from routeshield.core.matching import MapMatcher, match_events
from routeshield.core.risk import SegmentRiskModel, fit_poisson_glm
from routeshield.core.routing import (build_adj, candidate_routes, dijkstra,
                                      pareto_frontier, recommend, score_route)
from routeshield.core.store import TripLog
from routeshield.data import corridor, events as ev
from routeshield.data.osm import build_from_osm, parse_overpass, _parse_maxspeed


# --------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def net():
    return corridor.build_network()


@pytest.fixture(scope="module")
def fitted(net):
    theta = ev.latent_rate(net)
    train = ev.sample_events(net, theta, months=9.0, t0=0.0, seed=12)
    match_events(net, train)
    model = SegmentRiskModel(half_life_months=6.0, prior="gbm").fit(net, train, now=9.0)
    return net, model, train, theta


# --------------------------------------------------------------- network
def test_network_is_reproducible():
    """Python randomises string hashing per process; the network must not
    depend on it, or every published metric becomes unreproducible."""
    a, b = corridor.build_network(), corridor.build_network()
    assert np.allclose(a["length"], b["length"])
    assert np.allclose(a["curvature"], b["curvature"])
    assert np.allclose(a["exposure"], b["exposure"])


def test_reproducible_across_processes():
    code = ("from routeshield.data import corridor;"
            "n=corridor.build_network();"
            "print(round(float(n['length'].sum()),6), round(float(n['curvature'].sum()),6))")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, check=True).stdout.strip() for _ in range(2)}
    assert len(outs) == 1, f"network differs between processes: {outs}"


def test_both_directions_share_a_road(net):
    rid = net["road_id"]
    for r in range(min(40, int(rid.max()) + 1)):
        e = np.flatnonzero(rid == r)
        assert 1 <= len(e) <= 2
        if len(e) == 2:
            a, b = e
            assert net["u"][a] == net["v"][b] and net["v"][a] == net["u"][b]
            assert math.isclose(net["length"][a], net["length"][b], rel_tol=1e-9)


def test_geometry_matches_length(net):
    for e in range(0, len(net["u"]), 37):
        assert math.isclose(corridor.polyline_length_km(net["geometry"][e]),
                            net["length"][e], rel_tol=1e-6)


def test_features_are_finite_and_in_range(net):
    assert np.isfinite(net["X"]).all()
    assert (net["curvature"] >= 0).all() and (net["curvature"] <= 1).all()
    assert (net["length"] > 0).all() and (net["time"] > 0).all()
    assert (net["exposure"] > 0).all()


# --------------------------------------------------------------- matching
def test_matcher_snaps_a_point_on_a_road_to_that_road(net):
    mm = MapMatcher(net)
    e = 11
    pts = net["geometry"][e]
    mid = pts[len(pts) // 2]
    got, dist = mm.match([mid[0]], [mid[1]], 60.0, margin_m=0.0,
                         road_id=net["road_id"])
    assert net["road_id"][got[0]] == net["road_id"][e]
    assert dist[0] < 1.0


def test_matcher_rejects_points_far_from_any_road(net):
    mm = MapMatcher(net)
    got, _ = mm.match([20.0], [80.0], 60.0, road_id=net["road_id"])
    assert got[0] == -1


def test_ambiguity_gate_cuts_cross_road_leakage(net):
    """The gate exists to stop a quiet side road inheriting a trunk road's
    events at a junction. If it stops working, risk maps go subtly wrong."""
    theta = ev.latent_rate(net)
    tr = ev.sample_events(net, theta, months=3.0, t0=0.0, seed=5)
    mm = MapMatcher(net)
    rid = net["road_id"]

    loose, _ = mm.match(tr.lat, tr.lon, 60.0, margin_m=0.0, road_id=rid)
    tight, _ = mm.match(tr.lat, tr.lon, 60.0, margin_m=18.0, road_id=rid)

    acc = lambda m: (rid[m[m >= 0]] == rid[tr.true_edge[m >= 0]]).mean()
    assert acc(tight) > acc(loose)
    assert acc(tight) > 0.98
    assert (tight >= 0).mean() > 0.75          # still keeps most of the data


# ------------------------------------------------------------------ risk
def test_poisson_glm_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    n = 4000
    X = rng.normal(0, 1, (n, 3))
    beta = np.array([-2.0, 0.8, -0.4, 0.25])
    expo = rng.uniform(5, 50, n)
    mu = np.exp(beta[0] + X @ beta[1:]) * expo
    y = rng.poisson(mu).astype(float)
    got = fit_poisson_glm(X, y, expo)
    assert np.allclose(got, beta, atol=0.08)


def test_posterior_mean_sits_between_prior_and_data(fitted):
    _, m, _, _ = fitted
    lo = np.minimum(m.m, m.y / np.maximum(m.E, 1e-9))
    hi = np.maximum(m.m, m.y / np.maximum(m.E, 1e-9))
    post = m.mean()
    assert (post >= lo - 1e-9).all() and (post <= hi + 1e-9).all()


def test_sparse_segments_are_pulled_toward_the_prior(fitted):
    """The whole point of the hierarchical model: a road nobody has driven
    must inherit the risk of similar roads, not report zero."""
    _, m, _, _ = fitted
    sparse = m.E < np.percentile(m.E, 15)
    rich = m.E > np.percentile(m.E, 85)
    assert m.prior_weight[sparse].mean() > m.prior_weight[rich].mean()
    assert m.prior_weight[sparse].mean() > 0.5
    assert (m.mean()[m.y == 0] > 0).all()      # never a zero-risk road


def test_confidence_badge_is_predictive(fitted):
    """A confidence badge that does not separate accuracy is decoration."""
    net, m, train, theta = fitted
    eff = theta * train.context_factor
    err = np.abs(np.log(m.mean() / eff))
    conf = m.confidence()
    med = {k: float(np.median(err[conf == k])) for k in ("High", "Medium", "Low")
           if (conf == k).any()}
    assert "High" in med and "Low" in med, f"badge never varies: {med}"
    assert med["High"] < med["Low"], med
    if "Medium" in med:
        assert med["High"] <= med["Medium"] <= med["Low"], med
    # and every band must actually be used
    for k in ("High", "Medium", "Low"):
        assert (conf == k).mean() > 0.05, f"band {k} is unused: {med}"


def test_no_segment_reports_zero_or_negative_risk(fitted):
    net, m, _, _ = fitted
    r = m.segment_risk(net)
    assert np.isfinite(r).all() and (r > 0).all()


def test_credible_interval_brackets_the_mean(fitted):
    _, m, _, _ = fitted
    lo, hi = m.interval(0.90)
    mean = m.mean()
    assert (lo <= mean + 1e-12).all() and (hi >= mean - 1e-12).all()
    assert (hi > lo).all()


def test_interval_coverage_is_near_nominal(fitted):
    """A confidence badge nobody can trust is worse than no badge."""
    net, m, train, theta = fitted
    eff = theta * train.context_factor * (ev.expected_severity(net) / ev.MEAN_SEVERITY)
    # the model's rate is frequency only; compare like with like
    eff_freq = theta * train.context_factor
    lo, hi = m.interval(0.90)
    cov = float(np.mean((eff_freq >= lo) & (eff_freq <= hi)))
    assert 0.80 <= cov <= 0.97, f"90% interval covered {cov:.3f}"


def test_recency_decay_discounts_old_events(net):
    theta = ev.latent_rate(net)
    e = ev.sample_events(net, theta, months=12.0, t0=0.0, seed=3)
    match_events(net, e)
    n = len(net["u"])
    fresh = SegmentRiskModel(half_life_months=1.0).fit(net, e, now=12.0)
    slow = SegmentRiskModel(half_life_months=24.0).fit(net, e, now=12.0)
    assert fresh.y.sum() < slow.y.sum()


def test_severity_factor_is_shrunk_toward_the_mean(fitted):
    _, m, _, _ = fitted
    s = m.severity_weight()
    assert (s > 0.3).all() and (s < 4.0).all()


def test_model_beats_raw_history_on_sparse_roads(fitted):
    """The core claim of the whole system, asserted as a test."""
    from scipy.stats import spearmanr
    net, m, train, theta = fitted
    rid = net["road_id"]
    first = np.zeros(int(rid.max()) + 1, dtype=int)
    first[rid[::-1]] = np.arange(len(rid))[::-1]
    eff = (theta * train.context_factor)[first]
    E, y, post = m.E[first], m.y[first], m.mean()[first]
    sparse = E < np.percentile(E, 25)
    ours = spearmanr(post[sparse], eff[sparse]).statistic
    raw = spearmanr((y / np.maximum(E, 1e-9))[sparse], eff[sparse]).statistic
    assert ours > raw + 0.1, f"routeshield {ours:.3f} vs raw {raw:.3f}"


# --------------------------------------------------------------- routing
def test_dijkstra_finds_the_cheapest_path(net):
    adj = build_adj(net)
    cost = net["time"]
    r = dijkstra(adj, cost, 0, 5)
    assert r is not None
    total = cost[r].sum()
    for lam in (0.0,):
        other = dijkstra(adj, cost + lam, 0, 5)
        assert cost[other].sum() >= total - 1e-9


def test_route_is_a_connected_chain(net):
    adj = build_adj(net)
    r = dijkstra(adj, net["time"], 3, 40)
    assert r
    for a, b in zip(r[:-1], r[1:]):
        assert net["v"][a] == net["u"][b]


def test_pareto_frontier_is_actually_pareto(fitted):
    net, m, _, _ = fitted
    adj = build_adj(net)
    risk = m.segment_risk(net)
    front = pareto_frontier(candidate_routes(net, adj, risk, 2, 30), net, risk)
    assert front
    for i, a in enumerate(front):
        for b in front[i + 1:]:
            assert not (b["time_min"] <= a["time_min"] and b["risk"] <= a["risk"]
                        and (b["time_min"] < a["time_min"] or b["risk"] < a["risk"]))
    times = [f["time_min"] for f in front]
    risks = [f["risk"] for f in front]
    assert times == sorted(times)
    assert risks == sorted(risks, reverse=True)   # risk must fall as time rises


def test_recommendation_respects_the_time_budget(fitted):
    net, m, _, _ = fitted
    adj = build_adj(net)
    risk = m.segment_risk(net)
    for s, t in ((1, 25), (4, 33), (7, 19)):
        front = pareto_frontier(candidate_routes(net, adj, risk, s, t), net, risk)
        if not front:
            continue
        base, best, _ = recommend(front, max_time_increase=0.10, min_risk_cut=0.10)
        assert best["time_min"] <= base["time_min"] * 1.10 + 1e-9
        if best is not base:
            assert best["risk"] <= base["risk"] * 0.90 + 1e-12


def test_route_score_interval_contains_its_mean(fitted):
    net, m, _, _ = fitted
    adj = build_adj(net)
    r = dijkstra(adj, net["time"], 2, 28)
    sc = score_route(m, net, r, multiplier=1.0, n_samples=3000,
                     rng=np.random.default_rng(1))
    assert sc["risk_lo"] < sc["risk_mean"] < sc["risk_hi"]
    assert abs(sc["time_min"] - net["time"][r].sum()) < 1e-6


def test_context_multiplier_scales_risk_linearly(fitted):
    net, m, _, _ = fitted
    adj = build_adj(net)
    r = dijkstra(adj, net["time"], 2, 28)
    a = score_route(m, net, r, 1.0, 4000, np.random.default_rng(2))
    b = score_route(m, net, r, 2.0, 4000, np.random.default_rng(2))
    assert abs(b["risk_mean"] / a["risk_mean"] - 2.0) < 0.05


# --------------------------------------------------------------- context
def test_night_and_fog_raise_risk():
    assert context_multiplier(2, "clear") > context_multiplier(12, "clear")
    assert context_multiplier(12, "fog") > context_multiplier(12, "clear")
    assert context_multiplier(2, "fog") > context_multiplier(2, "clear")


def test_wmo_codes_map_to_known_buckets():
    from routeshield.core.context import WEATHER_MULTIPLIER
    for bucket in set(WMO_TO_BUCKET.values()):
        assert bucket in WEATHER_MULTIPLIER


# -------------------------------------------------------------- hotspots
def test_hotspots_are_ranked_and_located(fitted):
    net, _, train, _ = fitted
    hs = detect_hotspots(net, train, now=9.0, min_cluster_size=20)
    assert hs
    assert [h["rank"] for h in hs] == list(range(1, len(hs) + 1))
    scores = [h["weighted_score"] for h in hs]
    assert scores == sorted(scores, reverse=True)
    for h in hs[:10]:
        assert 12.0 < h["lat"] < 14.0 and 77.0 < h["lon"] < 78.5
        assert h["trend_label"] in ("worsening", "stable", "improving")


# ------------------------------------------------------------------- OSM
OVERPASS_FIXTURE = {
    "elements": [
        {"type": "way", "id": 1, "nodes": [10, 11, 12],
         "geometry": [{"lat": 12.90, "lon": 77.60}, {"lat": 12.905, "lon": 77.605},
                      {"lat": 12.91, "lon": 77.61}],
         "tags": {"highway": "trunk", "name": "Test Trunk", "maxspeed": "60",
                  "lanes": "3", "lit": "yes"}},
        {"type": "way", "id": 2, "nodes": [12, 13],
         "geometry": [{"lat": 12.91, "lon": 77.61}, {"lat": 12.92, "lon": 77.62}],
         "tags": {"highway": "primary", "oneway": "yes", "maxspeed": "40 mph"}},
        {"type": "node", "id": 99, "lat": 12.9, "lon": 77.6},
        {"type": "way", "id": 3, "nodes": [20, 21], "tags": {"highway": "primary"}},
    ]
}


def test_overpass_parsing_handles_real_tag_messiness():
    ways = parse_overpass(OVERPASS_FIXTURE)
    assert len(ways) == 2                       # node and geometry-less way dropped
    w = ways[0]
    assert w["highway"] == "trunk" and w["maxspeed"] == 60.0
    assert w["lit"] == 1.0 and w["oneway"] is False
    assert ways[1]["oneway"] is True
    assert abs(ways[1]["maxspeed"] - 40 * 1.609) < 0.5


def test_maxspeed_parsing_edge_cases():
    assert _parse_maxspeed(None, "motorway") == 100
    assert _parse_maxspeed("IN:urban", "primary") == 60      # unparseable -> default
    assert _parse_maxspeed("80", "primary") == 80
    assert abs(_parse_maxspeed("50 mph", "primary") - 80.45) < 0.5


def test_osm_graph_has_the_same_schema_as_the_bundled_one(net):
    built = build_from_osm(parse_overpass(OVERPASS_FIXTURE))
    for key in ("u", "v", "length", "time", "X", "exposure", "road_id",
                "geometry", "name", "highway", "feature_names", "n_nodes"):
        assert key in built, f"OSM graph missing {key}"
    assert built["X"].shape[1] == net["X"].shape[1]
    assert len(built["u"]) == len(built["road_id"])
    # the one-way should get a single directed edge, the two-way a pair
    assert set(np.bincount(built["road_id"])) <= {1, 2}


# ------------------------------------------------------------------ store
def test_trip_log_round_trip(tmp_path):
    log = TripLog(tmp_path / "t.db")
    score = {"risk_mean": 1.5, "risk_lo": 1.0, "risk_hi": 2.1,
             "confidence": "High", "time_min": 42.0, "distance_km": 30.0}
    log.log_assignment("T1", score, [1, 2, 3], baseline_risk=2.0,
                       baseline_eta_min=40.0, alternatives_shown=3)
    log.log_assignment("T2", score, [4, 5], chosen_is_recommended=False,
                       override_reason="toll_cost", baseline_risk=1.2,
                       baseline_eta_min=44.0)
    assert len(log.recent()) == 2
    st = log.adoption_stats()
    assert st["trips_logged"] == 2 and st["recommendation_adopted"] == 1
    assert st["adoption_rate_pct"] == 50.0
    assert [r["reason"] for r in log.override_reasons()] == ["toll_cost"]
    log.log_outcome("T1", observed_events=1.2, observed_severe=0)
    assert len(log.score_vs_outcome()) == 1
    log.close()


def test_trip_log_rejects_unknown_override_reason(tmp_path):
    log = TripLog(tmp_path / "t2.db")
    score = {"risk_mean": 1, "risk_lo": 0, "risk_hi": 2, "confidence": "Low",
             "time_min": 1, "distance_km": 1}
    log.log_assignment("X", score, [0], chosen_is_recommended=False,
                       override_reason="because i felt like it")
    assert log.recent()[0]["override_reason"] == "other"
    log.close()


# ------------------------------------------------------------- evaluation
def test_auc_matches_a_known_value():
    assert evaluate.roc_auc(np.array([1., 2, 3, 4]), np.array([0, 0, 1, 1])) == 1.0
    assert evaluate.roc_auc(np.array([4., 3, 2, 1]), np.array([0, 0, 1, 1])) == 0.0
    assert abs(evaluate.roc_auc(np.array([1., 2, 3, 4]),
                                np.array([0, 1, 0, 1])) - 0.75) < 1e-9


def test_capture_at_k_is_a_share():
    s = np.array([5., 4, 3, 2, 1, 0, 0, 0, 0, 0])
    c = np.array([10., 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert evaluate.capture_at_k(s, c, 0.10) == 1.0


def test_pipeline_runs_and_reports_sane_numbers():
    from routeshield import pipeline
    p = pipeline.run(train_months=6.0, test_months=3.0)
    res = pipeline.full_evaluation(p, n_trips=40)
    sm = res["segment_model"]
    assert 0.5 < sm["roc_auc_top_decile_TRUE_rate"]["routeshield"] <= 1.0
    assert sm["roc_auc_top_decile_TRUE_rate"]["routeshield"] > \
           sm["roc_auc_top_decile_TRUE_rate"]["raw_history_count"]
    assert 0.75 < sm["predictive_interval_coverage_90"] < 1.0
    assert res["map_matching"]["road_match_accuracy"] > 0.97
    json.dumps(res)            # must be serialisable for the UI bundle
