"""End-to-end pipeline: network -> events -> map-match -> risk -> hotspots -> evaluation.

One entry point used by the CLI, the FastAPI service, the static-demo builder
and the test suite, so every surface reports numbers produced by the same code
path. Nothing in the demo is hand-written.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .core import analytics, evaluate
from .core.hotspots import detect_hotspots
from .core.matching import match_events
from .core.risk import SegmentRiskModel
from .data import corridor, events as ev


@dataclass
class Pipeline:
    net: dict
    model: SegmentRiskModel
    train: ev.EventTable
    test: ev.EventTable
    test_counts: np.ndarray
    test_severe: np.ndarray
    test_weighted: np.ndarray
    hotspots: list[dict]
    theta_true: np.ndarray | None
    train_months: float
    test_months: float
    match_stats: dict
    timings: dict


def load_network(osm_path: str | Path | None = None) -> dict:
    """Real OSM graph if one has been fetched, else the bundled corridor."""
    if osm_path and Path(osm_path).exists():
        from .data.osm import build_from_osm

        payload = json.loads(Path(osm_path).read_text())
        ways = payload["ways"] if "ways" in payload else payload
        for w in ways:
            w["geometry"] = [tuple(p) for p in w["geometry"]]
        net = build_from_osm(ways)
        net["place"] = payload.get("place", "OpenStreetMap extract")
        return net
    return corridor.build_network()


def run(train_months: float = 9.0, test_months: float = 4.0,
        prior: str = "gbm", half_life: float = 6.0,
        osm_path: str | Path | None = None, seed: int = 0) -> Pipeline:
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    net = load_network(osm_path)
    timings["build_network_s"] = round(time.perf_counter() - t0, 3)

    theta = ev.latent_rate(net, seed=11 + seed)

    t0 = time.perf_counter()
    train = ev.sample_events(net, theta, months=train_months, t0=0.0,
                             seed=12 + seed, start_calendar_month=1)
    test = ev.sample_events(net, theta, months=test_months, t0=train_months,
                            seed=13 + seed,
                            start_calendar_month=1 + int(train_months) % 12)
    timings["simulate_events_s"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    match_stats = match_events(net, train)
    match_events(net, test)
    timings["map_match_s"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    model = SegmentRiskModel(half_life_months=half_life, prior=prior).fit(
        net, train, now=train_months)
    timings["fit_risk_model_s"] = round(time.perf_counter() - t0, 3)

    test_counts = np.zeros(len(net["u"]))
    ok = test.edge >= 0
    np.add.at(test_counts, test.edge[ok], 1.0)

    test_weighted = np.zeros(len(net["u"]))
    np.add.at(test_weighted, test.edge[ok], test.severity[ok] / ev.MEAN_SEVERITY)

    severe = {ev.EVENT_TYPES.index("near_miss"), ev.EVENT_TYPES.index("accident")}
    sev_mask = ok & np.isin(test.type_idx, list(severe))
    test_severe = np.zeros(len(net["u"]))
    np.add.at(test_severe, test.edge[sev_mask], 1.0)

    t0 = time.perf_counter()
    hotspots = detect_hotspots(net, train, now=train_months, half_life_months=half_life)
    timings["hotspots_s"] = round(time.perf_counter() - t0, 3)

    return Pipeline(net=net, model=model, train=train, test=test,
                    test_counts=test_counts, test_severe=test_severe,
                    test_weighted=test_weighted,
                    hotspots=hotspots,
                    theta_true=theta, train_months=train_months,
                    test_months=test_months, match_stats=match_stats,
                    timings=timings)


def full_evaluation(p: Pipeline, n_trips: int = 400) -> dict:
    """Every number quoted in the submission comes from this function."""
    effective_theta = p.theta_true * p.train.context_factor
    # Ground-truth risk is frequency x severity, exactly as the model defines
    # it — otherwise the model is scored on a different quantity than it
    # optimises and the comparison is meaningless.
    true_sev = ev.expected_severity(p.net) / ev.MEAN_SEVERITY
    true_risk_rate = effective_theta * true_sev
    seg = evaluate.segment_metrics(p.model, p.net, p.test_counts,
                                   p.test_months, effective_theta,
                                   test_severe_counts=p.test_severe)
    cf = evaluate.counterfactual_replay(p.model, p.net, p.test_counts,
                                        p.test_months, n_trips=n_trips,
                                        test_weighted_counts=p.test_weighted,
                                        theta_true=true_risk_rate)
    curve = evaluate.tradeoff_curve(p.model, p.net, p.test_weighted,
                                    p.test_months, n_trips=200,
                                    theta_true=true_risk_rate)
    scale = evaluate.scalability_benchmark(p.model, p.net, n_routes=300)

    conf = p.model.confidence()
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data_source": p.net["source"],
        "place": p.net["place"],
        "honesty_note": (
            "Road network geometry is real-world anchored; telematics events are "
            "SYNTHETIC because no public Indian fleet-telematics dataset exists. "
            "These metrics validate the pipeline and the modelling choices, not "
            "real-world accuracy. Real accuracy requires a fleet data-sharing "
            "agreement; see docs/DATA_PROVENANCE.md."
        ),
        "setup": {
            "train_months": p.train_months,
            "test_months": p.test_months,
            "train_events": int(len(p.train)),
            "test_events": int(len(p.test)),
            "segments": int(len(p.net["u"])),
            "nodes": int(p.net["n_nodes"]),
            "network_km": round(float(p.net["length"].sum()), 1),
            "prior": p.model.prior_kind,
            "half_life_months": p.model.half_life,
            "fitted_dispersion_a": round(p.model.a, 3),
            "prior_log_error_sigma": round(float(p.model.sigma_prior), 3),
            "roads": int(p.model.n_roads),
        },
        "map_matching": p.match_stats,
        "confidence_mix": {k: round(float((conf == k).mean()), 4)
                           for k in ("High", "Medium", "Low")},
        "segment_model": seg,
        "counterfactual": cf,
        "tradeoff_curve": curve,
        "scalability": scale,
        "timings_s": p.timings,
    }


def export_artifacts(p: Pipeline, results: dict, out_dir: str | Path = "artifacts") -> dict:
    """Write everything the UI and the static demo need."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    net, model = p.net, p.model
    risk = model.segment_risk(net)
    lo, hi = model.segment_risk_interval(net)
    conf = model.confidence()
    per100 = risk / np.maximum(net["length"], 1e-9) * 100

    rid = np.asarray(net["road_id"])
    seen: set[int] = set()
    display_flag = []
    for e in range(len(net["u"])):
        r = int(rid[e])
        display_flag.append(0 if r in seen else 1)
        seen.add(r)

    segments = [{
        "id": int(e),
        "road": int(rid[e]),
        "disp": display_flag[e],
        "u": int(net["u"][e]),
        "v": int(net["v"][e]),
        "name": str(net["name"][e]),
        "highway": str(net["highway"][e]),
        "km": round(float(net["length"][e]), 4),
        "min": round(float(net["time"][e]), 3),
        "risk": round(float(risk[e]), 6),
        "lo": round(float(lo[e]), 6),
        "hi": round(float(hi[e]), 6),
        "per100": round(float(per100[e]), 4),
        "conf": str(conf[e]),
        "lit": int(net["lit"][e]),
        "curv": round(float(net["curvature"][e]), 4),
        "junc": round(float(net["junction_density"][e]), 3),
        "grad": round(float(net["gradient"][e]), 3),
        "speed": int(net["maxspeed"][e]),
        "evt": round(float(model.y[e]), 3),
        "wevt": round(float(model.weighted_y[e]), 2),
        "expo": round(float(model.E[e]), 1),
        "sev": round(float(model.severity_weight()[e]), 3),
        "pw": round(float(model.prior_weight[e]), 3),
        "geom": [[round(a, 5), round(b, 5)] for a, b in net["geometry"][e]],
    } for e in range(len(net["u"]))]

    is_j = net.get("node_is_junction")
    nodes = [{"id": i, "key": net["node_ids"][i], "name": net["node_name"][i],
              "lat": round(float(net["node_lat"][i]), 6),
              "lon": round(float(net["node_lon"][i]), 6),
              "j": int(is_j[i]) if is_j is not None else 1}
             for i in range(net["n_nodes"])]

    bundle = {
        "meta": {
            "place": net["place"],
            "source": net["source"],
            "version": "1.0.0",
            "generated_at": results["generated_at"],
            "honesty_note": results["honesty_note"],
            "feature_names": list(net["feature_names"]),
            "beta": [round(float(b), 5) for b in model.beta],
            "dispersion_a": round(float(model.a), 4),
            "network_km": round(float(net["length"].sum()), 1),
        },
        "nodes": nodes,
        "segments": segments,
        "hotspots": p.hotspots[:60],
        "analytics": {
            "summary": analytics.fleet_summary(model, net, p.train, p.hotspots, p.train_months),
            "by_road": analytics.risk_by_road(model, net, top_k=15),
            "by_cohort": analytics.risk_by_cohort(p.train, p.train_months),
            "by_hour": analytics.risk_by_hour(p.train),
            "monthly": analytics.monthly_trend(p.train),
        },
        "evaluation": results,
    }

    (out / "network.json").write_text(json.dumps(bundle, separators=(",", ":")))
    (out / "results.json").write_text(json.dumps(results, indent=2))
    return bundle
