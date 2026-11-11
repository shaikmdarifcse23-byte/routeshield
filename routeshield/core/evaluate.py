"""Offline evaluation: does the score actually predict where incidents happen?

Protocol is a strict **temporal holdout** — fit on the earlier window, score
the later one. Evaluating on the same months you fitted would reward
memorisation of past events, which is precisely the failure mode of the raw
historical-count baseline this system is meant to beat.

Every metric is reported against at least one baseline, because an AUC with
nothing to compare it to says nothing.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.stats import gamma as gamma_dist
from scipy.stats import rankdata, spearmanr

from .routing import build_adj, candidate_routes, pareto_frontier, recommend


# --------------------------------------------------------------------------
# ranking metrics
# --------------------------------------------------------------------------
def roc_auc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank-based AUC, ties handled by average ranks."""
    pos = label > 0
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(score: np.ndarray, label: np.ndarray) -> float:
    order = np.argsort(-score)
    y = (label[order] > 0).astype(float)
    if y.sum() == 0:
        return float("nan")
    cum = np.cumsum(y)
    prec = cum / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def capture_at_k(score: np.ndarray, future_counts: np.ndarray,
                 frac: float = 0.10) -> float:
    """Share of future events falling on the top `frac` of segments by score."""
    k = max(1, int(len(score) * frac))
    idx = np.argsort(-score)[:k]
    return float(future_counts[idx].sum() / max(future_counts.sum(), 1e-9))


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------
def calibration_table(pred_rate: np.ndarray, obs_counts: np.ndarray,
                      exposure: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Predicted vs observed event rate, in equal-count bins of prediction."""
    order = np.argsort(pred_rate)
    bins = np.array_split(order, n_bins)
    rows = []
    for i, b in enumerate(bins):
        exp = float((pred_rate[b] * exposure[b]).sum())
        obs = float(obs_counts[b].sum())
        rows.append({
            "bin": i + 1,
            "n_segments": int(len(b)),
            "mean_predicted_rate": float(pred_rate[b].mean()),
            "expected_events": round(exp, 2),
            "observed_events": round(obs, 2),
            "ratio_obs_over_exp": round(obs / exp, 3) if exp > 1e-9 else None,
        })
    return rows


def calibration_error(rows: list[dict]) -> float:
    """Mean absolute log-ratio of observed to expected across bins."""
    rs = [r["ratio_obs_over_exp"] for r in rows if r["ratio_obs_over_exp"]]
    return float(np.mean(np.abs(np.log(rs)))) if rs else float("nan")


def interval_coverage(model, truth: np.ndarray, level: float = 0.90) -> float:
    lo, hi = model.interval(level)
    return float(np.mean((truth >= lo) & (truth <= hi)))


def _predictive_coverage_arrays(shape, rate, obs_counts, exposure_future, level=0.90):
    from scipy.stats import nbinom
    p = rate / (rate + exposure_future)
    lo_q = (1 - level) / 2
    lo = nbinom.ppf(lo_q, shape, p)
    hi = nbinom.ppf(1 - lo_q, shape, p)
    return float(np.mean((obs_counts >= lo) & (obs_counts <= hi)))


def predictive_coverage(model, obs_counts: np.ndarray, exposure_future: np.ndarray,
                        level: float = 0.90) -> float:
    """Coverage of the *posterior predictive* for the next period's counts.

    This is the only coverage statistic computable on real data, where the
    latent rate is never observed. Negative-Binomial predictive: integrating
    Poisson(theta * E_future) over the Gamma posterior.
    """
    a, b = model.shape, model.rate
    p = b / (b + exposure_future)
    from scipy.stats import nbinom
    lo_q = (1 - level) / 2
    lo = nbinom.ppf(lo_q, a, p)
    hi = nbinom.ppf(1 - lo_q, a, p)
    return float(np.mean((obs_counts >= lo) & (obs_counts <= hi)))


# --------------------------------------------------------------------------
# full segment evaluation
# --------------------------------------------------------------------------
def road_view(net: dict, model, test_counts: np.ndarray):
    """Collapse directed edges to physical roads.

    The model is fitted per road (see :class:`SegmentRiskModel`), and held-out
    counts must be aggregated the same way. Evaluating per *directed* edge
    would compare a road-level prediction against a coin-flip split of that
    road's events between its two carriageways — pure noise, and it makes
    every method look worse than it is.
    """
    rid = np.asarray(net["road_id"])
    n_roads = int(rid.max()) + 1
    first = np.zeros(n_roads, dtype=int)
    first[rid[::-1]] = np.arange(len(rid))[::-1]
    return {
        "n_roads": n_roads,
        "first": first,
        "counts": np.bincount(rid, weights=test_counts, minlength=n_roads),
        "exposure": np.bincount(rid, weights=net["exposure"], minlength=n_roads),
        "length": net["length"][first],
        "X": net["X"][first],
        "y": model.y[first],
        "E": model.E[first],
        "mean": model.mean()[first],
        "shape": model.shape[first],
        "rate": model.rate[first],
    }


def segment_metrics(model, net: dict, test_counts: np.ndarray,
                    test_months: float, theta_true: np.ndarray | None = None,
                    test_severe_counts: np.ndarray | None = None) -> dict:
    """Rank, calibrate and interval-check the model on held-out data."""
    rv = road_view(net, model, test_counts)
    counts = rv["counts"]
    label = (counts > 0).astype(int)
    exposure_future = rv["exposure"] * test_months
    test_counts = counts

    # Expected future events = rate x future exposure. Comparing rates alone
    # would unfairly favour low-traffic roads.
    pred = rv["mean"] * exposure_future
    raw = rv["y"]                                   # decayed historical count
    raw_rate = rv["y"] / np.maximum(rv["E"], 1e-9) * exposure_future
    glm_only = np.exp(np.column_stack([np.ones(len(raw)), rv["X"]]) @ model.beta) * exposure_future
    length_only = rv["length"]

    out = {
        "n_segments": int(len(label)),
        "positive_rate": float(label.mean()),
        "roc_auc": {
            "routeshield": roc_auc(pred, label),
            "raw_history_count": roc_auc(raw, label),
            "raw_history_rate": roc_auc(raw_rate, label),
            "road_features_only": roc_auc(glm_only, label),
            "length_only": roc_auc(length_only, label),
        },
        "average_precision": {
            "routeshield": average_precision(pred, label),
            "raw_history_count": average_precision(raw, label),
            "length_only": average_precision(length_only, label),
        },
        "capture_at_10pct": {
            "routeshield": capture_at_k(pred, test_counts, 0.10),
            "raw_history_count": capture_at_k(raw, test_counts, 0.10),
            "length_only": capture_at_k(length_only, test_counts, 0.10),
        },
        "capture_at_20pct": {
            "routeshield": capture_at_k(pred, test_counts, 0.20),
            "raw_history_count": capture_at_k(raw, test_counts, 0.20),
        },
        "n_roads": int(rv["n_roads"]),
        "spearman_pred_vs_observed_rate": float(spearmanr(
            rv["mean"], test_counts / np.maximum(exposure_future, 1e-9)).statistic),
    }

    # A harder, more operational label than "any event at all": is this
    # segment in the worst decile by observed *rate* next period? That is the
    # question a safety team actually asks, and unlike "any event" it is not
    # simply a proxy for how busy the road is.
    obs_rate = test_counts / np.maximum(exposure_future, 1e-9)
    hot = (obs_rate >= np.quantile(obs_rate, 0.90)).astype(int)
    out["roc_auc_top_decile_rate"] = {
        "routeshield": roc_auc(rv["mean"], hot),
        "raw_history_rate": roc_auc(raw / np.maximum(rv["E"], 1e-9), hot),
        "raw_history_count": roc_auc(raw, hot),
        "road_features_only": roc_auc(np.exp(
            np.column_stack([np.ones(len(raw)), rv["X"]]) @ model.beta), hot),
    }

    # Severe events (near-miss / accident) are the rare, high-consequence
    # outcome a safety team actually wants ranked. With ~98% of roads seeing
    # *some* event in a 4-month window, the "any event" label above is close
    # to degenerate and mostly measures traffic volume; this one is not.
    if test_severe_counts is not None:
        rid = np.asarray(net["road_id"])
        sev = np.bincount(rid, weights=test_severe_counts, minlength=rv["n_roads"])
        sev_label = (sev > 0).astype(int)
        out["severe_event_prevalence"] = float(sev_label.mean())
        out["roc_auc_severe_event"] = {
            "routeshield": roc_auc(pred, sev_label),
            "raw_history_count": roc_auc(raw, sev_label),
            "road_features_only": roc_auc(glm_only, sev_label),
            "length_only": roc_auc(length_only, sev_label),
        }
        out["capture_at_10pct_severe"] = {
            "routeshield": capture_at_k(pred, sev, 0.10),
            "raw_history_count": capture_at_k(raw, sev, 0.10),
        }

    cal = calibration_table(rv["mean"], test_counts, exposure_future)
    out["calibration"] = cal
    out["calibration_error_mean_abs_log_ratio"] = calibration_error(cal)
    out["predictive_interval_coverage_90"] = _predictive_coverage_arrays(
        rv["shape"], rv["rate"], test_counts, exposure_future, 0.90)

    # Sparse segments are the whole point of the Bayesian treatment: report
    # them separately rather than letting busy roads carry the average.
    sparse = rv["E"] < np.percentile(rv["E"], 25)
    out["sparse_segment_share"] = float(sparse.mean())
    out["roc_auc_sparse_only"] = {
        "routeshield": roc_auc(pred[sparse], label[sparse]),
        "raw_history_count": roc_auc(raw[sparse], label[sparse]),
    }
    out["spearman_sparse_only"] = {
        "routeshield": float(spearmanr(rv["mean"][sparse], obs_rate[sparse]).statistic),
        "raw_history_rate": float(spearmanr(
            (raw / np.maximum(rv["E"], 1e-9))[sparse], obs_rate[sparse]).statistic),
    }

    if theta_true is not None:
        theta_true = np.asarray(theta_true)[rv["first"]]
        model_lo = gamma_dist.ppf(0.05, rv["shape"], scale=1 / rv["rate"])
        model_hi = gamma_dist.ppf(0.95, rv["shape"], scale=1 / rv["rate"])
        # theta_true must already be the *effective* rate under the training
        # context (night share x season); the fitted rate estimates that, not
        # the bare structural rate. Comparing to the bare rate would look like
        # a 38% bias that is really just the context multiplier.
        sp_true = spearmanr(rv["mean"][sparse], theta_true[sparse]).statistic
        sp_true_raw = spearmanr((raw / np.maximum(rv["E"], 1e-9))[sparse],
                                theta_true[sparse]).statistic
        out["spearman_vs_true_rate_sparse_only"] = {
            "routeshield": float(sp_true),
            "raw_history_rate": float(sp_true_raw),
        }
        # Ranking against the ground-truth rate. The observed-rate label above
        # is itself a noisy draw, and a noisy baseline can track that noise;
        # this is the fair test of which estimator is closer to the truth.
        hot_true = (theta_true >= np.quantile(theta_true, 0.90)).astype(int)
        out["roc_auc_top_decile_TRUE_rate"] = {
            "routeshield": roc_auc(rv["mean"], hot_true),
            "raw_history_rate": roc_auc(raw / np.maximum(rv["E"], 1e-9), hot_true),
            "raw_history_count": roc_auc(raw, hot_true),
            "road_features_only": roc_auc(np.exp(
                np.column_stack([np.ones(len(raw)), rv["X"]]) @ model.beta), hot_true),
        }
        cov = (theta_true >= model_lo) & (theta_true <= model_hi)
        out["latent_rate_interval_coverage_90"] = float(cov.mean())
        out["latent_rate_interval_coverage_90_sparse"] = float(cov[sparse].mean())
        out["spearman_pred_vs_true_rate"] = float(
            spearmanr(rv["mean"], theta_true).statistic)
    return out


# --------------------------------------------------------------------------
# counterfactual replay
# --------------------------------------------------------------------------
def counterfactual_replay(model, net: dict, test_counts: np.ndarray,
                          test_months: float, n_trips: int = 400,
                          test_weighted_counts: np.ndarray | None = None,
                          max_time_increase: float = 0.15,
                          min_risk_cut: float = 0.10,
                          seed: int = 5,
                          theta_true: np.ndarray | None = None) -> dict:
    """Replay trips: what if the dispatcher had taken the recommendation?

    Routes are *chosen* with the model's predictions but *scored* with
    held-out observed events, so the system cannot grade its own homework.
    """
    rng = np.random.default_rng(seed)
    adj = build_adj(net)
    risk_pred = model.segment_risk(net)
    rid = np.asarray(net["road_id"])
    n_roads = int(rid.max()) + 1
    # Held-out observed risk must be the SAME quantity the model predicts:
    # severity-weighted events per traversal. Scoring a severity-weighted
    # prediction against a plain event count compares two different things
    # and the numbers disagree for no real reason.
    obs_source = test_weighted_counts if test_weighted_counts is not None else test_counts
    road_counts = np.bincount(rid, weights=obs_source, minlength=n_roads)
    road_expo = np.bincount(rid, weights=net["exposure"], minlength=n_roads)
    obs_rate = (road_counts / np.maximum(road_expo * test_months, 1e-9))[rid] * net["length"]
    risk_true = theta_true * net["length"] if theta_true is not None else None

    # Sample origins and destinations from named junctions, not from every
    # interior split point. A dispatcher routes between depots, hubs and
    # customer sites — real junctions — and scoring trips that start halfway
    # along a highway segment would understate how much route choice exists.
    junc = net.get("node_is_junction")
    pool = (np.flatnonzero(np.asarray(junc) == 1) if junc is not None
            else np.arange(net["n_nodes"]))
    if len(pool) < 2:
        pool = np.arange(net["n_nodes"])
    rows = []
    for _ in range(n_trips):
        s, t = rng.choice(pool, 2, replace=False)
        cands = candidate_routes(net, adj, risk_pred, int(s), int(t))
        if not cands:
            continue
        front = pareto_frontier(cands, net, risk_pred)
        base, best, _ = recommend(front, max_time_increase, min_risk_cut)
        switched = best is not base
        rows.append({
            "switched": switched,
            "t_base": base["time_min"], "t_best": best["time_min"],
            "true_base": (float(risk_true[base["edges"]].sum())
                          if risk_true is not None else 0.0),
            "true_best": (float(risk_true[best["edges"]].sum())
                          if risk_true is not None else 0.0),
            "dt_pct": best["time_min"] / max(base["time_min"], 1e-9) - 1,
            "obs_base": float(obs_rate[base["edges"]].sum()),
            "obs_best": float(obs_rate[best["edges"]].sum()),
            "true_cut": (1 - risk_true[best["edges"]].sum() / max(risk_true[base["edges"]].sum(), 1e-12))
                        if risk_true is not None else None,
            "front_size": len(front),
        })

    sw = [r for r in rows if r["switched"]]
    base_total = sum(r["obs_base"] for r in rows)
    rec_total = sum(r["obs_best"] if r["switched"] else r["obs_base"] for r in rows)
    # Fleet-wide figures: averaged over EVERY trip, not only the ones that
    # switched. Mixing the two populations in one comparison (risk across all
    # trips against time among switchers) overstates the cost several-fold.
    t_base = sum(r["t_base"] for r in rows)
    t_rec = sum(r["t_best"] if r["switched"] else r["t_base"] for r in rows)
    tr_base = sum(r["true_base"] for r in rows) if risk_true is not None else None
    tr_rec = (sum(r["true_best"] if r["switched"] else r["true_base"] for r in rows)
              if risk_true is not None else None)

    res = {
        "trips_evaluated": len(rows),
        "pct_with_safer_alternative": 100.0 * len(sw) / max(len(rows), 1),
        "avg_pareto_options": float(np.mean([r["front_size"] for r in rows])) if rows else 0.0,
        "avg_time_increase_pct_when_switched":
            100.0 * float(np.mean([r["dt_pct"] for r in sw])) if sw else 0.0,
        "heldout_observed_risk_baseline": round(base_total, 2),
        "heldout_observed_risk_recommended": round(rec_total, 2),
        "heldout_risk_reduction_pct": round(100.0 * (1 - rec_total / max(base_total, 1e-9)), 2),
        "fleet_time_cost_pct": round(100.0 * (t_rec / max(t_base, 1e-9) - 1), 3),
        "time_budget_pct": 100.0 * max_time_increase,
        "min_risk_cut_pct": 100.0 * min_risk_cut,
    }
    if tr_base is not None:
        res["fleet_true_risk_reduction_pct"] = round(
            100.0 * (1 - tr_rec / max(tr_base, 1e-9)), 3)
    if risk_true is not None and sw:
        res["avg_true_risk_reduction_pct_when_switched"] = \
            100.0 * float(np.mean([r["true_cut"] for r in sw]))
    return res


def tradeoff_curve(model, net: dict, test_counts: np.ndarray, test_months: float,
                   budgets=(0.0, 0.05, 0.10, 0.15, 0.20, 0.30),
                   n_trips: int = 200, seed: int = 5,
                   theta_true: np.ndarray | None = None) -> list[dict]:
    """How much safety does each extra minute of time budget buy?

    This is the curve a fleet operations head actually has to sign off on.
    """
    out = []
    for b in budgets:
        r = counterfactual_replay(model, net, test_counts, test_months,
                                  n_trips=n_trips, max_time_increase=b, seed=seed,
                                  theta_true=theta_true)
        out.append({
            "time_budget_pct": round(100 * b, 1),
            "trips_switched_pct": round(r["pct_with_safer_alternative"], 1),
            # fleet-wide: both numbers over every trip, so they are comparable
            "fleet_time_cost_pct": r["fleet_time_cost_pct"],
            "fleet_true_risk_reduction_pct": r.get("fleet_true_risk_reduction_pct"),
            "heldout_risk_reduction_pct": r["heldout_risk_reduction_pct"],
            "avg_time_cost_pct_when_switched": round(
                r["avg_time_increase_pct_when_switched"], 2),
        })
    return out


# --------------------------------------------------------------------------
# scalability
# --------------------------------------------------------------------------
def scalability_benchmark(model, net: dict, n_routes: int = 500,
                          seed: int = 3) -> dict:
    """Measure the two operations that must stay fast in production."""
    from .routing import score_route

    rng = np.random.default_rng(seed)
    adj = build_adj(net)
    risk = model.segment_risk(net)
    n_nodes = net["n_nodes"]

    pairs = [tuple(rng.choice(n_nodes, 2, replace=False)) for _ in range(n_routes)]

    t0 = time.perf_counter()
    routes = [dijkstra_route(net, adj, risk, int(s), int(t)) for s, t in pairs]
    t_route = time.perf_counter() - t0
    routes = [r for r in routes if r]

    t0 = time.perf_counter()
    for r in routes[:200]:
        score_route(model, net, r, n_samples=1000, rng=rng)
    t_score = time.perf_counter() - t0

    t0 = time.perf_counter()
    for s, t in pairs[:100]:
        c = candidate_routes(net, adj, risk, int(s), int(t))
        pareto_frontier(c, net, risk)
    t_alt = time.perf_counter() - t0

    return {
        "n_segments": int(len(net["u"])),
        "n_nodes": int(n_nodes),
        "shortest_path_ms_per_route": round(1000 * t_route / max(len(pairs), 1), 3),
        "score_route_ms_per_route_1000_samples": round(1000 * t_score / max(min(200, len(routes)), 1), 3),
        "alternatives_ms_per_request": round(1000 * t_alt / 100, 2),
        "note": "Single core, pure Python Dijkstra. Segment risk is precomputed "
                "in the nightly batch, so request-time cost is a table lookup "
                "plus a sum; it does not grow with event volume.",
    }


def dijkstra_route(net, adj, risk, s, t):
    from .routing import dijkstra
    return dijkstra(adj, net["time"], s, t)
