"""FastAPI service — the dispatch integration point.

Start it with::

    python -m routeshield.api.server        # http://127.0.0.1:8000

Endpoints
---------
``GET  /api/health``            build + data provenance
``GET  /api/nodes``             selectable origins and destinations
``GET  /api/segments``          segment risk table (the nightly batch output)
``GET  /api/hotspots``          ranked black spots
``POST /api/score``             score one planned route
``POST /api/plan``              alternatives + Pareto frontier + recommendation
``POST /api/assign``            log a dispatch decision
``GET  /api/trips``             recent assignments + adoption stats
``POST /api/outcome``           post-trip observed events, for score-vs-outcome
``GET  /api/analytics``         trend reports
``GET  /api/evaluation``        the full held-out evaluation

The model is fitted once at startup (in production this is a nightly batch
writing to a table); every request is a lookup plus a sum.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import pipeline
from ..core import analytics as an
from ..core.context import context_breakdown, context_multiplier, fetch_weather
from ..core.hotspots import hotspots_on_route
from ..core.routing import (build_adj, candidate_routes, pareto_frontier,
                            recommend, score_route)
from ..core.store import OVERRIDE_REASONS, TripLog

WEB_DIR = Path(__file__).resolve().parents[2] / "web"

app = FastAPI(title="RouteShield", version="1.0.0",
              description="Uncertainty-aware fleet route risk intelligence.")

STATE: dict = {}


@app.on_event("startup")
def _startup() -> None:
    osm = os.environ.get("ROUTESHIELD_OSM")
    p = pipeline.run(osm_path=osm)
    STATE["p"] = p
    STATE["adj"] = build_adj(p.net)
    STATE["risk"] = p.model.segment_risk(p.net)
    STATE["log"] = TripLog(os.environ.get("ROUTESHIELD_DB", "artifacts/routeshield.db"))
    STATE["rng"] = np.random.default_rng(0)
    res_path = Path("artifacts/results.json")
    STATE["eval"] = (pipeline.full_evaluation(p, n_trips=200)
                     if not res_path.exists() else
                     __import__("json").loads(res_path.read_text()))


# ----------------------------------------------------------------- models
class PlanRequest(BaseModel):
    origin: int
    destination: int
    hour: int = Field(12, ge=0, le=23)
    weather: str = "clear"
    cohort: str | None = None
    max_time_increase: float = Field(0.15, ge=0, le=2)
    min_risk_cut: float = Field(0.10, ge=0, le=1)
    use_live_weather: bool = False


class ScoreRequest(BaseModel):
    edges: list[int]
    hour: int = 12
    weather: str = "clear"
    cohort: str | None = None


class AssignRequest(BaseModel):
    edges: list[int]
    origin: int
    destination: int
    hour: int = 12
    weather: str = "clear"
    cohort: str | None = None
    chosen_is_recommended: bool = True
    baseline_risk: float | None = None
    baseline_eta_min: float | None = None
    alternatives_shown: int = 0
    override_reason: str | None = None
    dispatcher_id: str = "api"


class OutcomeRequest(BaseModel):
    trip_id: str
    observed_events: float
    observed_severe: int = 0


# ---------------------------------------------------------------- helpers
def _net():
    if "p" not in STATE:
        raise HTTPException(503, "model still warming up")
    return STATE["p"].net, STATE["p"].model


def _multiplier(hour, weather, cohort):
    return context_multiplier(hour, weather, cohort)


def _route_payload(net, edges):
    return {
        "edges": [int(e) for e in edges],
        "geometry": [net["geometry"][e] for e in edges],
        "roads": list(dict.fromkeys(str(net["name"][e]) for e in edges)),
    }


# --------------------------------------------------------------- endpoints
@app.get("/api/health")
def health():
    net, model = _net()
    return {
        "status": "ok", "version": "1.0.0",
        "data_source": net["source"], "place": net["place"],
        "segments": int(len(net["u"])), "roads": int(model.n_roads),
        "nodes": int(net["n_nodes"]),
        "honesty_note": STATE["eval"]["honesty_note"],
    }


@app.get("/api/nodes")
def nodes(junctions_only: bool = True):
    net, _ = _net()
    isj = net.get("node_is_junction")
    out = []
    for i in range(net["n_nodes"]):
        if junctions_only and isj is not None and not isj[i]:
            continue
        out.append({"id": i, "name": net["node_name"][i],
                    "lat": float(net["node_lat"][i]),
                    "lon": float(net["node_lon"][i])})
    return sorted(out, key=lambda n: n["name"])


@app.get("/api/segments")
def segments(limit: int = 2000):
    net, model = _net()
    risk = STATE["risk"]
    lo, hi = model.segment_risk_interval(net)
    conf = model.confidence()
    return [{
        "id": e, "name": str(net["name"][e]), "km": float(net["length"][e]),
        "risk": float(risk[e]), "lo": float(lo[e]), "hi": float(hi[e]),
        "risk_per_100km": float(risk[e] / max(net["length"][e], 1e-9) * 100),
        "confidence": str(conf[e]),
        "evidence": model.evidence(e),
    } for e in range(min(limit, len(net["u"])))]


@app.get("/api/segments/{edge}")
def segment_detail(edge: int):
    net, model = _net()
    if not 0 <= edge < len(net["u"]):
        raise HTTPException(404, "no such segment")
    return {
        "id": edge, "name": str(net["name"][edge]),
        "highway": str(net["highway"][edge]),
        "length_km": float(net["length"][edge]),
        "risk": float(STATE["risk"][edge]),
        "confidence": str(model.confidence()[edge]),
        "evidence": model.evidence(edge),
        "why": [{"factor": k, "contribution": v}
                for k, v in model.attribution(net, edge)],
        "geometry": net["geometry"][edge],
    }


@app.get("/api/hotspots")
def hotspots(limit: int = 50):
    return STATE["p"].hotspots[:limit]


@app.post("/api/score")
def score(req: ScoreRequest):
    net, model = _net()
    if not req.edges:
        raise HTTPException(400, "edges must not be empty")
    if max(req.edges) >= len(net["u"]) or min(req.edges) < 0:
        raise HTTPException(400, "edge index out of range")
    mult = _multiplier(req.hour, req.weather, req.cohort)
    sc = score_route(model, net, req.edges, mult, rng=STATE["rng"])
    sc["context"] = context_breakdown(req.hour, req.weather, req.cohort)
    sc["hotspots"] = hotspots_on_route(STATE["p"].hotspots, req.edges)
    sc["why"] = [{"factor": k, "contribution": v}
                 for k, v in model.attribution(net, req.edges[len(req.edges) // 2])]
    return sc


@app.post("/api/plan")
def plan(req: PlanRequest):
    net, model = _net()
    n = net["n_nodes"]
    if not (0 <= req.origin < n and 0 <= req.destination < n):
        raise HTTPException(400, "node index out of range")
    if req.origin == req.destination:
        raise HTTPException(400, "origin and destination are the same")

    weather = req.weather
    weather_meta = None
    if req.use_live_weather:
        w = fetch_weather(float(net["node_lat"][req.origin]),
                          float(net["node_lon"][req.origin]))
        weather, weather_meta = w["bucket"], w

    mult = _multiplier(req.hour, weather, req.cohort)
    cands = candidate_routes(net, STATE["adj"], STATE["risk"], req.origin, req.destination)
    if not cands:
        raise HTTPException(404, "no route between these nodes")
    front = pareto_frontier(cands, net, STATE["risk"])
    base, best, reason = recommend(front, req.max_time_increase, req.min_risk_cut)

    def pack(r):
        sc = score_route(model, net, r["edges"], mult, rng=STATE["rng"])
        sc.update(_route_payload(net, r["edges"]))
        return sc

    return {
        "context": context_breakdown(req.hour, weather, req.cohort),
        "context_multiplier": mult,
        "weather_source": weather_meta,
        "frontier": [pack(r) for r in front],
        "baseline_index": front.index(base),
        "recommended_index": front.index(best),
        "recommendation_reason": reason,
        "hotspots_on_recommended": hotspots_on_route(STATE["p"].hotspots, best["edges"]),
    }


@app.post("/api/assign")
def assign(req: AssignRequest):
    net, model = _net()
    if req.override_reason and req.override_reason not in OVERRIDE_REASONS:
        raise HTTPException(400, f"override_reason must be one of {OVERRIDE_REASONS}")
    if not req.chosen_is_recommended and not req.override_reason:
        raise HTTPException(400, "an override requires a reason code")
    mult = _multiplier(req.hour, req.weather, req.cohort)
    sc = score_route(model, net, req.edges, mult, rng=STATE["rng"])
    trip_id = "T" + uuid.uuid4().hex[:10].upper()
    STATE["log"].log_assignment(
        trip_id, sc, req.edges, dispatcher_id=req.dispatcher_id,
        origin_node=req.origin, destination_node=req.destination,
        origin_name=str(net["node_name"][req.origin]),
        destination_name=str(net["node_name"][req.destination]),
        depart_hour=req.hour, weather=req.weather, driver_cohort=req.cohort,
        chosen_is_recommended=req.chosen_is_recommended,
        baseline_risk=req.baseline_risk, baseline_eta_min=req.baseline_eta_min,
        alternatives_shown=req.alternatives_shown,
        override_reason=req.override_reason)
    return {"trip_id": trip_id, "logged": True, "score": sc}


@app.post("/api/outcome")
def outcome(req: OutcomeRequest):
    STATE["log"].log_outcome(req.trip_id, req.observed_events, req.observed_severe)
    return {"ok": True}


@app.get("/api/trips")
def trips(limit: int = 50):
    log = STATE["log"]
    return {"stats": log.adoption_stats(), "overrides": log.override_reasons(),
            "recent": log.recent(limit), "score_vs_outcome": log.score_vs_outcome()}


@app.get("/api/analytics")
def analytics():
    p = STATE["p"]
    return {
        "summary": an.fleet_summary(p.model, p.net, p.train, p.hotspots, p.train_months),
        "by_road": an.risk_by_road(p.model, p.net, top_k=20),
        "by_cohort": an.risk_by_cohort(p.train, p.train_months),
        "by_hour": an.risk_by_hour(p.train),
        "monthly": an.monthly_trend(p.train),
        "recurring_hotspots": an.recurring_hotspots(p.hotspots)[:20],
    }


@app.get("/api/evaluation")
def evaluation():
    return STATE["eval"]


# ----------------------------------------------------- static dispatcher UI
if WEB_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

    @app.get("/")
    def root():
        return FileResponse(str(WEB_DIR / "index.html"))


def main() -> None:                                   # pragma: no cover
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))


if __name__ == "__main__":                            # pragma: no cover
    main()
