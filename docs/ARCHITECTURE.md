# Architecture

```
                 ┌──────────────── nightly batch ────────────────┐
telematics ──────► ingest/dedupe ──► map-match (ambiguity gate)   │
OpenStreetMap ───► feature builder (curvature, junctions,         │
                   lighting, gradient, speed, lanes)              │
DEM / Open-Meteo ► gradient, weather history                      │
                                   │                              │
                     ┌─────────────┴──────────────┐               │
                     │  frequency: Gamma-Poisson  │               │
                     │  severity : shrunk factor  │               │
                     │  + prior-error inflation   │               │
                     └─────────────┬──────────────┘               │
                                   ▼                              │
                   segment_risk table / vector tiles ─────────────┘
                                   │
dispatcher UI ──► /api/plan ──► lookup + sum + context multiplier
                                   ├─► Pareto alternatives (λ sweep + Dijkstra)
                                   ├─► attribution + hotspots on route
                                   └─► /api/assign ──► route_assignments
                                                            │
                           observed events ──► post-trip join ──► analytics,
                                                       score-vs-outcome, overrides
```

## Why the split matters

Everything expensive happens in the batch. Request time is a table lookup and a
sum over the route's segments, plus a Monte Carlo draw for the interval. That
is why scoring a route costs **0.74 ms** and does not grow with event volume —
only the nightly fit does, and that is well under a second for the whole network.

## Module map

| Module | Responsibility |
|---|---|
| `data/corridor.py` | bundled geo-anchored Bengaluru network |
| `data/osm.py` | live Overpass ingestion → identical schema |
| `data/events.py` | event schema, CSV loader, labelled synthetic generator |
| `core/matching.py` | spatial hash + perpendicular distance + ambiguity gate |
| `core/risk.py` | GLM/GBM prior, empirical Bayes, posterior, attribution |
| `core/hotspots.py` | HDBSCAN black spots with trend |
| `core/routing.py` | Dijkstra, λ sweep, Pareto, recommendation, route scoring |
| `core/context.py` | time-of-day / weather / cohort multipliers, Open-Meteo |
| `core/evaluate.py` | temporal holdout, baselines, calibration, counterfactual |
| `core/analytics.py` | trend reports by road, cohort, hour, month |
| `core/store.py` | SQLite trip log (PostGIS-shaped) |
| `api/server.py` | FastAPI |
| `pipeline.py` | one path used by CLI, API, tests and the static demo |

## Scaling

* **Segments.** Risk is per physical road, computed vectorised; 292 roads fit
  in about 0.7 s, and the cost is linear. A national network (~10⁷ segments) is a
  partitioned batch job, not a different algorithm.
* **Events.** Only the nightly fit sees them. Counts and exposure are additive,
  so the batch becomes an incremental update rather than a refit.
* **Requests.** Stateless; the segment table is read-only at request time and
  caches trivially. Alternatives are the costly path at 1.4 ms, still ~700 rps
  per core in pure Python.
* **Production swaps.** SQLite → PostGIS (the schema is unchanged, only the
  geometry column type); the in-process table → a materialised view plus vector
  tiles for the map.

## Why road-level, not direction-level

The two directed edges of a carriageway share geometry, so a GPS event cannot
be attributed to one direction — matching it is a coin flip, which splits counts
at random while each direction carries its own exposure. The model therefore
fits per physical road and broadcasts the posterior to both directions.
Evaluation aggregates the same way; doing otherwise compares a road-level
prediction against a random split of that road's events and makes every method
look worse than it is.
