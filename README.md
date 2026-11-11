# RouteShield

**Score a planned fleet route for risk *before* dispatch — with an honest confidence interval, an explanation a dispatcher can argue with, and safer alternatives on an explicit risk-vs-time trade-off.**

Open source (MIT). Runs end to end in about 5 seconds on a laptop, with no API keys and no GPU.

🔗 **[Live demo](https://shaikmdarifcse23-byte.github.io/routeshield/)** · the full model runs client-side, no backend required.

---

## The problem

Dispatchers assign routes by distance or ETA. The thing that actually hurts — an unlit stretch, a ghat curve, a junction with a history of harsh braking, fog season, a 2 a.m. departure — is invisible at the moment the decision is made.

The obvious fix is a heat map of historical incidents. It does not work, for a reason that is easy to miss:

> **A raw event count tells you how *busy* a road is, not how *risky* it is.**

On our held-out evaluation, ranking roads by raw historical event count scores **AUC 0.46** at identifying the genuinely highest-rate roads — *worse than a coin flip*. It ranks the ORR highly because everything drives on the ORR. Meanwhile most of the network has almost no fleet history at all, and a count-based map reports those roads as "zero risk" rather than "unknown".

## What RouteShield does

| | |
|---|---|
| **Ranks risk, not traffic** | AUC **0.961** at identifying the worst decile of roads by true event rate, against 0.840 for historical rates and 0.459 for raw counts. |
| **Knows what it doesn't know** | Every segment carries a 90% credible interval. Measured coverage: **89.0%** against the true rate, **91.8%** on the posterior predictive. |
| **Helps most where data is thinnest** | On the quartile of roads with the least history, Spearman correlation with the true rate is **0.789** vs **0.476** for raw history — a 66% improvement, exactly where a naive map fails. |
| **Shows the trade, doesn't hide it** | Alternatives are a Pareto frontier of time against risk. Fleet-wide, adopting the recommendations cuts true risk **4.0%** for **+1.0%** travel time. |
| **Explains itself** | Exact additive attribution — closed-form, not a sampled approximation — plus the black spots on the route and the evidence behind each score. |

---

## Quick start

```bash
pip install -r requirements.txt
python -m routeshield.cli demo        # full pipeline + evaluation, ~5 s
```

Then either:

```bash
make web      # static dispatcher console on http://localhost:8080
make serve    # FastAPI service + UI on  http://localhost:8000
make test     # 35 tests, ~13 s
```

### Use real OpenStreetMap data

The bundled network is geo-anchored to real Bengaluru junctions but its
inter-junction geometry is interpolated (see [Honesty](#honesty)). One command
replaces it with genuine OSM geometry and tags:

```bash
make osm                                         # fetches via Overpass
ROUTESHIELD_OSM=artifacts/osm_ways.json make demo
```

Nothing downstream changes — `routeshield.data.osm` emits the identical schema.

---

## How it works

```
telematics events ──► map-match (ambiguity-gated) ──┐
OpenStreetMap ──────► segment features ─────────────┼──► segment risk model ──► nightly batch
DEM / Open-Meteo ───► gradient, weather ────────────┘      (frequency × severity,
                                                            Gamma-Poisson posterior)
                                                                   │
dispatcher ──► /api/plan ──► route scoring + Pareto alternatives ◄──┘
                   │
                   └──► trip log ──► post-trip join ──► analytics, score-vs-outcome
```

### 1. Map matching, with a gate that refuses to guess

Events arrive as raw GPS with 10–30 m of error. Near a junction two roads are
almost equidistant, and forcing a match there is how a quiet side road silently
inherits a trunk road's harsh-braking events.

Measured on this network, forcing every match misattributes **4.6%** of events
— enough to make a handful of low-traffic roads look like the worst in the city,
and enough to wipe out the benefit of re-routing entirely (we measured it: the
counterfactual benefit went from +22% to −0.2%).

So the matcher rejects a point when the best candidate on a *different* road is
within 18 m of the best overall. Cost: about 13% of events dropped. Benefit:
road-level accuracy **95.4% → 99.5%**.

### 2. Frequency × severity, not "weighted events"

Counting severity-weighted events in a single Poisson rate conflates two
different processes. One logged accident carries ten times a harsh brake, the
"count" becomes wildly overdispersed, the fitted prior collapses, and the model
degenerates into the raw-history baseline it was meant to beat.

So the model is the standard actuarial decomposition:

* **Frequency** — a clean Gamma-Poisson on event *counts*, a genuine counting process.
* **Severity** — a separately shrunk per-road severity factor.

### 3. Hierarchical Bayesian rate estimation

Each road has a latent event rate `θ` per vehicle-km.

1. **Prior from road structure.** A Poisson gradient-boosting model (blended
   with a GLM so a thin leaf can never collapse the variance) maps curvature,
   junction density, lighting, gradient, speed limit and lanes to an expected
   rate. A road with no history inherits the risk of structurally similar roads
   instead of defaulting to zero.
2. **Empirical Bayes.** `θ ~ Gamma(a, a/m)`, with the dispersion `a` fitted by
   maximising the Negative-Binomial marginal likelihood — the amount of
   shrinkage is learned, not hand-tuned.
3. **Posterior**, in closed form: `θ | data ~ Gamma(a + y, a/m + E)`.

**Plus a correction that most empirical-Bayes implementations skip.** That
textbook posterior treats the fitted prior mean as if it were known exactly. It
is not, and where the prior carries most of the weight — on a sparse network,
most roads — you get a tight interval around a wrong centre. Measured 90%
coverage of the plain version: **0.69**. So the prior's own out-of-fold error
`σ` is estimated and folded in, weighted by how much the prior contributes:

```
CV²(total) = 1/(a + y) + (w · σ)²
```

Coverage after the correction: **0.89**.

### 4. Alternatives as a Pareto frontier

Sweeping a penalty parameter λ over `cost = time + λ · risk` and running
Dijkstra at each setting traces the lower-left boundary of the achievable
(time, risk) set exactly. That is cheaper and more complete than
k-shortest-paths, which tends to return k near-identical detours around the same
corner. The dispatcher sees the frontier, not a single opaque recommendation.

### 5. The trip log makes it falsifiable

Every dispatch is recorded with the score, the interval and the alternatives
that were on screen *at decision time*, plus a reason code when the dispatcher
overrides. Join that to what actually happened and you can ask whether the
scores were right — and a recommendation that gets overridden 80% of the time
is a broken recommendation, with the reason codes to say why.

---

## Evidence

Strict **temporal holdout**: fit on 9 months, score the next 4. Every metric is
reported against at least one baseline, because an AUC with nothing to compare
it to says nothing.

### Ranking (AUC, worst decile of roads by *true* rate)

| | AUC |
|---|---|
| **RouteShield** | **0.961** |
| Road features only (no history) | 0.932 |
| Historical event *rate* | 0.840 |
| Historical event *count* | 0.459 |
| Segment length only | — |

### Where it matters most — roads with the least history

Spearman correlation with the true rate, bottom quartile by exposure:

| | ρ |
|---|---|
| **RouteShield** | **0.789** |
| Raw historical rate | 0.476 |

### Calibration

| | |
|---|---|
| 90% interval coverage, true rate | **89.0%** |
| 90% interval coverage, posterior predictive | **91.8%** |
| Calibration error (mean abs log obs/exp) | 0.085 |
| Confidence badge separation (median \|log\| error) | High 0.10 · Medium 0.15 · Low 0.24 |

### Counterfactual replay

Routes are **chosen** with the model's predictions but **scored** with held-out
observed events, so the system cannot grade its own homework.

| | |
|---|---|
| Trips with a safer alternative inside a +15% time budget | 11.6% |
| True risk reduction on those trips | **22.4%** |
| **Fleet-wide true risk reduction** (averaged over *every* trip) | **4.0%** |
| **Fleet-wide travel-time cost** (same population) | **+1.0%** |

Both fleet-wide figures average over every trip, switched or not. Quoting a
risk reduction across all trips against a time cost measured only on the trips
that switched would overstate the cost several-fold — they are different
populations.

### Performance

| Operation | Time |
|---|---|
| Fit the whole network | ~0.7 s |
| Map-match 43k events | 1.2 s |
| Shortest path | 0.10 ms |
| Score a route (1000 MC draws) | 0.74 ms |
| Full alternatives + Pareto | 1.4 ms |

Segment risk is precomputed in a nightly batch, so request-time cost is a table
lookup plus a sum — it does not grow with event volume.

Reproduce everything: `python -m routeshield.cli demo` writes
`artifacts/results.json`.

---

## Honesty

**No public Indian fleet-telematics dataset exists.** Harsh-braking and
near-miss events with GPS traces are proprietary. So:

* **Road network** — junction coordinates are real Bengaluru locations;
  inter-junction geometry is interpolated, not true OSM polylines. Tagged
  `GEO_ANCHORED_SYNTHETIC` everywhere it surfaces. `make osm` replaces it with
  real OSM geometry and tags, and the OSM ingestion is unit-tested against a
  recorded Overpass fixture.
* **Telematics events** — generated by a labelled synthetic process seeded from
  real structure: road features, the MoRTH severity mix, India-specific night /
  monsoon / fog exposure, and a severity tilt toward high-speed unlit roads.
* **Therefore** every metric above validates the *pipeline and the modelling
  choices*, not real-world accuracy. Real accuracy requires a fleet
  data-sharing agreement. The deployment plan starts in shadow mode precisely
  to measure it.

The context multipliers (night 1.55×, fog 1.60×, rain 1.30×) are
order-of-magnitude values grounded in published Indian road-safety statistics,
deliberately exposed as one editable table rather than buried in code.
`calibrate_from_events()` re-fits them from an operator's own data.

See [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md).

---

## Repository layout

```
routeshield/
  data/      corridor.py (bundled network) · osm.py (live Overpass) · events.py
  core/      matching · risk · hotspots · routing · context · evaluate · analytics · store
  api/       FastAPI service
  pipeline.py, cli.py
web/         static dispatcher console (GitHub Pages)
scripts/     fetch_osm.py
tests/       35 tests
docs/        architecture · data provenance · event schema · deployment · model card
```

## Deployment

Shadow → advisory → scale. See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

Privacy by construction: vehicle IDs hashed at ingest, reports at cohort level
only, driver identity never in a report, routes advisory with override and
reason logging.

## Licence

MIT. OpenStreetMap data fetched by `scripts/fetch_osm.py` is © OpenStreetMap
contributors under ODbL 1.0.
