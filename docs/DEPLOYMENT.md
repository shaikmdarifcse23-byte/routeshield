# Deployment plan

## Phase 1 — Shadow (weeks 1–6, one corridor)

Scores are computed and logged; **nothing is shown to dispatchers**. Each night
the batch runs; each week predicted risk is compared against observed events on
the routes that were actually driven.

*Exit criteria:* posterior-predictive 90% interval coverage within 85–95%;
ranking AUC against observed next-period events above the operator's current
baseline; map-match road accuracy above 98%.

This phase exists because every number in this repo is measured on synthetic
events. Shadow mode is where real accuracy is established.

## Phase 2 — Advisory (weeks 7–16)

Dispatchers see the score, the interval, the explanation and the alternatives.
The safer option is recommended but never forced; overrides require a reason
code and are logged.

*Watch:* adoption rate, override reasons, and the score-vs-outcome join. An
option rejected 80% of the time is a broken recommendation, not a stubborn
dispatcher — the reason codes say which.

*Exit criteria:* adoption above 40%, no increase in SLA breaches, measurable
reduction in risk-weighted exposure on the pilot corridor.

## Phase 3 — Scale

Multi-region rollout, nightly retraining, drift monitoring on feature
distributions and calibration, cohort reports to the safety team, and the
segment-risk table published as vector tiles.

## Stack

| Concern | Prototype | Production |
|---|---|---|
| Graph + features | in-process numpy | PostGIS + OSM extracts |
| Batch | `pipeline.run()` | Airflow / Dagster nightly |
| Serving | FastAPI | FastAPI behind a gateway, read replica |
| Trip log | SQLite | PostgreSQL |
| Map | canvas + XYZ tiles | MapLibre + self-hosted vector tiles |

## Monitoring

Calibration drift (coverage per week), feature drift, map-match accuracy and
rejection rate, adoption and override mix, and the score-vs-outcome residual.
If coverage leaves 85–95%, the model is no longer honest about uncertainty and
the badge must be withdrawn until it is refitted.

## Governance and privacy

* Vehicle IDs hashed at ingest; driver identity never enters a report.
* Reports at cohort level only — the system is a road-risk tool, not a driver
  surveillance tool, and is explicitly not to be used for individual discipline.
* Routes are advisory; the dispatcher is accountable and can always override.
* Every score shown is logged with the interval and the alternatives, so any
  decision can be reconstructed after the fact.
