# Telematics event schema

Drop-in format for real fleet data. `routeshield.data.events.load_events_csv`
reads exactly this.

| column | type | required | notes |
|---|---|---|---|
| `event_id` | string | yes | unique |
| `ts` | ISO-8601 | yes | event time, UTC or with offset |
| `lat`, `lon` | float | yes | WGS84, raw GPS — map matching is our job |
| `type` | enum | yes | `harsh_accel`, `harsh_brake`, `harsh_turn`, `overspeed`, `near_miss`, `accident` |
| `severity` | float 0–1 | no | defaults to the type's weight |
| `vehicle_id` | string | yes | **hashed at ingest** |
| `driver_cohort` | string | no | tenure band, e.g. `tenure_1_3y` — never a personal identifier |
| `speed_kmh` | float | no | |
| `trip_id` | string | no | links to the dispatch log |

## Exposure is not optional

Rates need a denominator. Without vehicle-km per segment the system can only
count events, which is the failure mode it exists to fix. Supply either:

* `trips.csv` — `trip_id, start_ts, end_ts, edge_sequence` (post-matched), or
* `traces.csv` — `trip_id, ts, lat, lon` at ≥ 0.2 Hz, which we map-match.

## Dispatch log (written by RouteShield)

`route_assignments`: `trip_id, created_at, dispatcher_id, origin_node,
destination_node, origin_name, destination_name, depart_hour, weather,
driver_cohort, chosen_route, chosen_is_recommended, predicted_risk, risk_lo90,
risk_hi90, confidence, eta_min, distance_km, baseline_risk, baseline_eta_min,
alternatives_shown, override_reason`

`trip_outcomes`: `trip_id, completed_at, observed_events, observed_severe`

The join of those two tables is what makes the system falsifiable.

## Privacy

Vehicle identifiers are hashed at ingest. Reports aggregate to cohort level and
driver identity never enters one. Routes are advisory: a dispatcher can always
override, and the override is logged with a reason code rather than silently
discarded.
