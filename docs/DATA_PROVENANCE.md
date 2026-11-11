# Data provenance

| Need | Source | Licence | Status in this repo |
|---|---|---|---|
| Road graph, geometry, junctions, lighting, maxspeed, lanes, oneway | OpenStreetMap via Overpass | ODbL 1.0 | **Implemented** (`routeshield/data/osm.py`), fetched by `make osm`; parser unit-tested against a recorded Overpass fixture |
| Bundled offline network | Derived: real junction coordinates + interpolated geometry | MIT (this repo) | **`GEO_ANCHORED_SYNTHETIC`** — see below |
| Accident statistics, severity mix, night-crash share | MoRTH *Road Accidents in India* annual reports; data.gov.in | Government open data | Used to set severity weights and the direction/order of magnitude of context multipliers |
| Weather (current + history) | Open-Meteo | CC-BY 4.0, no key | **Implemented** (`core/context.py`), with a climatology fallback when offline |
| Elevation / gradient | SRTM / Copernicus DEM | Open | Interface present; gradient is a corridor attribute in the bundled network |
| Telematics events (harsh brake/turn/accel, overspeed, near-miss, accident) | **None public** | — | **Synthetic**, clearly labelled |

## The honest part

There is no public Indian fleet-telematics dataset. Harsh-braking and near-miss
events with GPS traces are proprietary to fleet operators and telematics
vendors. Every published number in this repo therefore validates the
**pipeline and the modelling choices**, not real-world accuracy.

### What is real

* Junction coordinates (Bengaluru ORR, Hosur Road, Sarjapur Road, NH44, NH48,
  NH75, NH275, NH648 and the inner arterials) are real-world locations.
* Road classes, speed limits and lighting reflect those corridors' actual
  character.
* Junction density and curvature are computed from the graph and the geometry.
* The OSM ingestion path is real, working code against the live Overpass API.

### What is not

* **Inter-junction polyline geometry** is interpolated (great-circle with a
  proportional bow on ring roads and a gentle meander on long legs), not true
  OSM ways. `make osm` replaces it.
* **Telematics events** come from `routeshield/data/events.py`. The generator
  is seeded from real structure — road features, the MoRTH severity mix, a
  28% night-driving share at a 1.55× multiplier, monsoon and fog seasonality,
  and a severity tilt toward high-speed unlit roads — but it is synthetic.
* **Fleet exposure** (vehicle-km per segment) is drawn heavy-tailed on purpose,
  because a real fleet concentrates on a few lanes and the sparse-data regime is
  what the Bayesian treatment exists for.

### Dropping real data in

`load_events_csv()` reads the schema in `EVENT_SCHEMA.md`. Nothing else changes.
The evaluation module computes every metric that is available without ground
truth (`predictive_interval_coverage_90`, calibration, ranking against observed
next-period events); the ground-truth metrics simply stop being reported.

Verify each dataset's current licence and access terms before use. Any derived
segment-risk table redistributed publicly inherits OSM's ODbL.
