"""Dispatch integration: the trip log.

Task 6 of the brief, and the piece that makes the whole system falsifiable.
Every dispatch decision is written down with the score that was shown at the
time, the alternatives that were offered, and — when the dispatcher overrides
the recommendation — the reason they gave.

Two things follow from that record:

* **Post-trip comparison.** Join the log to observed events after the fact and
  you can ask whether the scores were right, per route and per period.
* **Override analytics.** A recommendation that dispatchers reject 80% of the
  time is a broken recommendation, and the reason codes say why.

SQLite keeps the prototype dependency-free. The schema maps one-to-one onto
PostGIS in production; only the geometry column type changes.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS route_assignments (
    trip_id            TEXT PRIMARY KEY,
    created_at         TEXT NOT NULL,
    dispatcher_id      TEXT,
    origin_node        INTEGER,
    destination_node   INTEGER,
    origin_name        TEXT,
    destination_name   TEXT,
    depart_hour        INTEGER,
    weather            TEXT,
    driver_cohort      TEXT,
    chosen_route       TEXT NOT NULL,     -- JSON list of segment ids
    chosen_is_recommended INTEGER NOT NULL,
    predicted_risk     REAL NOT NULL,
    risk_lo90          REAL NOT NULL,
    risk_hi90          REAL NOT NULL,
    confidence         TEXT NOT NULL,
    eta_min            REAL NOT NULL,
    distance_km        REAL NOT NULL,
    baseline_risk      REAL,
    baseline_eta_min   REAL,
    alternatives_shown INTEGER DEFAULT 0,
    override_reason    TEXT
);

CREATE TABLE IF NOT EXISTS trip_outcomes (
    trip_id            TEXT PRIMARY KEY REFERENCES route_assignments(trip_id),
    completed_at       TEXT,
    observed_events    REAL,
    observed_severe    INTEGER
);

CREATE INDEX IF NOT EXISTS idx_assign_created ON route_assignments(created_at);
"""

OVERRIDE_REASONS = (
    "customer_time_commitment",
    "vehicle_restriction",
    "local_knowledge",
    "toll_cost",
    "road_closure",
    "other",
)


class TripLog:
    def __init__(self, path: str | Path = "artifacts/routeshield.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # -- writes -----------------------------------------------------------
    def log_assignment(self, trip_id: str, score: dict, edges: list[int], *,
                       dispatcher_id: str = "demo",
                       origin_node: int = -1, destination_node: int = -1,
                       origin_name: str = "", destination_name: str = "",
                       depart_hour: int = 12, weather: str = "clear",
                       driver_cohort: str | None = None,
                       chosen_is_recommended: bool = True,
                       baseline_risk: float | None = None,
                       baseline_eta_min: float | None = None,
                       alternatives_shown: int = 0,
                       override_reason: str | None = None) -> str:
        if override_reason and override_reason not in OVERRIDE_REASONS:
            override_reason = "other"
        self.conn.execute(
            "INSERT OR REPLACE INTO route_assignments VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (trip_id, dt.datetime.now(dt.timezone.utc).isoformat(), dispatcher_id,
             origin_node, destination_node, origin_name, destination_name,
             depart_hour, weather, driver_cohort,
             json.dumps([int(e) for e in edges]), int(chosen_is_recommended),
             float(score["risk_mean"]), float(score["risk_lo"]), float(score["risk_hi"]),
             str(score["confidence"]), float(score["time_min"]), float(score["distance_km"]),
             baseline_risk, baseline_eta_min, int(alternatives_shown), override_reason))
        self.conn.commit()
        return trip_id

    def log_outcome(self, trip_id: str, observed_events: float,
                    observed_severe: int = 0) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO trip_outcomes VALUES (?,?,?,?)",
            (trip_id, dt.datetime.now(dt.timezone.utc).isoformat(),
             float(observed_events), int(observed_severe)))
        self.conn.commit()

    # -- reads ------------------------------------------------------------
    def recent(self, limit: int = 50) -> list[dict]:
        cur = self.conn.execute(
            "SELECT * FROM route_assignments ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in cur.fetchall()]

    def adoption_stats(self) -> dict:
        cur = self.conn.execute(
            "SELECT COUNT(*) n, SUM(chosen_is_recommended) taken, "
            "SUM(COALESCE(baseline_risk,predicted_risk) - predicted_risk) risk_saved, "
            "SUM(eta_min - COALESCE(baseline_eta_min, eta_min)) time_cost "
            "FROM route_assignments")
        r = dict(cur.fetchone())
        n = r["n"] or 0
        return {
            "trips_logged": n,
            "recommendation_adopted": r["taken"] or 0,
            "adoption_rate_pct": round(100 * (r["taken"] or 0) / n, 1) if n else 0.0,
            "cumulative_risk_avoided": round(r["risk_saved"] or 0.0, 3),
            "cumulative_extra_minutes": round(r["time_cost"] or 0.0, 1),
        }

    def override_reasons(self) -> list[dict]:
        cur = self.conn.execute(
            "SELECT override_reason reason, COUNT(*) n FROM route_assignments "
            "WHERE override_reason IS NOT NULL GROUP BY reason ORDER BY n DESC")
        return [dict(r) for r in cur.fetchall()]

    def score_vs_outcome(self) -> list[dict]:
        """The post-trip comparison: predicted risk against what happened."""
        cur = self.conn.execute(
            "SELECT a.trip_id, a.predicted_risk, a.risk_lo90, a.risk_hi90, "
            "a.confidence, o.observed_events, o.observed_severe "
            "FROM route_assignments a JOIN trip_outcomes o USING (trip_id)")
        return [dict(r) for r in cur.fetchall()]

    def close(self) -> None:
        self.conn.close()
