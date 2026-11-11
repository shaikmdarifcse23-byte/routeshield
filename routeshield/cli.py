"""Command-line entry point.

    python -m routeshield.cli demo        # full pipeline + evaluation + artifacts
    python -m routeshield.cli serve       # FastAPI dispatcher service
    python -m routeshield.cli evaluate    # metrics only, to stdout
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def cmd_demo(args) -> int:
    from . import pipeline
    osm = args.osm or os.environ.get("ROUTESHIELD_OSM")
    print("Running pipeline ...", flush=True)
    p = pipeline.run(train_months=args.train_months, test_months=args.test_months,
                     prior=args.prior, osm_path=osm)
    print(f"  network   : {len(p.net['u'])} segments, {p.model.n_roads} roads, "
          f"{p.net['length'].sum():.0f} km  [{p.net['source']}]")
    print(f"  events    : {len(p.train):,} train / {len(p.test):,} test")
    print(f"  matched   : {p.match_stats['match_rate']:.1%} "
          f"(road accuracy {p.match_stats.get('road_match_accuracy', float('nan')):.1%})")
    print(f"  hotspots  : {len(p.hotspots)}")
    print("Evaluating ...", flush=True)
    res = pipeline.full_evaluation(p, n_trips=args.trips)
    pipeline.export_artifacts(p, res, args.out)

    web = Path("web/data")
    if Path("web/index.html").exists():
        web.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(args.out) / "network.json", web / "network.json")
        print(f"  demo data -> {web / 'network.json'}")

    sm = res["segment_model"]
    print("\n--- headline (held-out, see artifacts/results.json for all of it) ---")
    print(f"  rank top-decile TRUE rate  AUC : {sm['roc_auc_top_decile_TRUE_rate']['routeshield']:.3f}"
          f"   (raw history {sm['roc_auc_top_decile_TRUE_rate']['raw_history_rate']:.3f})")
    print(f"  sparse roads, Spearman vs true : {sm['spearman_vs_true_rate_sparse_only']['routeshield']:.3f}"
          f"   (raw history {sm['spearman_vs_true_rate_sparse_only']['raw_history_rate']:.3f})")
    print(f"  90% interval coverage          : {sm['latent_rate_interval_coverage_90']:.1%}")
    cf = res["counterfactual"]
    print(f"  trips with a safer alternative : {cf['pct_with_safer_alternative']:.1f}%")
    print(f"  true risk cut when switched    : {cf.get('avg_true_risk_reduction_pct_when_switched', 0):.1f}%"
          f" for +{cf['avg_time_increase_pct_when_switched']:.1f}% time")
    print(f"\nWrote {args.out}/network.json and {args.out}/results.json")
    return 0


def cmd_evaluate(args) -> int:
    from . import pipeline
    p = pipeline.run(osm_path=args.osm or os.environ.get("ROUTESHIELD_OSM"))
    print(json.dumps(pipeline.full_evaluation(p, n_trips=args.trips), indent=2))
    return 0


def cmd_serve(args) -> int:
    import uvicorn
    from .api.server import app
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="routeshield", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run the pipeline and write artifacts")
    d.add_argument("--out", default="artifacts")
    d.add_argument("--trips", type=int, default=500)
    d.add_argument("--train-months", type=float, default=9.0)
    d.add_argument("--test-months", type=float, default=4.0)
    d.add_argument("--prior", choices=["gbm", "glm"], default="gbm")
    d.add_argument("--osm", default=None, help="path to a cached OSM ways file")
    d.set_defaults(func=cmd_demo)

    e = sub.add_parser("evaluate", help="print the evaluation as JSON")
    e.add_argument("--trips", type=int, default=400)
    e.add_argument("--osm", default=None)
    e.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("serve", help="run the FastAPI dispatcher service")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
