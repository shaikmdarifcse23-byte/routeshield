"""Fetch a real OpenStreetMap road graph and cache it for RouteShield.

    python scripts/fetch_osm.py --city bengaluru --out artifacts/osm_ways.json

Then point the pipeline at it:

    ROUTESHIELD_OSM=artifacts/osm_ways.json python -m routeshield.cli demo

Data: (c) OpenStreetMap contributors, ODbL 1.0. Be a good citizen of the
public Overpass instances: this makes one request, caches the result, and
never runs in a loop.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from routeshield.data.osm import CITY_BBOX, build_from_osm, fetch_overpass, parse_overpass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", default="bengaluru", choices=sorted(CITY_BBOX))
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("S", "W", "N", "E"),
                    help="explicit bounding box, overrides --city")
    ap.add_argument("--out", default="artifacts/osm_ways.json")
    args = ap.parse_args()

    bbox = tuple(args.bbox) if args.bbox else CITY_BBOX[args.city]
    print(f"Querying Overpass for {args.city} {bbox} ...", flush=True)
    payload = fetch_overpass(bbox)

    ways = parse_overpass(payload)
    print(f"  {len(ways)} drivable ways parsed")
    if not ways:
        print("No ways returned. Check the bounding box.", file=sys.stderr)
        return 1

    net = build_from_osm(ways)
    print(f"  -> {len(net['u'])} directed segments, {net['n_nodes']} nodes, "
          f"{net['length'].sum():.1f} km")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "place": f"OpenStreetMap: {args.city} {bbox}",
        "licence": "(c) OpenStreetMap contributors, ODbL 1.0",
        "bbox": list(bbox),
        "ways": [{k: v for k, v in w.items() if k != "tags"} for w in ways],
    }))
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print("\nNext:  ROUTESHIELD_OSM=%s python -m routeshield.cli demo" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
