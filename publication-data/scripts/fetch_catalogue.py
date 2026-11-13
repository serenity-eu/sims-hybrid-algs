#!/usr/bin/env python3
"""Fetch a UP42 catalogue pool for one of the paper's areas of interest.

This does NOT reproduce the archived snapshot in
``satellite-data/scene_catalogue/`` and cannot: the UP42 catalogue is
live, and ``cost`` comes from ``/v2/orders/estimate``, which is a live price.
A search run today returns a different scene set at different prices. The
archived snapshot is the root of the reproducibility chain; this script builds
a comparable pool of your own.

Output schema matches the snapshot: the five fields the instance generator
reads, plus ``collection`` and ``acquisitionDate``, which carry the Airbus
attribution (``phr`` -> "(C) CNES <year>, Distribution Airbus DS"; ``pneo`` ->
"(C) Airbus DS <year>").

Credentials are read from ``UP42_USERNAME`` and ``UP42_PASSWORD``.

Usage:

    python fetch_catalogue.py --city lagos_nigeria --out pool.geojson
    python fetch_catalogue.py --city paris --max-images 2000 --with-costs --out paris.geojson
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AOIS = ROOT / "publication-data" / "satellite-data" / "aois"

CITIES = ("lagos_nigeria", "mexico_city", "paris", "rio_de_janeiro", "tokyo_bay")

# The fields the instance generator reads, plus what attribution needs.
# generate_random_clouds.rs consumes exactly: geometry, cost, resolution,
# cloudCoverage, providerProperties.incidenceAngle. collection and
# acquisitionDate carry the Airbus attribution (phr -> "(C) CNES <year>,
# Distribution Airbus DS"; pneo -> "(C) Airbus DS <year>").
SLIM_FIELDS = (
    "id",
    "collection",
    "providerName",
    "acquisitionDate",
    "cost",
    "resolution",
    "cloudCoverage",
)


def slim(feature: dict) -> dict:
    """Reduce a UP42 catalogue feature to the archived schema."""
    p = feature["properties"]
    pp = p.get("providerProperties") or {}
    if isinstance(pp, str):
        try:
            pp = json.loads(pp)
        except json.JSONDecodeError:
            pp = {}
    props = {k: p.get(k) for k in SLIM_FIELDS}
    props["providerProperties"] = {"incidenceAngle": float(pp.get("incidenceAngle") or 0.0)}
    return {"type": "Feature", "geometry": feature["geometry"], "properties": props}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", choices=CITIES, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-images", type=int, default=2000)
    ap.add_argument("--start-date", default="2021-01-01")
    ap.add_argument("--end-date", default=None, help="default: today")
    ap.add_argument(
        "--with-costs",
        action="store_true",
        help="also call /v2/orders/estimate per scene (free, but one call per scene)",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    aoi_path = AOIS / f"{args.city}.geojson"
    if not aoi_path.exists():
        raise SystemExit(f"no AOI for {args.city}: {aoi_path}")

    import geopandas

    from sims.core.data_providers import up42_provider
    from sims.core.data_providers.up42_provider import CollectionName, SearchParameters

    aoi_gdf = geopandas.read_file(aoi_path)
    params = SearchParameters(
        start_date=datetime.strptime(args.start_date, "%Y-%m-%d"),
        end_date=datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else datetime.now(),
        collections=[CollectionName.PLEYADES_HIGH_RES, CollectionName.PLEIADES_NEO, CollectionName.SPOT],
        max_image_count=args.max_images,
    )

    print(f"searching {args.city} ({args.start_date} .. {args.end_date or 'today'}), limit {args.max_images}")
    images_gdf = up42_provider.fetch(aoi_gdf, params, with_costs=args.with_costs)
    print(f"{len(images_gdf)} scenes returned")

    fc = json.loads(images_gdf.to_json())
    fc["features"] = [slim(f) for f in fc["features"]]
    args.out.write_text(json.dumps(fc, separators=(",", ":")))
    print(f"wrote {args.out}")
    print(
        "NOTE: this is a fresh pool, not the archived snapshot. Costs are live "
        "estimates as of today and the scene set will differ from the paper's."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
