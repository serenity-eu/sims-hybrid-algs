#!/usr/bin/env python3
"""Fetch Airbus quicklooks for the scenes an instance was built from.

Each ``<instance>.meta.json`` carries a ``scene_ids`` list in the same order as
the image arrays in the matching ``.dzn``: image ``j`` of the instance is
``scene_ids[j]``. Every scene is hosted by ``oneatlas``:

    GET /catalog/oneatlas/image/{scene_id}/quicklook

Quicklooks and thumbnails are previews: free, low resolution, no order. Full
data products require a paid order, and the ``cost`` recorded per scene is a
snapshot of a live estimate rather than a current price. Nothing in the paper
requires the imagery: instance clouds are synthetic, generated from the
recorded seed.

Credentials are read from ``UP42_USERNAME`` and ``UP42_PASSWORD``.

Usage:

    python fetch_quicklooks.py --instance lagos_nigeria_150 --out-dir ./quicklooks
    python fetch_quicklooks.py --all --out-dir ./quicklooks
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INSTANCES = ROOT / "publication-data" / "satellite-data" / "benchmark_instances"

# Every scene in the pool is hosted by oneatlas; see the manifest.
HOST = "oneatlas"

# The published benchmark: 5 AOIs x 5 sizes, enumerated rather than globbed.
CITIES = ("lagos_nigeria", "mexico_city", "paris", "rio_de_janeiro", "tokyo_bay")
SIZES = (100, 125, 150, 175, 200)


def scene_ids_for(meta_path: Path) -> list[str]:
    meta = json.loads(meta_path.read_text())
    ids = meta.get("scene_ids")
    if not ids:
        raise SystemExit(
            f"{meta_path.name} has no 'scene_ids'; regenerate the instances "
            f"with sims-preprocessing/generate_benchmark_instances.sh."
        )
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--instance", help="instance stem, e.g. lagos_nigeria_150")
    g.add_argument("--all", action="store_true", help="every instance in the benchmark")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--kind", choices=("quicklook", "thumbnail"), default="quicklook")
    ap.add_argument("--dry-run", action="store_true", help="list scene ids, fetch nothing")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.all:
        metas = [INSTANCES / f"{c}_{n}.meta.json" for c in CITIES for n in SIZES]
        missing = [m.name for m in metas if not m.exists()]
        if missing:
            raise SystemExit(f"missing instance metadata: {', '.join(missing)}")
    else:
        metas = [INSTANCES / f"{args.instance}.meta.json"]
        if not metas[0].exists():
            raise SystemExit(f"no such instance: {metas[0]}")

    ids: list[str] = []
    for m in metas:
        ids.extend(scene_ids_for(m))
    unique = sorted(set(ids))
    print(f"{len(metas)} instance(s), {len(ids)} image slots, {len(unique)} distinct scenes")

    if args.dry_run:
        for i in unique:
            print(i)
        return 0

    # Imported late so --dry-run works without the sims-core extras installed.
    from sims.core.data_providers import up42_provider

    saved = up42_provider.fetch_previews(unique, HOST, args.out_dir, kind=args.kind)
    missing = len(unique) - len(saved)
    print(f"saved {len(saved)}/{len(unique)} {args.kind}s to {args.out_dir}")
    if missing:
        print(f"{missing} scene(s) had no {args.kind}; UP42 does not provide one for every scene.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
