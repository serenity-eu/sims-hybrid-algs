#!/usr/bin/env python3
"""Write MANIFEST.sha256 over every archived artifact, stage by stage.

Regenerate a stage, hash it, compare. The catalogue snapshot cannot be
regenerated (see fetch_catalogue.py); its integrity rests on the checksum.

    python make_manifest.py            # write publication-data/MANIFEST.sha256
    python make_manifest.py --check    # verify the tree against it
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PD = ROOT / "publication-data"
MANIFEST = PD / "MANIFEST.sha256"

# The published benchmark: 5 AOIs x 5 sizes. The instance stages are enumerated
# rather than globbed.
CITIES = ("lagos_nigeria", "mexico_city", "paris", "rio_de_janeiro", "tokyo_bay")
SIZES = (100, 125, 150, 175, 200)
STEMS = tuple(f"{c}_{n}" for c in CITIES for n in SIZES)

# (stage label, paths relative to the repository root, glob)
# pattern None means "the published benchmark only", expanded from STEMS.
STAGES: list[tuple[str, list[str], str | None]] = [
    ("stage 0  frozen catalogue snapshot (not reproducible: live catalogue, live pricing)",
     ["publication-data/satellite-data/scene_catalogue",
      "publication-data/satellite-data/aois"], "*.geojson"),
    ("stage 2  problem instances (deterministic: seed_for(city, N, 0))",
     ["publication-data/satellite-data/benchmark_instances"], None),
    ("stage 3  exact fronts (deterministic; kept where --pseudo-source resolves them)",
     ["sims-core/tests/data/gpbaa_2d_gurobi_hard",
      "sims-core/tests/data/an_2d_gurobi_hard"], None),
    ("stage 4  experiment results (NOT bitwise: 200 s wall-clock budget per run)",
     ["publication-data/biobjective/gpba",
      "publication-data/biobjective/an"], "*.json"),
    ("stage 4  binary traces",
     ["publication-data/biobjective/gpba/traces",
      "publication-data/biobjective/an/traces"], "*.tar.gz"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect() -> list[tuple[str, list[Path]]]:
    out = []
    for label, dirs, pattern in STAGES:
        files: list[Path] = []
        for d in dirs:
            base = ROOT / d
            if not base.exists():
                continue
            if pattern is None:
                for stem in STEMS:
                    files.extend(sorted(q for q in base.glob(f"{stem}.*") if q.is_file()))
            else:
                for p in sorted(base.glob(pattern)):
                    if p.is_file():
                        files.append(p)
        out.append((label, sorted(set(files))))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    stages = collect()

    if args.check:
        if not MANIFEST.exists():
            raise SystemExit(f"no manifest at {MANIFEST}")
        expected = {}
        for line in MANIFEST.read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            digest, _, name = line.partition("  ")
            expected[name] = digest
        bad = missing = 0
        seen = set()
        for _, files in stages:
            for p in files:
                rel = str(p.relative_to(ROOT))
                seen.add(rel)
                if rel not in expected:
                    print(f"UNLISTED  {rel}")
                    missing += 1
                elif sha256(p) != expected[rel]:
                    print(f"MISMATCH  {rel}")
                    bad += 1
        for rel in expected:
            if rel not in seen:
                print(f"ABSENT    {rel}")
                missing += 1
        total = sum(len(f) for _, f in stages)
        print(f"checked {total} files: {bad} mismatched, {missing} unlisted/absent")
        return 1 if (bad or missing) else 0

    lines = [
        "# SHA-256 of every archived artifact, by pipeline stage.",
        "# Verify with: python publication-data/scripts/make_manifest.py --check",
        "# Paths are relative to the repository root.",
        "#",
        "# Stage 0 is a frozen snapshot: the UP42 catalogue is live and its prices",
        "# are live estimates, so the fetch cannot be repeated. Stages 2 and 3 are",
        "# deterministic and should hash identically after regeneration. Stage 4 is",
        "# a 200 s wall-clock budget per run and will NOT hash identically on a",
        "# re-run; rebuild the paper from the stored results instead.",
        "",
    ]
    total = 0
    for label, files in stages:
        lines.append(f"# --- {label} ({len(files)} files)")
        for p in files:
            lines.append(f"{sha256(p)}  {p.relative_to(ROOT)}")
            total += 1
        lines.append("")
    MANIFEST.write_text("\n".join(lines))
    print(f"wrote {MANIFEST} ({total} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
