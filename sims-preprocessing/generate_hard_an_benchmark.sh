#!/usr/bin/env bash
# Generate the A&N-calibrated SIMS benchmark ("instances_hard_an").
#
# Design target (see paper/ppsn2026/instance_generation_difficulty.md, Part II):
#   * Aneja & Nair returns 5-15 solutions at the 120 s experiment budget --
#     the exact phase works but does not exhaust the front;
#   * PLS improves substantially on the exact-only result.
#
# A&N is a stronger exact solver on this problem than GPBA-A, so this set is
# markedly harder than the GPBA-calibrated one in Part I: the usable fragment
# window is ~13k-27k rather than ~7k-14k.
#
# All five cities share ONE size ladder (200-275 step 25): a complete 5x4 grid,
# no instance skipped, one timeout per size. The ladder is bounded on both sides
# by measurement: paris only enters the 5-15 A&N window at N >= 200 (below that
# it enumerates 23-24 points), while lagos/rio/mexico collapse to 2-3 solutions
# by N = 300.
#
# Pools are the UNION of instances_pool and biobjective_fetch (see
# merge_pools.py): two partially-overlapping UP42 fetches. The union adds 742
# images for tokyo_bay, 384 for rio_de_janeiro, 182 for paris. That is what lets
# tokyo_bay keep cap 0.5 / band 20-80 -- the setting with the right difficulty --
# which its original 218-image pool could not sustain past N = 200.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SD="$ROOT/publication-data/satellite-data"
BIN="$ROOT/sims-preprocessing/target/release/generate_random_clouds"
OUT="${1:-$SD/instances_hard_an}"
# Optional ladder override, e.g. 100,125,150,175,200 for the 25-instance set
# behind the biobjective paper. Empty keeps each city's calibrated ladder.
SIZES_OVERRIDE="${2:-}"

[[ -x "$BIN" ]] || (cd "$ROOT/sims-preprocessing" && cargo build --release --bin generate_random_clouds)

# city -> cap  cloud-band  cloud-blobs  sizes
# city | pool | cap | cloud-band | cloud-blobs | sizes
#
# POOL is per city and deliberate:
#   merged = instances_pool U biobjective_fetch (two partially-overlapping UP42
#            fetches; +742 images for tokyo_bay, +384 rio, +182 paris)
#   base   = instances_pool only
#
# tokyo_bay REQUIRES merged: its cap 0.5 / band 20-80 setting is the one with the
# right difficulty, and the base pool's 218 surviving images cannot reach N=275.
#
# paris REQUIRES base: on the merged pool it returns 17 A&N solutions at N=200 and
# no cap fixes it (0.30 -> 17, 0.25 -> 17, 0.20 -> 25 — tighter caps make it
# EASIER because the count is set by the front's supported-point structure, which
# the extra imagery enriches). On the base pool cap 0.30 gives 11/13/12.
# UNIFORM configuration: one cap, one cloud band, one cloud model for every
# city -- only N varies. This is the defensible form: a per-city difference is
# then a property of the city, not of how its instances were generated.
#
# The single deviation is rio_de_janeiro's cap. Two patches of its AOI (89.6 km2
# and 7.9 km2) are covered ONLY by images with footprints >= 0.72x the AOI
# (median 1.6x), so cap 0.5 leaves them uncovered and validate() rejects the
# instance. No fetch fixes this -- the in-band scenes over those patches are all
# large-footprint. cap 1.0 is the tightest that covers them.
#
# Reaching a shared 10-90 band needed three targeted catalogue fetches, all
# priced against the full AOI: paris coverage gaps (+32 images), rio coverage
# gaps (+13), and mexico_city in-band imagery (+49, via a year-sliced search --
# the catalogue endpoint returns at most 400 scenes per call, so a single query
# saw only a fraction of what exists).
CFG=(
  "lagos_nigeria   merged 0.5 10,90 0 200,225,250,275"
  "paris           merged 0.5 10,90 0 200,225,250,275"
  "rio_de_janeiro  merged 1.0 10,90 0 200,225,250,275"
  "tokyo_bay       merged 0.5 10,90 0 200,225,250,275"
  "mexico_city     merged 0.5 10,90 0 200,225,250,275"
)

mkdir -p "$OUT"
for line in "${CFG[@]}"; do
  read -r city pool cap band blobs sizes <<<"$line"
  [ -n "$SIZES_OVERRIDE" ] && sizes="$SIZES_OVERRIDE"
  pooldir=$([ "$pool" = base ] && echo instances_pool || echo instances_pool_merged)
  echo "=== $city ($pool pool, cap $cap, band $band, blobs $blobs, N=$sizes)"
  "$BIN" \
    "$SD/$pooldir/${city}_image_set.geojson" \
    "$SD/aois/${city}.geojson" \
    "$OUT" "$city" \
    --sizes "$sizes" \
    --max-img-frac "$cap" \
    --cloud-band "$band" \
    --cloud-blobs "$blobs" \
    --grid-long 110 \
    --keep-frac 0.9
done
echo "instances in $OUT"

# The 25-instance set behind the biobjective paper is the same configuration
# on a lower ladder; the seed is derived from (city, size, --seed) and --seed
# defaults to 0, so this reproduces those instances byte for byte:
#
#   ./generate_hard_an_benchmark.sh "$SD/instances_hard_an" 100,125,150,175,200

