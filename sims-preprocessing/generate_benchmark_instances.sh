#!/usr/bin/env bash
# Generate the 25 problem instances behind the biobjective SIMS paper:
# 5 areas of interest x {100,125,150,175,200} images.
#
# Input   publication-data/satellite-data/scene_catalogue/<city>_image_set.geojson
#         publication-data/satellite-data/aois/<city>.geojson
# Output  <out-dir>/<city>_<N>.dzn and <city>_<N>.meta.json
#
# Deterministic: the seed is derived from (city, N, --seed) and --seed defaults
# to 0, so a bare run reproduces the published instances byte for byte.
#
#   ./generate_benchmark_instances.sh [out-dir] [sizes]
#
# out-dir defaults to publication-data/satellite-data/benchmark_instances.
# sizes overrides the ladder, e.g. 225,250,275,300. No published result uses
# any size outside the default ladder.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SD="$ROOT/publication-data/satellite-data"
BIN="$ROOT/sims-preprocessing/target/release/generate_random_clouds"
OUT="${1:-$SD/benchmark_instances}"
SIZES_OVERRIDE="${2:-}"

[[ -x "$BIN" ]] || (cd "$ROOT/sims-preprocessing" && cargo build --release --bin generate_random_clouds)

# Generation parameters per city:
#   pool | max-img-frac | cloud-band | cloud-blobs | sizes
#
# max-img-frac caps a footprint as a fraction of the AOI. rio_de_janeiro uses
# 1.0 because two patches of its AOI are covered only by footprints >= 0.72x
# the AOI; a tighter cap leaves them uncovered and the instance is rejected.
CFG=(
  "lagos_nigeria   merged 0.5 10,90 0 100,125,150,175,200"
  "paris           merged 0.5 10,90 0 100,125,150,175,200"
  "rio_de_janeiro  merged 1.0 10,90 0 100,125,150,175,200"
  "tokyo_bay       merged 0.5 10,90 0 100,125,150,175,200"
  "mexico_city     merged 0.5 10,90 0 100,125,150,175,200"
)

mkdir -p "$OUT"
for line in "${CFG[@]}"; do
  read -r city pool cap band blobs sizes <<<"$line"
  [ -n "$SIZES_OVERRIDE" ] && sizes="$SIZES_OVERRIDE"
  pooldir=$([ "$pool" = base ] && echo instances_pool || echo scene_catalogue)
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
