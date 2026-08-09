#!/usr/bin/env bash
#
# Fetch and prepare the official Gurobi runtime for building augmecon-rs with
# `--features gurobi`, so you never have to hand-export GUROBI_HOME / rpath.
#
# It downloads the official linux64 tarball from packages.gurobi.com, extracts
# it into a gitignored cache (.gurobi/), and writes a gitignored
# .cargo/config.toml that sets GUROBI_HOME and bakes a runtime rpath. After
# running it once, `cargo build --release --features gurobi` (and running the
# binary) work with no further environment setup.
#
# It does NOT install a license — you still need your own Gurobi license to run
# a solve. It only removes the install/env-var step.
#
# Usage:
#   scripts/fetch-gurobi.sh              # default version (13.0.2)
#   scripts/fetch-gurobi.sh 13.0.2       # explicit version
#   GUROBI_CACHE=/some/dir scripts/fetch-gurobi.sh   # custom cache location
#
set -euo pipefail

VERSION="${1:-13.0.2}"

# Split X.Y.Z
if [[ ! "$VERSION" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
    echo "error: version must look like 13.0.2 (got '$VERSION')" >&2
    exit 2
fi
MAJOR="${BASH_REMATCH[1]}"
MINOR="${BASH_REMATCH[2]}"
PATCH="${BASH_REMATCH[3]}"

# Naming conventions used by Gurobi's distribution:
#   download dir on server : <MAJOR>.<MINOR>            e.g. 13.0
#   tarball                : gurobi<MAJOR>.<MINOR>.<PATCH>_linux64.tar.gz
#   extracted top dir      : gurobi<MAJOR><MINOR><PATCH>  e.g. gurobi1302
#   shared library         : libgurobi<MAJOR><MINOR>.so   e.g. libgurobi130.so
SERVER_DIR="${MAJOR}.${MINOR}"
TARBALL="gurobi${VERSION}_linux64.tar.gz"
URL="https://packages.gurobi.com/${SERVER_DIR}/${TARBALL}"
TOPDIR="gurobi${MAJOR}${MINOR}${PATCH}"
LIBNAME="gurobi${MAJOR}${MINOR}"

# Repo root = parent of this script's directory.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="${GUROBI_CACHE:-$ROOT/.gurobi}"
GUROBI_HOME="$CACHE/$TOPDIR/linux64"
LIBDIR="$GUROBI_HOME/lib"

write_cargo_config() {
    local cfg="$ROOT/.cargo/config.toml"
    mkdir -p "$ROOT/.cargo"
    local begin="# >>> gurobi (managed by scripts/fetch-gurobi.sh) >>>"
    local end="# <<< gurobi <<<"
    local block
    block="$(cat <<EOF
$begin
[env]
GUROBI_HOME = "$GUROBI_HOME"

[target.x86_64-unknown-linux-gnu]
rustflags = ["-C", "link-args=-Wl,-rpath,$LIBDIR"]
$end
EOF
)"
    if [[ -f "$cfg" ]] && grep -qF "$begin" "$cfg"; then
        # Replace the existing managed block in place.
        local tmp; tmp="$(mktemp)"
        awk -v b="$begin" -v e="$end" -v repl="$block" '
            $0==b {print repl; skip=1; next}
            skip && $0==e {skip=0; next}
            !skip {print}
        ' "$cfg" > "$tmp"
        mv "$tmp" "$cfg"
    else
        # Append (preserving any existing, non-managed content).
        [[ -f "$cfg" ]] && printf '\n' >> "$cfg"
        printf '%s\n' "$block" >> "$cfg"
    fi
    echo "wrote GUROBI_HOME + rpath to $cfg"
}

if [[ -f "$LIBDIR/lib${LIBNAME}.so" ]]; then
    echo "Gurobi $VERSION already present at $GUROBI_HOME — skipping download."
    write_cargo_config
    exit 0
fi

echo "Fetching Gurobi $VERSION from $URL"
mkdir -p "$CACHE"
tmp_tar="$(mktemp "${TMPDIR:-/tmp}/gurobi.XXXXXX.tar.gz")"
trap 'rm -f "$tmp_tar"' EXIT

# The distribution server publishes the tarball's MD5 in a response header;
# grab it first so we can verify integrity after download.
expected_md5=""
if command -v curl >/dev/null 2>&1; then
    expected_md5="$(curl -fsSI "$URL" 2>/dev/null \
        | tr -d '\r' | awk -F': ' 'tolower($1)=="x-amz-meta-checksum_md5"{print $2}')"
    curl -fL --retry 3 -o "$tmp_tar" "$URL"
elif command -v wget >/dev/null 2>&1; then
    wget -O "$tmp_tar" "$URL"
else
    echo "error: need curl or wget to download" >&2
    exit 1
fi

if [[ -n "$expected_md5" ]] && command -v md5sum >/dev/null 2>&1; then
    got_md5="$(md5sum "$tmp_tar" | awk '{print $1}')"
    if [[ "$got_md5" != "$expected_md5" ]]; then
        echo "error: MD5 mismatch (expected $expected_md5, got $got_md5) — download corrupt" >&2
        exit 1
    fi
    echo "MD5 verified ($got_md5)"
fi

echo "Extracting into $CACHE"
tar -xzf "$tmp_tar" -C "$CACHE"

if [[ ! -f "$LIBDIR/lib${LIBNAME}.so" ]]; then
    echo "error: expected $LIBDIR/lib${LIBNAME}.so after extraction; layout changed?" >&2
    ls -la "$LIBDIR" 2>/dev/null || true
    exit 1
fi

write_cargo_config
echo
echo "Done. Build with:"
echo "    cargo build --release --features gurobi   # from augmecon-rs/ or with -p augmecon"
echo "(You still need a valid Gurobi license to run a solve.)"
