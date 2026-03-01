#!/usr/bin/env bash
# Extract objectives.bin from a trace .tar.gz archive and place it in benches/data/
#
# Usage:
#   ./benches/extract_trace.sh <trace_archive.tar.gz> <instance_name>
#
# Example:
#   ./benches/extract_trace.sh /path/to/lagos_nigeria_100_trace.tar.gz lagos_nigeria_100
#
# This will create benches/data/lagos_nigeria_100.bin

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <trace_archive.tar.gz> <instance_name>"
    echo "Example: $0 /path/to/trace.tar.gz lagos_nigeria_100"
    exit 1
fi

ARCHIVE="$1"
INSTANCE_NAME="$2"
OUTPUT="$DATA_DIR/${INSTANCE_NAME}.bin"

mkdir -p "$DATA_DIR"

tar xzf "$ARCHIVE" objectives.bin -O > "$OUTPUT"

SIZE=$(stat --format=%s "$OUTPUT" 2>/dev/null || stat -f%z "$OUTPUT" 2>/dev/null)
NUM_SOLUTIONS=$((SIZE / 32))  # 4 objectives x 8 bytes each

echo "Extracted $OUTPUT ($NUM_SOLUTIONS solutions, $SIZE bytes)"
