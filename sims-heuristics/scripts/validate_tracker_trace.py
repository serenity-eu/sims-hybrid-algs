#!/usr/bin/env python3
"""Validate tracker trace consistency.

This checks that a u16-encoded tracker trace is a *valid* sequence relative to a single
selected-set state machine.

Trace encoding (little-endian u16):
  record = (op << 12) | image_index
  op:
    0 TrackAdd
    1 TrackRem
    2 PeekAdd
    3 PeekRem
        4 Reset  (clear selected-set state)

By default we validate TrackAdd/TrackRem strictly and also validate PeekAdd/PeekRem
consistency (PeekAdd requires not selected; PeekRem requires selected).

Run (from repo root):
  uv run --no-project python sims-heuristics/scripts/validate_tracker_trace.py \
    --instance sims-heuristics/data/lagos_nigeria_30.dzn \
    --trace sims-heuristics/benches/data/debug_lagos_30_tracker_calls.u16
"""

from __future__ import annotations

import argparse
import os
import re
import struct
import sys
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable


class Op(IntEnum):
    TrackAdd = 0
    TrackRem = 1
    PeekAdd = 2
    PeekRem = 3
    Reset = 4


@dataclass(frozen=True)
class Violation:
    pos: int
    op: Op
    idx: int
    reason: str


_NUM_IMAGES_RE = re.compile(r"^\s*num_images\s*=\s*(\d+)\s*;\s*$")


def parse_num_images_from_dzn(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = _NUM_IMAGES_RE.match(line)
            if m:
                return int(m.group(1))
    raise RuntimeError(f"failed to find 'num_images = ...;' in {path}")


def iter_trace_records_u16(path: str) -> Iterable[int]:
    with open(path, "rb") as f:
        data = f.read()
    if len(data) % 2 != 0:
        raise RuntimeError(f"trace length must be even bytes, got {len(data)}")

    # Fast unpack; returns tuple[int,...]
    # '<' little-endian, 'H' u16
    return struct.iter_unpack("<H", data)


def decode_record_u16(record: int) -> tuple[Op, int]:
    op_raw = (record >> 12) & 0xF
    try:
        op = Op(op_raw)
    except ValueError:
        raise RuntimeError(f"unknown op {op_raw} in record 0x{record:04x}")
    idx = record & 0x0FFF
    return op, idx


def validate_trace(
    *,
    trace_path: str,
    num_images: int,
    strict_peek: bool,
    reset_on_duplicate_add: bool,
    max_violations: int,
    max_records: int | None,
) -> list[Violation]:
    selected = bytearray(num_images)  # 0/1

    inferred_resets = 0
    first_reset_at: int | None = None

    violations: list[Violation] = []
    total = 0

    with open(trace_path, "rb") as f:
        while True:
            if max_records is not None and total >= max_records:
                break

            chunk = f.read(2)
            if not chunk:
                break
            if len(chunk) != 2:
                violations.append(
                    Violation(total, Op.TrackAdd, -1, f"truncated record at byte offset {total * 2}")
                )
                break

            (record,) = struct.unpack("<H", chunk)
            op, idx = decode_record_u16(record)

            if idx >= num_images:
                violations.append(Violation(total, op, idx, f"index out of range (num_images={num_images})"))
                if len(violations) >= max_violations:
                    break
                total += 1
                continue

            is_sel = selected[idx] == 1

            if op == Op.TrackAdd:
                if is_sel:
                    if reset_on_duplicate_add:
                        inferred_resets += 1
                        if first_reset_at is None:
                            first_reset_at = total
                        selected = bytearray(num_images)
                        selected[idx] = 1
                    else:
                        violations.append(Violation(total, op, idx, "TrackAdd but image already selected"))
                else:
                    selected[idx] = 1
            elif op == Op.TrackRem:
                if not is_sel:
                    violations.append(Violation(total, op, idx, "TrackRem but image not selected"))
                else:
                    selected[idx] = 0
            elif op == Op.PeekAdd:
                if strict_peek and is_sel:
                    violations.append(Violation(total, op, idx, "PeekAdd but image already selected"))
            elif op == Op.PeekRem:
                if strict_peek and not is_sel:
                    violations.append(Violation(total, op, idx, "PeekRem but image not selected"))
            elif op == Op.Reset:
                selected = bytearray(num_images)
            else:
                violations.append(Violation(total, op, idx, "unknown op"))

            if len(violations) >= max_violations:
                break

            total += 1

    # If we're in reset mode, append a summary as a synthetic violation on failure-free runs.
    # (We avoid printing from inside validate_trace so the caller controls output.)
    if not violations and reset_on_duplicate_add:
        # Encode summary using a Violation-like struct for simple printing.
        if inferred_resets > 0:
            at = -1 if first_reset_at is None else first_reset_at
            violations.append(
                Violation(
                    at,
                    Op.TrackAdd,
                    -1,
                    f"INFO: inferred {inferred_resets} implicit reset(s) due to duplicate TrackAdd",
                )
            )

    return violations


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate sims-heuristics tracker u16 trace.")
    ap.add_argument("--instance", required=True, help="Path to .dzn file (for num_images)")
    ap.add_argument("--trace", required=True, help="Path to .u16 trace")
    ap.add_argument(
        "--no-strict-peek",
        action="store_true",
        help="Only validate TrackAdd/TrackRem consistency; ignore Peek* consistency.",
    )
    ap.add_argument(
        "--reset-on-duplicate-add",
        action="store_true",
        help=(
            "Treat TrackAdd on an already-selected image as an implicit reset (tracker/solution "
            "reinitialization not encoded in the trace). Still fails on invalid removals."
        ),
    )
    ap.add_argument(
        "--max-violations",
        type=int,
        default=20,
        help="Stop after this many violations (default: 20)",
    )
    ap.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Validate only first N records (0 = all).",
    )

    args = ap.parse_args()

    num_images = parse_num_images_from_dzn(args.instance)
    strict_peek = not args.no_strict_peek
    max_records = None if args.max_records == 0 else args.max_records

    violations = validate_trace(
        trace_path=args.trace,
        num_images=num_images,
        strict_peek=strict_peek,
        reset_on_duplicate_add=args.reset_on_duplicate_add,
        max_violations=args.max_violations,
        max_records=max_records,
    )

    if not violations:
        mode = "strict" if strict_peek else "track-only"
        limit = "all" if max_records is None else str(max_records)
        print(f"OK: trace is valid ({mode}), checked {limit} records; num_images={num_images}")
        return 0

    # Reset mode uses a synthetic INFO record when there were no true violations.
    if args.reset_on_duplicate_add and len(violations) == 1 and violations[0].reason.startswith("INFO:"):
        mode = "strict" if strict_peek else "track-only"
        limit = "all" if max_records is None else str(max_records)
        print(f"OK (with implicit resets): trace is valid ({mode}), checked {limit} records; num_images={num_images}")
        print(f"  {violations[0].reason}")
        if violations[0].pos >= 0:
            print(f"  first reset inferred at record #{violations[0].pos}")
        return 0

    print(f"FAIL: found {len(violations)} violation(s) (showing up to {args.max_violations})")
    for v in violations:
        print(f"  at record #{v.pos}: {v.op.name} idx={v.idx}: {v.reason}")

    if max_records is None:
        print("Tip: run with --max-records N to narrow down quickly")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
