#!/usr/bin/env python3
"""
perf_tree.py – Parse an inferno-folded stacks file and print a levelled call tree.

The folded format is the same one produced by:
    cargo flamegraph --post-process 'tee file.folded' ...

Usage:
    python3 perf_tree.py <folded_file> [options]

Examples:
    python3 perf_tree.py flamegraph_opt3.folded --max-depth 1
    python3 perf_tree.py flamegraph_opt3.folded --max-depth 4 --min-percent 1.0
    python3 perf_tree.py flamegraph_opt3.folded --max-depth 6 --filter solve
    python3 perf_tree.py flamegraph_opt3.folded --max-depth 5 --focus bitset_encoded_solution

Options:
    --max-depth N       Maximum tree depth to print (default: 4).
                        depth=1 prints only the synthetic "all" root.
                        depth=2 adds the first real frame level, etc.
    --min-percent P     Hide nodes whose inclusive % is below P (default: 0.5).
    --width W           Max function name display width in chars (default: 70).
    --filter SUBSTR     Only include stacks that contain SUBSTR anywhere.
    --focus SUBSTR      Re-root the tree at the first node whose name contains
                        SUBSTR, then print from there.
    --no-dedup          By default, Rust symbols often appear as duplicates
                        (inlined copies).  This flag disables the merge.
    --sort-self         Sort children by self% instead of total% (default: total%).
"""

import argparse
import sys
from collections import defaultdict
from typing import Optional


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

class Node:
    __slots__ = ('name', 'self_count', 'total_count', 'children')

    def __init__(self, name: str):
        self.name        = name
        self.self_count  = 0   # stacks that END at this exact frame
        self.total_count = 0   # all stacks passing THROUGH this frame
        self.children: dict[str, 'Node'] = {}

    def child(self, name: str) -> 'Node':
        if name not in self.children:
            self.children[name] = Node(name)
        return self.children[name]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_folded(filepath: str, filter_substr: Optional[str]) -> list[tuple[list[str], int]]:
    stacks: list[tuple[list[str], int]] = []
    with open(filepath, encoding='utf-8', errors='replace') as fh:
        for raw in fh:
            line = raw.rstrip('\n')
            if not line:
                continue
            sep = line.rfind(' ')
            if sep == -1:
                continue
            frames_str = line[:sep]
            try:
                count = int(line[sep + 1:])
            except ValueError:
                continue
            if filter_substr and filter_substr.lower() not in frames_str.lower():
                continue
            stacks.append((frames_str.split(';'), count))
    return stacks


# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------

def build_tree(stacks: list[tuple[list[str], int]]) -> Node:
    root = Node('all')
    for frames, count in stacks:
        root.total_count += count
        node = root
        for frame in frames:
            child = node.child(frame)
            child.total_count += count
            node = child
        node.self_count += count
    return root


# ---------------------------------------------------------------------------
# Focus: re-root at a subtree containing a substring
# ---------------------------------------------------------------------------

def find_focus(node: Node, substr: str) -> Optional[Node]:
    """BFS for the first node whose name contains substr (case-insensitive)."""
    lower = substr.lower()
    queue = [node]
    while queue:
        current = queue.pop(0)
        if lower in current.name.lower() and current is not node:
            return current
        queue.extend(current.children.values())
    return None


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

_BOX_LAST  = '└── '
_BOX_MID   = '├── '
_BOX_VT    = '│   '
_BOX_SPACE = '    '


def _trim(name: str, width: int) -> str:
    if len(name) <= width:
        return name
    # Keep the tail (more informative for long Rust symbols)
    return '…' + name[-(width - 1):]


def _print_node(
    node: Node,
    grand_total: int,
    max_depth: int,
    min_pct: float,
    width: int,
    sort_self: bool,
    depth: int,
    prefix: str,
    is_last: bool,
    is_root: bool,
):
    if depth > max_depth:
        return

    total_pct = 100.0 * node.total_count / grand_total if grand_total else 0.0
    self_pct  = 100.0 * node.self_count  / grand_total if grand_total else 0.0

    if not is_root and total_pct < min_pct:
        return

    connector   = '' if is_root else (_BOX_LAST if is_last else _BOX_MID)
    child_pfx   = '' if is_root else (prefix + (_BOX_SPACE if is_last else _BOX_VT))

    name_col = _trim(node.name, width)

    if node.self_count > 0:
        row = (f"{prefix}{connector}"
               f"{name_col:<{width}}  "
               f"{total_pct:6.2f}%  "
               f"(self {self_pct:.2f}%)")
    else:
        row = (f"{prefix}{connector}"
               f"{name_col:<{width}}  "
               f"{total_pct:6.2f}%")

    print(row)

    if depth == max_depth:
        return

    # Sort children
    key_fn = (lambda n: n.self_count) if sort_self else (lambda n: n.total_count)
    visible   = [c for c in node.children.values()
                 if 100.0 * c.total_count / grand_total >= min_pct]
    invisible = [c for c in node.children.values()
                 if 100.0 * c.total_count / grand_total < min_pct]
    visible.sort(key=key_fn, reverse=True)

    hidden_count = sum(c.total_count for c in invisible)

    for i, child in enumerate(visible):
        last_child = (i == len(visible) - 1) and hidden_count == 0
        _print_node(
            child, grand_total, max_depth, min_pct, width, sort_self,
            depth + 1, child_pfx, last_child, is_root=False,
        )

    if hidden_count > 0:
        hidden_pct = 100.0 * hidden_count / grand_total
        n_hidden   = len(invisible)
        print(f"{child_pfx}{_BOX_LAST}"
              f"[{n_hidden} nodes below {min_pct:.1f}% threshold, "
              f"total {hidden_pct:.2f}%]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Print a levelled call tree from an inferno-folded stacks file.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('Options:')[1] if 'Options:' in __doc__ else '',
    )
    parser.add_argument('folded_file')
    parser.add_argument('--max-depth',   type=int,   default=4,   metavar='N')
    parser.add_argument('--min-percent', type=float, default=0.5, metavar='P')
    parser.add_argument('--width',       type=int,   default=55,  metavar='W')
    parser.add_argument('--filter',      metavar='SUBSTR',
                        help='Only stacks containing this substring')
    parser.add_argument('--focus',       metavar='SUBSTR',
                        help='Re-root tree at first node whose name contains this')
    parser.add_argument('--no-dedup',    action='store_true',
                        help='Disable inlined-duplicate merging (not yet implemented)')
    parser.add_argument('--sort-self',   action='store_true',
                        help='Sort children by self%% instead of total%%')
    args = parser.parse_args()

    print(f"Loading {args.folded_file} …", file=sys.stderr)
    stacks = parse_folded(args.folded_file, args.filter)
    print(f"  {len(stacks):,} stack entries", file=sys.stderr)

    root = build_tree(stacks)
    grand_total = root.total_count
    print(f"  {grand_total:,} total samples", file=sys.stderr)

    display_root = root
    if args.focus:
        found = find_focus(root, args.focus)
        if found is None:
            print(f"WARNING: --focus '{args.focus}' not found; showing full tree.",
                  file=sys.stderr)
        else:
            display_root = found
            print(f"  Focused on: {found.name}", file=sys.stderr)

    print(file=sys.stderr)
    _print_node(
        display_root, grand_total,
        max_depth=args.max_depth,
        min_pct=args.min_percent,
        width=args.width,
        sort_self=args.sort_self,
        depth=1,
        prefix='',
        is_last=True,
        is_root=True,
    )


if __name__ == '__main__':
    main()
