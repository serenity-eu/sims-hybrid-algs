#!/usr/bin/env python3
"""Postprocess an inferno/flamegraph-rs SVG to shorten frame labels.

Goal: rewrite each frame label to only the function name with parentheses, e.g.
- "...::NdTreeSolutionSet<...>::run" -> "run()"
- "<...>::from_selected_images_condensed::<...>" -> "from_selected_images_condensed()"

This script edits only text content (<title>, and <text> if present) and does not
change geometry attributes, so the SVG structure/rendering stays valid.

Usage:
  python shorten_flamegraph_svg.py input.svg output.svg

Tip (in this repo):
  uv run --no-project -s shorten_flamegraph_svg.py flamegraph.svg flamegraph.short.svg
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET


SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
INFERNO_NS = "http://github.com/jonhoo/inferno"


_TITLE_SUFFIX_RE = re.compile(r"\s*\((?:\d+(?:,\d+)*)(?:\s+samples)?\s*,\s*[\d.]+%\)\s*$")
_TITLE_SUFFIX_CAPTURE_RE = re.compile(
    r"\s*(\((?:\d+(?:,\d+)*)(?:\s+samples)?\s*,\s*[\d.]+%\))\s*$"
)


def _strip_title_suffix(title: str) -> str:
    # Handles common formats:
    #  - "foo (123 samples, 4.56%)"
    #  - "foo (123, 4.56%)"
    return _TITLE_SUFFIX_RE.sub("", title).strip()


def _extract_title_suffix(title: str) -> str | None:
    """Extract the trailing "(N samples, P%)" suffix, if present."""
    m = _TITLE_SUFFIX_CAPTURE_RE.search(title)
    return m.group(1) if m else None


def _strip_trailing_generic_args(name: str) -> str:
    """Strip trailing Rust generic instantiation like "::<...>" or "<...>" at end.

    Repeats until no trailing generic block remains.
    """

    s = name.strip()

    def find_matching_lt(text: str, gt_index: int) -> int | None:
        depth = 0
        for i in range(gt_index, -1, -1):
            ch = text[i]
            if ch == ">":
                depth += 1
            elif ch == "<":
                depth -= 1
                if depth == 0:
                    return i
        return None

    while s.endswith(">"):
        lt = find_matching_lt(s, len(s) - 1)
        if lt is None:
            break
        prefix = s[:lt].rstrip()
        # Allow stripping either "::<...>" or plain "<...>" at end.
        if prefix.endswith("::"):
            prefix = prefix[:-2].rstrip()
        s = prefix

    return s


def _last_path_segment_outside_generics(name: str) -> str:
    """Return last ::segment not inside <...> generic nesting."""

    depth = 0
    last_sep = None
    i = 0
    while i < len(name) - 1:
        ch = name[i]
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif depth == 0 and name[i : i + 2] == "::":
            last_sep = i
            i += 1
        i += 1

    if last_sep is None:
        return name.strip()
    return name[last_sep + 2 :].strip()


def _split_path_segments_outside_generics(name: str) -> list[str]:
    """Split a Rust path by ::, ignoring separators inside <...> generic nesting."""
    parts: list[str] = []
    depth = 0
    start = 0
    i = 0
    while i < len(name) - 1:
        ch = name[i]
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif depth == 0 and name[i : i + 2] == "::":
            parts.append(name[start:i].strip())
            start = i + 2
            i += 1
        i += 1

    parts.append(name[start:].strip())
    return [p for p in parts if p]


def _looks_like_type_name(seg: str) -> bool:
    # Heuristic: types/traits usually contain uppercase ASCII.
    return any("A" <= c <= "Z" for c in seg)


def _looks_like_module_name(seg: str) -> bool:
    """Heuristic for Rust module/function path segments.

    We only want a short, stable prefix for free/module-level functions.
    """

    s = seg.strip()
    if not s:
        return False
    if " " in s:
        return False
    if any(ch in s for ch in "<>[]{}()"):
        return False
    # Modules are typically snake_case-ish.
    return all(
        ("a" <= c <= "z") or ("0" <= c <= "9") or c in "_-$" for c in s
    )


def _clean_owner_segment(seg: str) -> str:
    """Extract a concise owner name from segments like "<T as Trait>" or "ResidualSolution<4>"."""

    s = seg.strip()
    if not s:
        return s

    # Strip surrounding <...> if present.
    if s.startswith("<") and s.endswith(">"):
        s = s[1:-1].strip()

    # For "T as Trait" prefer the trait side.
    if " as " in s:
        s = s.split(" as ", 1)[1].strip()

    # Strip trailing generics.
    s = _strip_trailing_generic_args(s)

    # Keep only the last path segment.
    s = _last_path_segment_outside_generics(s)
    s = _strip_trailing_generic_args(s)

    return s.strip()


def shorten_symbol(raw: str) -> str:
    s = raw.strip()

    if not s:
        return s

    # Preserve special placeholder frames.
    if s == "[unknown]":
        return s

    # Drop sample/% suffixes if present.
    s = _strip_title_suffix(s)

    # Some frames include extra whitespace/newlines.
    s = " ".join(s.split())

    # Remove trailing generic instantiations.
    s = _strip_trailing_generic_args(s)

    # Keep last segment (method/function name). Optionally prefix with owner type.
    parts = _split_path_segments_outside_generics(s)
    last = _strip_trailing_generic_args(parts[-1]) if parts else s

    owner = ""
    if len(parts) >= 2:
        candidate_owner = _clean_owner_segment(parts[-2])
        # Prefer a type/trait-like owner.
        if candidate_owner and _looks_like_type_name(candidate_owner):
            owner = candidate_owner
        # If there's no owning type (free function), keep the immediate module name.
        elif candidate_owner and _looks_like_module_name(candidate_owner):
            owner = candidate_owner

    seg = last.strip()
    if owner:
        seg = f"{owner}.{seg}"

    # Normalize common Rust closure marker.
    if "{{closure" in seg:
        seg = "closure"

    # If we somehow ended up with an impl/trait "<T as Trait>" segment, fall back.
    seg = seg.strip()
    if not seg:
        seg = s.strip() or raw.strip()

    # Ensure () suffix.
    if not seg.endswith("()"):  # don't double-append
        seg = f"{seg}()"

    return seg


def shorten_title_preserve_suffix(raw_title: str) -> str:
    """Shorten the symbol portion but preserve the original suffix with samples/%.

    Example:
        "foo::bar (123 samples, 4.56%)" -> "bar() (123 samples, 4.56%)"
    """

    suffix = _extract_title_suffix(raw_title)
    short = shorten_symbol(raw_title)
    if suffix is None:
        return short
    return f"{short} {suffix}"


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("Usage: python shorten_flamegraph_svg.py <input.svg> <output.svg>")
        return 2

    in_path, out_path = argv[1], argv[2]

    # Preserve namespaces so the embedded FlameGraph JS keeps working.
    # If ElementTree invents prefixes like `ns0:svg`, `getElementsByTagName("svg")`
    # may not match the root element, and fluid resizing breaks.
    ET.register_namespace("", SVG_NS)
    ET.register_namespace("xlink", XLINK_NS)
    ET.register_namespace("fg", INFERNO_NS)

    try:
        tree = ET.parse(in_path)
    except Exception as e:
        print(f"Error parsing SVG: {e}")
        return 1

    root = tree.getroot()

    # Inferno/flamegraph-rs SVGs typically use these namespaces, but we avoid
    # hard-dependence by matching tags by suffix (works for default ns).

    titles_changed = 0
    texts_changed = 0

    for g in root.iter():
        if not str(g.tag).endswith("g"):
            continue

        title_el = None
        text_el = None
        for child in list(g):
            if str(child.tag).endswith("title"):
                title_el = child
            elif str(child.tag).endswith("text"):
                text_el = child

        if title_el is None or title_el.text is None:
            continue

        original_title = title_el.text
        short_title = shorten_title_preserve_suffix(original_title)
        short_text = shorten_symbol(original_title)

        if short_title != original_title:
            title_el.text = short_title
            titles_changed += 1

        # Update visible label too (if present). This keeps geometry intact.
        if text_el is not None and (text_el.text is not None):
            original_text = text_el.text
            # Avoid rewriting empty/placeholder labels.
            if original_text.strip():
                if short_text != original_text:
                    text_el.text = short_text
                    texts_changed += 1

    # Serialize. ElementTree tends to invent prefixes like `ns0:` for the SVG namespace.
    # That can break FlameGraph's embedded JS (it does `getElementsByTagName("svg")`),
    # so after writing we normalize namespaces back to a default SVG namespace.
    tree.write(out_path, encoding="utf-8", xml_declaration=True)

    try:
        with open(out_path, "r", encoding="utf-8") as f:
            svg_text = f.read()

        # 1) Restore default SVG namespace (no prefix) and remove ns0: element prefixes.
        #    This is intentionally a simple text transformation over ElementTree output.
        svg_text = svg_text.replace(
            'xmlns:ns0="http://www.w3.org/2000/svg"',
            'xmlns="http://www.w3.org/2000/svg"',
        )
        svg_text = svg_text.replace("<ns0:", "<").replace("</ns0:", "</")

        # 2) Prefer the traditional inferno prefix used by flamegraph-rs.
        svg_text = svg_text.replace(
            'xmlns:ns1="http://github.com/jonhoo/inferno"',
            'xmlns:fg="http://github.com/jonhoo/inferno"',
        )
        svg_text = svg_text.replace("ns1:", "fg:")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(svg_text)
    except OSError:
        # If we can't re-open the file, keep the serialized SVG as-is.
        pass

    print(
        f"Wrote {out_path}. Updated titles: {titles_changed}, updated text labels: {texts_changed}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
