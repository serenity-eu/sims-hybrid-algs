#!/usr/bin/env python3
"""Convert the Hybrid Benchmark Report Markdown to a self-contained HTML file.

Embeds all referenced SVG plots inline (as base64 data-URIs) so the output is
a single portable HTML file with no external dependencies.

Usage:
    uv run --with 'markdown>=3.6' --with 'pygments>=2.17' \
        python3 scripts/generate_hybrid_benchmark_html.py \
        --md docs/HYBRID_BENCHMARK_REPORT.md \
        --output docs/HYBRID_BENCHMARK_REPORT.html

    # Or if you already have the deps installed:
    python3 scripts/generate_hybrid_benchmark_html.py \
        --md docs/HYBRID_BENCHMARK_REPORT.md \
        --output docs/HYBRID_BENCHMARK_REPORT.html
"""

from __future__ import annotations

import argparse
import base64
import os
import re
import sys
from pathlib import Path

import markdown
from markdown.extensions.toc import TocExtension


# ── SVG inlining ─────────────────────────────────────────────────────────────


def inline_svgs(html: str, base_dir: Path) -> str:
    """Replace <img src="…/foo.svg"> tags with inline base64 data-URIs."""

    img_re = re.compile(
        r'<img\s+([^>]*?)src="([^"]+\.svg)"([^>]*?)/?>',
        re.IGNORECASE,
    )

    def _replace(m: re.Match) -> str:
        prefix_attrs = m.group(1)
        src = m.group(2)
        suffix_attrs = m.group(3)
        svg_path = base_dir / src
        if not svg_path.exists():
            print(f"  ⚠  SVG not found, leaving as-is: {svg_path}", file=sys.stderr)
            return m.group(0)
        svg_bytes = svg_path.read_bytes()
        b64 = base64.b64encode(svg_bytes).decode("ascii")
        data_uri = f"data:image/svg+xml;base64,{b64}"
        return f'<img {prefix_attrs}src="{data_uri}"{suffix_attrs} />'

    return img_re.sub(_replace, html)


# ── CSS theme ────────────────────────────────────────────────────────────────

CSS = r"""
:root {
    --bg: #ffffff;
    --fg: #1a1a2e;
    --accent: #2196F3;
    --accent2: #4CAF50;
    --border: #e0e0e0;
    --code-bg: #f5f5f5;
    --table-head: #f0f4f8;
    --table-stripe: #fafbfc;
    --blockquote-bg: #f8f9fb;
    --blockquote-border: #2196F3;
    --shadow: rgba(0,0,0,0.06);
    --max-width: 1100px;
}

*, *::before, *::after { box-sizing: border-box; }

html {
    font-size: 16px;
    scroll-behavior: smooth;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    line-height: 1.7;
    color: var(--fg);
    background: var(--bg);
    margin: 0;
    padding: 0;
}

.container {
    max-width: var(--max-width);
    margin: 0 auto;
    padding: 2rem 2.5rem 4rem;
}

/* ── Typography ──────────────────────────────────────────────────────── */

h1 {
    font-size: 2rem;
    margin-top: 0;
    padding-bottom: 0.6rem;
    border-bottom: 3px solid var(--accent);
    color: var(--fg);
}

h2 {
    font-size: 1.5rem;
    margin-top: 2.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid var(--border);
}

h3 {
    font-size: 1.2rem;
    margin-top: 2rem;
    color: #333;
}

h4 { font-size: 1.05rem; margin-top: 1.4rem; }

p { margin: 0.8rem 0; }

a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

/* ── Lead paragraph (subtitle) ───────────────────────────────────────── */

h1 + blockquote {
    font-size: 1.05rem;
    color: #555;
    border-left: 4px solid var(--blockquote-border);
    background: var(--blockquote-bg);
    margin: 1.2rem 0 2rem;
    padding: 1rem 1.4rem;
    border-radius: 0 6px 6px 0;
}

blockquote {
    border-left: 4px solid var(--blockquote-border);
    background: var(--blockquote-bg);
    margin: 1rem 0;
    padding: 0.8rem 1.2rem;
    border-radius: 0 6px 6px 0;
}

blockquote p { margin: 0.3rem 0; }

/* ── Code ────────────────────────────────────────────────────────────── */

code {
    background: var(--code-bg);
    padding: 0.15em 0.4em;
    border-radius: 4px;
    font-size: 0.9em;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
}

pre {
    background: #282c34;
    color: #abb2bf;
    padding: 1rem 1.2rem;
    border-radius: 8px;
    overflow-x: auto;
    font-size: 0.85rem;
    line-height: 1.5;
    box-shadow: 0 2px 8px var(--shadow);
}

pre code {
    background: transparent;
    padding: 0;
    color: inherit;
}

/* ── Tables ──────────────────────────────────────────────────────────── */

table {
    border-collapse: collapse;
    width: 100%;
    margin: 1rem 0 1.4rem;
    font-size: 0.9rem;
    box-shadow: 0 1px 4px var(--shadow);
    border-radius: 6px;
    overflow: hidden;
}

thead th {
    background: var(--table-head);
    font-weight: 600;
    text-align: left;
    padding: 0.6rem 0.8rem;
    border-bottom: 2px solid var(--border);
    white-space: nowrap;
}

tbody td {
    padding: 0.5rem 0.8rem;
    border-bottom: 1px solid var(--border);
}

tbody tr:nth-child(even) { background: var(--table-stripe); }
tbody tr:hover { background: #e8f0fe; }

/* Centre-aligned cells if specified */
td[align="center"], th[align="center"] { text-align: center; }

/* ── Images / SVG ────────────────────────────────────────────────────── */

img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 1.2rem auto;
    border-radius: 6px;
    box-shadow: 0 2px 12px var(--shadow);
}

/* ── Horizontal rules ────────────────────────────────────────────────── */

hr {
    border: none;
    border-top: 2px solid var(--border);
    margin: 2.5rem 0;
}

/* ── Lists ───────────────────────────────────────────────────────────── */

ul, ol { padding-left: 1.6rem; }
li { margin: 0.25rem 0; }
li > p { margin: 0.2rem 0; }

/* ── TOC (generated by toc extension) ────────────────────────────────── */

.toc {
    background: var(--blockquote-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.5rem;
    margin: 1.5rem 0 2rem;
    font-size: 0.92rem;
    line-height: 1.6;
}

.toc > ul { padding-left: 0; list-style: none; }
.toc ul ul { padding-left: 1.3rem; }
.toc li { margin: 0.15rem 0; }

/* ── Strong inside table cells (bold winners) ────────────────────────── */

td strong { color: var(--accent2); }

/* ── Print ───────────────────────────────────────────────────────────── */

@media print {
    body { font-size: 11pt; }
    .container { max-width: 100%; padding: 0; }
    img { box-shadow: none; page-break-inside: avoid; }
    h2, h3 { page-break-after: avoid; }
    table { page-break-inside: avoid; }
    pre { white-space: pre-wrap; word-wrap: break-word; }
}

/* ── Responsive tweaks ───────────────────────────────────────────────── */

@media (max-width: 800px) {
    .container { padding: 1rem; }
    h1 { font-size: 1.5rem; }
    h2 { font-size: 1.25rem; }
    table { font-size: 0.8rem; }
    th, td { padding: 0.35rem 0.5rem; }
}
"""


# ── HTML template ────────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title}</title>
<style>
{css}
</style>
</head>
<body>
<div class="container">
{body}
</div>
</body>
</html>
"""


# ── Markdown → HTML pipeline ────────────────────────────────────────────────


def convert(md_path: Path, output_path: Path) -> None:
    md_text = md_path.read_text(encoding="utf-8")

    # Extract title from the first H1
    title_match = re.search(r"^#\s+(.+)$", md_text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Benchmark Report"

    # Configure markdown extensions
    extensions = [
        "tables",
        "fenced_code",
        "codehilite",
        "attr_list",
        "md_in_html",
        TocExtension(
            permalink=False,
            toc_depth="2-3",
        ),
    ]
    extension_configs = {
        "codehilite": {
            "css_class": "highlight",
            "guess_lang": False,
            "noclasses": True,
            "pygments_style": "one-dark",
        },
    }

    md = markdown.Markdown(
        extensions=extensions,
        extension_configs=extension_configs,
    )

    body_html = md.convert(md_text)

    # The markdown TOC extension generates [TOC] markers or we can use the
    # toc attribute.  The report has its own "## Table of Contents" with
    # manual links, so we leave those as-is.

    # Inline SVGs relative to the markdown file's directory
    base_dir = md_path.parent
    body_html = inline_svgs(body_html, base_dir)

    # Assemble final HTML
    final_html = HTML_TEMPLATE.format(
        title=title,
        css=CSS,
        body=body_html,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(final_html, encoding="utf-8")

    size_kb = output_path.stat().st_size / 1024
    print(f"  ✓ {output_path}  ({size_kb:.0f} KB)")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Convert Hybrid Benchmark Report Markdown to self-contained HTML.",
    )
    parser.add_argument(
        "--md",
        type=Path,
        default=Path("docs/HYBRID_BENCHMARK_REPORT.md"),
        help="Input Markdown file (default: docs/HYBRID_BENCHMARK_REPORT.md)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML file (default: same name as input with .html extension)",
    )
    args = parser.parse_args()

    if not args.md.exists():
        print(f"Error: {args.md} not found.", file=sys.stderr)
        sys.exit(1)

    output = args.output or args.md.with_suffix(".html")

    print(f"Converting {args.md} → {output}")
    print(f"  Base dir for SVGs: {args.md.parent}")

    svg_count = len(
        re.findall(
            r"!\[.*?\]\(.*?\.svg\)",
            args.md.read_text(encoding="utf-8"),
        )
    )
    print(f"  SVG references found: {svg_count}")

    convert(args.md, output)

    # Count inlined images in output
    out_text = output.read_text(encoding="utf-8")
    inlined = out_text.count("data:image/svg+xml;base64,")
    print(f"  Inlined SVGs: {inlined}/{svg_count}")
    if inlined < svg_count:
        print(
            f"  ⚠  {svg_count - inlined} SVG(s) could not be inlined (file not found?)",
            file=sys.stderr,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
