"""Turn benchmark JSON into a markdown table, SVG chart, and README patch."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

SCENARIO_LABELS = {
    "torch_cuda": "PyTorch CUDA",
    "yolo": "YOLOv8n",
    "qwen": "Qwen 2.5-0.5B",
    "qwen7": "Qwen 2.5-7B",
    "qwen32": "Qwen 2.5-32B",
    "whisper": "Whisper-tiny",
}

README_START = "<!-- BENCHMARK_RESULTS:START -->"
README_END = "<!-- BENCHMARK_RESULTS:END -->"
NOTES_START = "<!-- kryo-gpu-benchmarks -->"
NOTES_END = "<!-- /kryo-gpu-benchmarks -->"
CHART_REL = "benchmarks/results/charts/cold-vs-kryo.svg"

COLD_COLOR = "#8b95a8"
KRYO_COLOR = "#5ec8e6"
SPEEDUP_COLOR = "#7ee0a3"
BG = "#16181d"
GRID = "#2a2e36"
TEXT = "#e8eaed"
MUTED = "#9aa3b2"


def load_results(path: Path) -> dict[str, Any]:
    """Load a runner.py JSON document."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected object in {path}")
    return data


def mean_seconds(mode: object) -> float | None:
    """Read mean seconds from a cold/kryo block."""
    if not isinstance(mode, dict):
        return None
    total = mode.get("total")
    if not isinstance(total, dict):
        return None
    value = total.get("mean")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def scenario_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten scenarios into display rows."""
    scenarios = results.get("scenarios")
    if not isinstance(scenarios, dict):
        return []
    rows: list[dict[str, Any]] = []
    for name, data in scenarios.items():
        if not isinstance(name, str) or not isinstance(data, dict):
            continue
        if "error" in data and mean_seconds(data.get("cold")) is None:
            rows.append({"name": name, "error": str(data["error"])})
            continue
        cold = mean_seconds(data.get("cold"))
        kryo = mean_seconds(data.get("kryo"))
        speedup = data.get("speedup")
        if not isinstance(speedup, (int, float)) and cold and kryo and kryo > 0:
            speedup = cold / kryo
        rows.append(
            {
                "name": name,
                "label": SCENARIO_LABELS.get(name, name),
                "cold": cold,
                "kryo": kryo,
                "speedup": float(speedup) if isinstance(speedup, (int, float)) else None,
            }
        )
    return rows


def caption(metadata: object) -> str:
    """One-line hardware / sample-size summary."""
    if not isinstance(metadata, dict):
        return "Cold start vs Kryo restore"
    parts: list[str] = []
    gpu = metadata.get("gpu")
    if isinstance(gpu, str) and gpu:
        parts.append(f"**{gpu}**")
    instance = metadata.get("instance_type")
    if isinstance(instance, str) and instance:
        parts.append(f"Lambda `{instance}`")
    driver = metadata.get("driver")
    if isinstance(driver, str) and driver:
        parts.append(f"driver {driver}")
    runs = metadata.get("runs_per_mode")
    if isinstance(runs, int):
        sample = "run" if runs == 1 else "runs"
        parts.append(f"{runs} timed {sample} + warmup")
    tag = metadata.get("release_tag")
    if isinstance(tag, str) and tag:
        parts.append(f"release `{tag}`")
    note = metadata.get("note")
    line = " · ".join(parts) if parts else "Cold start vs Kryo restore"
    if isinstance(note, str) and note:
        return f"{line}. {note}"
    return line


def markdown_table(results: dict[str, Any]) -> str:
    """GitHub-flavored table of cold vs restore means."""
    lines = [
        "| Scenario | Cold start | Kryo restore | Speedup |",
        "|----------|------------|--------------|---------|",
    ]
    for row in scenario_rows(results):
        label = SCENARIO_LABELS.get(str(row["name"]), str(row["name"]))
        if "error" in row:
            lines.append(f"| {label} | — | — | error |")
            continue
        cold = row.get("cold")
        kryo = row.get("kryo")
        speedup = row.get("speedup")
        cold_s = f"{cold:.2f}s" if isinstance(cold, float) else "—"
        kryo_s = f"{kryo:.2f}s" if isinstance(kryo, float) else "—"
        speed_s = f"{speedup:.1f}x" if isinstance(speedup, float) else "—"
        lines.append(f"| {label} | {cold_s} | {kryo_s} | {speed_s} |")
    return "\n".join(lines)


def readme_block(results: dict[str, Any], image_rel: str) -> str:
    """README section between the result markers."""
    cap = caption(results.get("metadata"))
    return (
        f"{README_START}\n"
        f"![Cold start vs Kryo restore]({image_rel})\n\n"
        f"{cap}\n\n"
        f"{markdown_table(results)}\n"
        f"{README_END}"
    )


def patch_readme(readme: Path, results: dict[str, Any], image_rel: str) -> None:
    """Replace or insert the marked benchmark results block."""
    text = readme.read_text(encoding="utf-8")
    block = readme_block(results, image_rel)
    pattern = re.compile(
        re.escape(README_START) + r".*?" + re.escape(README_END),
        re.DOTALL,
    )
    if pattern.search(text):
        text = pattern.sub(block, text, count=1)
    else:
        heading = "## Contributing"
        if heading in text:
            text = text.replace(heading, f"{block}\n\n{heading}", 1)
        else:
            text = f"{text.rstrip()}\n\n{block}\n"
    readme.write_text(text, encoding="utf-8")


def xml_escape(text: str) -> str:
    """Escape text for SVG."""
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def nice_ceiling(value: float) -> float:
    """Round up to a chart-friendly y-axis max."""
    if value <= 0:
        return 1.0
    magnitude = float(10 ** math.floor(math.log10(value)))
    for step in (1.0, 2.0, 2.5, 5.0, 10.0):
        candidate = step * magnitude
        if candidate >= value:
            return float(candidate)
    return float(10.0 * magnitude)


def svg_chart(results: dict[str, Any]) -> str:
    """Grouped bar chart of cold vs Kryo restore times."""
    rows = [row for row in scenario_rows(results) if isinstance(row.get("cold"), float)]
    width, height = 880, 420
    left, right, top, bottom = 64, 28, 78, 62
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [float(row["cold"]) for row in rows]
    values.extend(float(row["kryo"]) for row in rows if isinstance(row.get("kryo"), float))
    y_max = nice_ceiling(max(values, default=1.0) * 1.12)
    n = max(len(rows), 1)
    group_w = plot_w / n
    bar_w = min(36.0, group_w * 0.28)
    gap = 8.0

    def y_pos(seconds: float) -> float:
        return top + plot_h * (1.0 - seconds / y_max)

    ticks = int(y_max) if y_max in {1.0, 2.0, 5.0, 10.0} else 4
    grid_parts: list[str] = []
    for i in range(ticks + 1):
        value = y_max * i / ticks
        y = y_pos(value)
        if abs(value - round(value)) < 1e-6:
            tick = f"{round(value)}s"
        else:
            tick = f"{value:.1f}s"
        grid_parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        grid_parts.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" fill="{MUTED}" font-size="12" '
            f'text-anchor="end" font-family="ui-sans-serif, system-ui, sans-serif">'
            f"{tick}</text>"
        )

    bars: list[str] = []
    for index, row in enumerate(rows):
        center = left + group_w * (index + 0.5)
        cold = float(row["cold"])
        kryo = row.get("kryo")
        cold_x = center - gap / 2 - bar_w
        cold_y = y_pos(cold)
        bars.append(
            f'<rect x="{cold_x:.1f}" y="{cold_y:.1f}" width="{bar_w:.1f}" '
            f'height="{top + plot_h - cold_y:.1f}" rx="4" fill="{COLD_COLOR}"/>'
        )
        if isinstance(kryo, float):
            kryo_x = center + gap / 2
            kryo_y = y_pos(kryo)
            bars.append(
                f'<rect x="{kryo_x:.1f}" y="{kryo_y:.1f}" width="{bar_w:.1f}" '
                f'height="{top + plot_h - kryo_y:.1f}" rx="4" fill="{KRYO_COLOR}"/>'
            )
            speedup = row.get("speedup")
            if isinstance(speedup, float):
                bars.append(
                    f'<text x="{kryo_x + bar_w / 2:.1f}" y="{kryo_y - 8:.1f}" '
                    f'fill="{SPEEDUP_COLOR}" font-size="12" font-weight="600" '
                    f'text-anchor="middle" '
                    f'font-family="ui-sans-serif, system-ui, sans-serif">'
                    f"{speedup:.1f}x</text>"
                )
        label = xml_escape(str(row["label"]))
        bars.append(
            f'<text x="{center:.1f}" y="{height - 28}" fill="{TEXT}" font-size="13" '
            f'text-anchor="middle" font-family="ui-sans-serif, system-ui, sans-serif">'
            f"{label}</text>"
        )

    metadata = results.get("metadata")
    subtitle = caption(metadata) if isinstance(metadata, dict) else ""
    subtitle = re.sub(r"[*`]", "", subtitle)

    font = "ui-sans-serif, system-ui, sans-serif"
    return "\n".join(
        [
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
                f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
                'aria-label="Cold start vs Kryo restore">'
            ),
            f'<rect width="{width}" height="{height}" rx="16" fill="{BG}"/>',
            (
                f'<text x="{left}" y="32" fill="{TEXT}" font-size="20" '
                f'font-weight="700" font-family="{font}">Cold start vs Kryo restore</text>'
            ),
            (
                f'<text x="{left}" y="54" fill="{MUTED}" font-size="12" '
                f'font-family="{font}">{xml_escape(subtitle)}</text>'
            ),
            "<g>",
            (f'<rect x="{width - 250}" y="18" width="12" height="12" rx="2" fill="{COLD_COLOR}"/>'),
            (
                f'<text x="{width - 234}" y="28" fill="{MUTED}" font-size="12" '
                f'font-family="{font}">Cold start</text>'
            ),
            (f'<rect x="{width - 140}" y="18" width="12" height="12" rx="2" fill="{KRYO_COLOR}"/>'),
            (
                f'<text x="{width - 124}" y="28" fill="{MUTED}" font-size="12" '
                f'font-family="{font}">Kryo restore</text>'
            ),
            "</g>",
            "".join(grid_parts),
            "".join(bars),
            "</svg>",
            "",
        ]
    )


def merge_release_notes(existing: str, section: str) -> str:
    """Insert or replace the Kryo GPU benchmark section in release notes."""
    pattern = re.compile(
        re.escape(NOTES_START) + r".*?" + re.escape(NOTES_END),
        re.DOTALL,
    )
    block = f"{NOTES_START}\n{section.rstrip()}\n{NOTES_END}"
    if pattern.search(existing):
        return pattern.sub(block, existing, count=1)
    return f"{existing.rstrip()}\n\n{block}\n"


def release_section(results: dict[str, Any], tag: str, repo: str) -> str:
    """Markdown for GitHub release notes, with chart from the release asset."""
    image = f"https://github.com/{repo}/releases/download/{tag}/kryo-benchmarks.svg"
    return (
        "## GPU benchmarks\n\n"
        f"![Cold start vs Kryo restore]({image})\n\n"
        f"{caption(results.get('metadata'))}\n\n"
        f"{markdown_table(results)}\n"
    )


def main() -> None:
    """CLI for CI publishing and local README/chart regeneration."""
    parser = argparse.ArgumentParser(description="Format Kryo benchmark results")
    parser.add_argument("--json", required=True, help="Path to runner JSON")
    parser.add_argument("--svg", help="Write SVG chart here")
    parser.add_argument("--markdown", help="Write markdown table here")
    parser.add_argument("--patch-readme", help="Replace README result markers")
    parser.add_argument("--image-rel", default=CHART_REL, help="README image path")
    parser.add_argument("--notes-in", help="Existing release notes file")
    parser.add_argument("--notes-out", help="Write merged release notes here")
    parser.add_argument("--tag", default="", help="Release tag for notes image URLs")
    parser.add_argument("--repo", default="", help="owner/repo for notes image URLs")
    args = parser.parse_args()

    results = load_results(Path(args.json))
    table = markdown_table(results)

    if args.svg:
        svg_path = Path(args.svg)
        svg_path.parent.mkdir(parents=True, exist_ok=True)
        svg_path.write_text(svg_chart(results), encoding="utf-8")
    if args.markdown:
        md_path = Path(args.markdown)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(f"{caption(results.get('metadata'))}\n\n{table}\n", encoding="utf-8")
    if args.patch_readme:
        patch_readme(Path(args.patch_readme), results, args.image_rel)
    if args.notes_out:
        existing = Path(args.notes_in).read_text(encoding="utf-8") if args.notes_in else ""
        if not args.tag or not args.repo:
            parser.error("--notes-out requires --tag and --repo")
        section = release_section(results, args.tag, args.repo)
        Path(args.notes_out).write_text(merge_release_notes(existing, section), encoding="utf-8")


if __name__ == "__main__":
    main()
