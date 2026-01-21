"""Analysis and graph generation for cold start benchmarks."""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
GRAPHS_DIR = Path(__file__).parent / "graphs"

# Standardized phase order for consistent display
PHASE_ORDER = ["import", "cuda_init", "model_load", "first_inference"]
PHASE_COLORS = {
    "import": "#4C72B0",
    "cuda_init": "#55A868",
    "model_load": "#C44E52",
    "first_inference": "#8172B3",
}


def load_results(results_path: Path) -> dict[str, Any]:
    """Load results from JSON file."""
    with results_path.open(encoding="utf-8") as f:
        result: dict[str, Any] = json.load(f)
        return result


def print_table(results: dict[str, Any]) -> None:
    """Print a formatted table of results."""
    scenarios = results.get("scenarios", {})
    if not scenarios:
        print("No scenarios found in results")
        return

    # Header
    header = (
        f"{'Scenario':<18} | {'Total (s)':<10} | {'import':<10} | "
        f"{'cuda_init':<10} | {'model_load':<10} | {'first_inference':<15}"
    )
    print(header)
    print("-" * len(header))

    # Rows
    for scenario_name, scenario_data in scenarios.items():
        if "error" in scenario_data:
            print(f"{scenario_name:<18} | ERROR: {scenario_data['error']}")
            continue

        total = scenario_data.get("total", {}).get("mean", 0)
        phases = scenario_data.get("phases", {})

        row = f"{scenario_name:<18} | {total:<10.2f} |"
        for phase in PHASE_ORDER:
            if phase in phases:
                value = phases[phase].get("mean", 0)
                row += f" {value:<10.2f}|"
            else:
                row += f" {'-':<10}|"

        print(row)


def generate_phase_breakdown(results: dict[str, Any], output_path: Path) -> None:
    """Generate stacked bar chart showing phase breakdown per scenario."""
    scenarios = results.get("scenarios", {})
    valid_scenarios = {k: v for k, v in scenarios.items() if "error" not in v}

    if not valid_scenarios:
        print("No valid scenarios to plot")
        return

    _, ax = plt.subplots(figsize=(12, 6))

    scenario_names = list(valid_scenarios.keys())
    x = np.arange(len(scenario_names))
    width = 0.6

    bottoms = np.zeros(len(scenario_names))

    for phase in PHASE_ORDER:
        values = []
        for scenario_name in scenario_names:
            phases = valid_scenarios[scenario_name].get("phases", {})
            if phase in phases:
                values.append(phases[phase].get("mean", 0))
            else:
                values.append(0)

        values_arr = np.array(values)
        ax.bar(
            x,
            values_arr,
            width,
            bottom=bottoms,
            label=phase,
            color=PHASE_COLORS.get(phase, "#999999"),
        )
        bottoms += values_arr

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Cold Start Phase Breakdown by Scenario")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, rotation=45, ha="right")
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Generated: {output_path}")


def generate_total_comparison(results: dict[str, Any], output_path: Path) -> None:
    """Generate bar chart comparing total cold start times."""
    scenarios = results.get("scenarios", {})
    valid_scenarios = {k: v for k, v in scenarios.items() if "error" not in v}

    if not valid_scenarios:
        print("No valid scenarios to plot")
        return

    _, ax = plt.subplots(figsize=(10, 6))

    scenario_names = list(valid_scenarios.keys())
    totals = [
        valid_scenarios[s].get("total", {}).get("mean", 0) for s in scenario_names
    ]
    errors = [valid_scenarios[s].get("total", {}).get("std", 0) for s in scenario_names]

    x = np.arange(len(scenario_names))
    bars = ax.bar(x, totals, yerr=errors, capsize=5, color="#4C72B0", alpha=0.8)

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Total Cold Start Time (seconds)")
    ax.set_title("Total Cold Start Time Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, rotation=45, ha="right")

    # Add value labels on bars
    for bar, total in zip(bars, totals):  # noqa: B905
        height = bar.get_height()
        ax.annotate(
            f"{total:.2f}s",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Generated: {output_path}")


def generate_variance_plot(results: dict[str, Any], output_path: Path) -> None:
    """Generate box plots showing distribution across runs."""
    scenarios = results.get("scenarios", {})
    valid_scenarios = {k: v for k, v in scenarios.items() if "error" not in v}

    if not valid_scenarios:
        print("No valid scenarios to plot")
        return

    _, ax = plt.subplots(figsize=(12, 6))

    scenario_names = list(valid_scenarios.keys())

    # Create box plot data from percentiles
    # We'll approximate the distribution using p50, p95, p99
    box_data = []
    for scenario_name in scenario_names:
        total = valid_scenarios[scenario_name].get("total", {})
        if total:
            # Create synthetic data points based on statistics
            mean = total.get("mean", 0)
            std = total.get("std", 0)
            p50 = total.get("p50", mean)
            p95 = total.get("p95", mean + std)
            p99 = total.get("p99", mean + 2 * std)

            # Create a simple representation
            box_data.append([mean - std, p50, mean, p95, p99])
        else:
            box_data.append([0, 0, 0, 0, 0])

    # Use bxp for custom box plots
    bxp_stats = []
    for name in scenario_names:
        total = valid_scenarios[name].get("total", {})
        bxp_stats.append(
            {
                "med": total.get("p50", 0),
                "q1": total.get("mean", 0) - total.get("std", 0) * 0.675,
                "q3": total.get("mean", 0) + total.get("std", 0) * 0.675,
                "whislo": total.get("mean", 0) - total.get("std", 0) * 2,
                "whishi": total.get("p99", 0),
                "fliers": [],
            }
        )

    bp = ax.bxp(bxp_stats, showfliers=False, patch_artist=True)

    # Color the boxes
    for patch in bp["boxes"]:
        patch.set_facecolor("#4C72B0")
        patch.set_alpha(0.6)

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Total Cold Start Time (seconds)")
    ax.set_title("Cold Start Time Distribution by Scenario")
    ax.set_xticklabels(scenario_names, rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Generated: {output_path}")


def main() -> None:
    """CLI entrypoint for analyzing benchmark results and generating graphs."""
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input results file (default: results/latest.json)",
    )
    parser.add_argument(
        "--no-graphs",
        action="store_true",
        help="Skip graph generation",
    )

    args = parser.parse_args()

    # Load results
    results_path = Path(args.input) if args.input else RESULTS_DIR / "latest.json"
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        print("Run benchmarks first: python runner.py --all --runs 10")
        return

    results = load_results(results_path)

    # Print metadata
    metadata = results.get("metadata", {})
    if metadata:
        print("\nBenchmark Metadata:")
        print(f"  Timestamp: {metadata.get('timestamp', 'N/A')}")
        print(f"  Python: {metadata.get('python_version', 'N/A')}")
        print(f"  Platform: {metadata.get('platform', 'N/A')}")
        if "gpu" in metadata:
            print(f"  GPU: {metadata.get('gpu')}")
        print()

    # Print table
    print_table(results)

    # Generate graphs
    if not args.no_graphs:
        GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

        print("\nGenerating graphs...")
        generate_phase_breakdown(results, GRAPHS_DIR / "phase_breakdown.png")
        generate_total_comparison(results, GRAPHS_DIR / "total_comparison.png")
        generate_variance_plot(results, GRAPHS_DIR / "variance.png")


if __name__ == "__main__":
    main()
