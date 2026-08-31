#!/usr/bin/env python3
"""Plot the BA(2) C_train=1000, FPA=2 test-cascade experiment.

This version deliberately matches the visual style used by the existing BA(2)
sensitivity figures (for example, ``sensitivity_cascades_combined.pdf``):

  * categorical/equally spaced x positions;
  * vertical x tick labels for dense cascade sweeps;
  * blue circles for Noise-free and orange squares for Noisy;
  * light major grid;
  * combined figure title ``Sensitivity to cascades``;
  * panel titles ``Smooth L1`` and ``Acc@0.1``;
  * one shared x label below the combined figure;
  * one shared legend on the right with title ``Noise``.

Outputs:
  1. Combined two-panel figure: Smooth L1 and Accuracy@0.1
  2. Single Smooth L1 figure
  3. Single Accuracy@0.1 figure
  4. CSV containing the exact plotted values

The experiment has one training run, so no error bars or uncertainty bands are
shown. If duplicate rows are present for one cascade budget, the script warns
and averages them only to avoid silently selecting a row.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from fire import Fire


DEFAULT_ROOT = Path(
    "./results/cross_cascade_generalization"
)
CASCADE_ORDER = [5, 10, 50, 100, 500, 700, 1000]
NOISE_ORDER = ("nop", "p020_q060")
NOISE_LABELS = {
    "nop": "Noise-free",
    "p020_q060": "Noisy",
}
NOISE_STYLES = {
    "nop": {"marker": "o", "linestyle": "-"},
    "p020_q060": {"marker": "s", "linestyle": "-"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create single and combined sensitivity-style plots for the "
            "C_train=1000, FPA=2 cross-cascade test experiment."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Experiment root containing nop/ and p020_q060/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <root>/plots",
    )
    parser.add_argument("--dpi", type=int, default=400, help="PNG resolution.")
    parser.add_argument("--show", action="store_true", help="Display figures after saving.")
    return parser.parse_args()


def configure_matplotlib() -> None:
    """Use the same typography settings as plot_sensitivity.py."""
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_case(csv_path: Path, expected_noise: str) -> List[Dict[str, float]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing paired result CSV: {csv_path}")

    raw_rows: List[Dict[str, float]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"noise", "test_cascades", "l1", "acc@0.1", "acc@0.2"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

        for row in reader:
            if row["noise"] != expected_noise:
                raise ValueError(
                    f"Unexpected noise tag {row['noise']!r} in {csv_path}; "
                    f"expected {expected_noise!r}."
                )
            raw_rows.append(
                {
                    "test_cascades": int(row["test_cascades"]),
                    "l1": float(row["l1"]),
                    "acc@0.1": float(row["acc@0.1"]),
                    "acc@0.2": float(row["acc@0.2"]),
                }
            )

    if not raw_rows:
        raise ValueError(f"No rows found in {csv_path}")

    grouped: Dict[int, List[Dict[str, float]]] = {}
    for row in raw_rows:
        grouped.setdefault(int(row["test_cascades"]), []).append(row)

    rows: List[Dict[str, float]] = []
    for cascades in CASCADE_ORDER:
        entries = grouped.get(cascades, [])
        if not entries:
            raise ValueError(f"Missing C_test={cascades} in {csv_path}")
        if len(entries) > 1:
            print(
                f"WARNING: {csv_path.name} has {len(entries)} rows for "
                f"C_test={cascades}; plotting their mean without error bars."
            )
        rows.append(
            {
                "test_cascades": cascades,
                "l1": float(np.mean([r["l1"] for r in entries])),
                "acc@0.1": float(np.mean([r["acc@0.1"] for r in entries])),
                "acc@0.2": float(np.mean([r["acc@0.2"] for r in entries])),
            }
        )

    unexpected = sorted(set(grouped).difference(CASCADE_ORDER))
    if unexpected:
        print(f"WARNING: ignoring unexpected test cascade values: {unexpected}")

    return rows


def write_combined_csv(
    data: Dict[str, List[Dict[str, float]]], output_path: Path
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["noise", "test_cascades", "l1", "acc@0.1", "acc@0.2"],
        )
        writer.writeheader()
        for noise in NOISE_ORDER:
            for row in data[noise]:
                writer.writerow({"noise": noise, **row})


def categorical_axis_data() -> Tuple[List[int], np.ndarray, Dict[int, float]]:
    ordered_values = list(CASCADE_ORDER)
    positions = np.arange(len(ordered_values), dtype=float)
    x_lookup = {value: position for value, position in zip(ordered_values, positions)}
    return ordered_values, positions, x_lookup


def apply_categorical_ticks(
    ax: plt.Axes, ordered_values: Sequence[int], positions: np.ndarray
) -> None:
    """Match the cascade panel in the existing sensitivity figures."""
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [str(value) for value in ordered_values],
        rotation=90,
        ha="center",
        va="top",
    )
    if len(positions):
        ax.set_xlim(-0.45, len(positions) - 0.55)


def metric_config(metric: str) -> Mapping[str, str]:
    configs = {
        "l1": {
            "ylabel": "Smooth L1 error",
            "panel_title": "Smooth L1",
            "single_title": "Test Cascades sensitivity",
            "stem": "cross_cascade_trainC1000_fpa2_smooth_l1",
        },
        "acc@0.1": {
            "ylabel": "Accuracy@0.1",
            "panel_title": "Acc@0.1",
            "single_title": "Test Cascades sensitivity",
            "stem": "cross_cascade_trainC1000_fpa2_acc01",
        },
    }
    return configs[metric]


def add_noise_curves(
    ax: plt.Axes,
    *,
    data: Dict[str, List[Dict[str, float]]],
    x_lookup: Mapping[int, float],
    metric: str,
    collect_legend: bool = False,
):
    """Draw one-run curves with the same line/marker styling as sensitivity v5."""
    handles = []
    labels = []

    # Do not specify colors: fixed plotting order intentionally uses Matplotlib's
    # default blue/orange cycle, matching the existing sensitivity figures.
    for noise in NOISE_ORDER:
        rows = data[noise]
        xs = np.asarray([x_lookup[int(row["test_cascades"])] for row in rows], dtype=float)
        values = np.asarray([float(row[metric]) for row in rows], dtype=float)

        (line,) = ax.plot(
            xs,
            values,
            marker=NOISE_STYLES[noise]["marker"],
            linestyle=NOISE_STYLES[noise]["linestyle"],
            linewidth=1.7,
            markersize=4.8,
            label=NOISE_LABELS[noise],
            zorder=3,
        )
        if collect_legend:
            handles.append(line)
            labels.append(NOISE_LABELS[noise])

    return handles, labels


def local_y_limits(
    data: Dict[str, List[Dict[str, float]]], metric: str
) -> Tuple[float, float]:
    values = [float(row[metric]) for noise in NOISE_ORDER for row in data[noise]]
    low = min(values)
    high = max(values)
    span = max(high - low, 1e-6)
    pad = 0.10 * span
    if metric == "acc@0.1":
        return max(0.0, low - pad), min(1.005, high + pad)
    return max(0.0, low - pad), high + pad


def save_single_metric(
    *,
    data: Dict[str, List[Dict[str, float]]],
    metric: str,
    output_dir: Path,
    dpi: int,
    show: bool,
) -> Tuple[Path, Path]:
    cfg = metric_config(metric)
    ordered_values, positions, x_lookup = categorical_axis_data()

    # Exact dimensions/margins used by the existing sensitivity single plots.
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    handles, labels = add_noise_curves(
        ax,
        data=data,
        x_lookup=x_lookup,
        metric=metric,
        collect_legend=True,
    )

    ax.set_title(cfg["single_title"])
    ax.set_xlabel("Number of cascades per seed", labelpad=8)
    ax.set_ylabel(cfg["ylabel"])
    apply_categorical_ticks(ax, ordered_values, positions)
    ax.set_ylim(*local_y_limits(data, metric))
    ax.grid(True, which="major", axis="both", alpha=0.22, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(handles, labels, title="Noise", frameon=False, loc="best")

    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.30, top=0.88)

    pdf_path = output_dir / f"{cfg['stem']}.pdf"
    png_path = output_dir / f"{cfg['stem']}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=dpi)
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved PNG: {png_path}")

    if show:
        plt.show()
    plt.close(fig)
    return pdf_path, png_path


def save_combined(
    *,
    data: Dict[str, List[Dict[str, float]]],
    output_dir: Path,
    dpi: int,
    show: bool,
) -> Tuple[Path, Path]:
    ordered_values, positions, x_lookup = categorical_axis_data()

    # Match plot_sensitivity.py / sensitivity_cascades_combined.pdf.
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5), sharex=True)

    legend_handles = []
    legend_labels = []
    for panel_index, metric in enumerate(("l1", "acc@0.1")):
        ax = axes[panel_index]
        cfg = metric_config(metric)
        handles, labels = add_noise_curves(
            ax,
            data=data,
            x_lookup=x_lookup,
            metric=metric,
            collect_legend=(panel_index == 0),
        )
        if panel_index == 0:
            legend_handles = handles
            legend_labels = labels

        ax.set_title(cfg["panel_title"])
        ax.set_ylabel(cfg["ylabel"])
        apply_categorical_ticks(ax, ordered_values, positions)
        ax.set_ylim(*local_y_limits(data, metric))
        ax.grid(True, which="major", axis="both", alpha=0.22, linewidth=0.7)
        ax.set_axisbelow(True)

    # Same title/x-label/legend placement as the supplied sensitivity figure.
    fig.suptitle("Sensitivity to test sample cascades", y=0.985, fontsize=13)
    fig.supxlabel("Number of cascades per seed", y=0.035)
    fig.legend(
        legend_handles,
        legend_labels,
        title="Noise",
        loc="center left",
        bbox_to_anchor=(0.985, 0.52),
        frameon=False,
    )

    fig.subplots_adjust(left=0.09, right=0.85, bottom=0.28, top=0.83, wspace=0.28)

    pdf_path = output_dir / "cross_cascade_trainC1000_fpa2_combined.pdf"
    png_path = output_dir / "cross_cascade_trainC1000_fpa2_combined.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=dpi)
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved PNG: {png_path}")

    if show:
        plt.show()
    plt.close(fig)
    return pdf_path, png_path




def main(mode="normal", root=DEFAULT_ROOT, output_dir=None, dpi=300, show=False) -> int:

    global CASCADE_ORDER

    if mode == "small-test":
        print("Running in small-test mode: using small values for training/testing.")
        CASCADE_ORDER = [5, 50]

    if output_dir is None:
        output_dir = root / "plots"

    output_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()

    data: Dict[str, List[Dict[str, float]]] = {}
    for noise in NOISE_ORDER:
        path = root / noise / "paired_cross_cascade_metrics.csv"
        print(f"Loading {NOISE_LABELS[noise]}: {path}")
        data[noise] = load_case(path, noise)

    save_combined(
        data=data,
        output_dir=output_dir,
        dpi=dpi,
        show=show,
    )
    save_single_metric(
        data=data,
        metric="l1",
        output_dir=output_dir,
        dpi=dpi,
        show=show,
    )
    save_single_metric(
        data=data,
        metric="acc@0.1",
        output_dir=output_dir,
        dpi=dpi,
        show=show,
    )

    csv_path = output_dir / "cross_cascade_trainC1000_fpa2_plot_data.csv"
    write_combined_csv(data, csv_path)
    print(f"Saved plot data: {csv_path}")
    print("No error bars were drawn because this experiment has one run.")
    return 0


if __name__ == "__main__":
    Fire(main)
