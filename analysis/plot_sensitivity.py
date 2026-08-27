#!/usr/bin/env python3
"""Create paper-ready BA(2) sensitivity plots from completed logs.

Default outputs
---------------
A. Eight single plots (one figure per sensitivity case and metric):
   sensitivity_{assignments,fpa,cascades,nodes}_{smooth_l1,acc01}.{pdf,png}

B. Four per-case combined plots, styled like the supplied example:
   sensitivity_{assignments,fpa,cascades,nodes}_combined.{pdf,png}
   Each figure has two aligned panels: Smooth L1 and Accuracy@0.1.

C. Two all-case combined plots retained from v4:
   sensitivity_all_smooth_l1.{pdf,png}
   sensitivity_all_acc01.{pdf,png}
   Each figure has four panels: Assignments, FPA, Cascades, Nodes.

D. sensitivity_test_unseen_summary.csv

Important plotting choice
-------------------------
Every x-axis uses categorical/equally spaced positions. The actual tested values
are tick labels, but numerical magnitude does not control horizontal spacing.
This keeps dense sweeps such as 200, 500, 700, 1000 cleanly aligned.

Metrics are taken from the final/last K-RUN SUMMARY -> test_unseen partition.
Completed K=1 logs that have no K-RUN SUMMARY are accepted only when the last
explicit run marker is RUN 1/1; in that case the last completed Test unseen line
is used and std=0.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ROOT = Path("/scratch/svc_td_fincomp/vrango/hic_new/sensitivity_ba2")

CASE_ORDER = ("assignments", "fpa", "cascades", "nodes")
CASES: Dict[str, Dict[str, str]] = {
    "assignments": {
        "panel_title": "Assignments",
        "xlabel": "Number of assignments",
    },
    "fpa": {
        "panel_title": "FPA",
        "xlabel": "Feature realizations per assignment",
    },
    "cascades": {
        "panel_title": "Cascades",
        "xlabel": "Number of cascades per seed",
    },
    "nodes": {
        "panel_title": "Nodes",
        "xlabel": "Number of nodes",
    },
}

NOISE_LABELS = {
    "nop": "Noise-free",
    "p020_q060": "Noisy",
}
NOISE_ORDER = ("nop", "p020_q060")

FLOAT_RE = r"(?:[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|inf|-inf)"
PM_RE = r"(?:\+\-|\+/-|±)"


@dataclass(frozen=True)
class Result:
    case: str
    x: int
    noise: str
    smooth_l1_mean: float
    smooth_l1_std: float
    acc01_mean: float
    acc01_std: float
    log_file: Path
    fpa_master_tag: Optional[str] = None
    k_runs: int = 0
    parse_source: str = "k_run_summary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create single and combined BA(2) sensitivity figures for Smooth L1 "
            "and Accuracy@0.1."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root containing assignments/fpa/cascades/nodes/logs directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <root>/final_plots",
    )
    parser.add_argument(
        "--fpa-master-tag",
        default="auto",
        help=(
            "Filter FPA logs by filename tag, e.g. masterfpa100 or "
            "masterfpa1000. Default: auto."
        ),
    )
    parser.add_argument("--dpi", type=int, default=400, help="PNG resolution.")
    parser.add_argument(
        "--show", action="store_true", help="Display figures after saving."
    )
    parser.add_argument(
        "--strict-noise-pairs",
        action="store_true",
        help="Fail if one noise condition is missing at any x value.",
    )
    parser.add_argument(
        "--independent-y",
        action="store_true",
        help=(
            "Allow every panel to choose its own y limits. By default all four "
            "panels in an all-case metric figure share the same y scale."
        ),
    )
    parser.add_argument(
        "--skip-singles",
        action="store_true",
        help="Do not create the eight individual single-panel figures.",
    )
    parser.add_argument(
        "--skip-case-combined",
        action="store_true",
        help="Do not create the four two-panel per-sensitivity combined figures.",
    )
    parser.add_argument(
        "--skip-all-combined",
        action="store_true",
        help="Do not create the two four-panel all-sensitivity metric figures.",
    )
    return parser.parse_args()


def last_k_run_summary(text: str, path: Path) -> str:
    marker = "========== K-RUN SUMMARY =========="
    index = text.rfind(marker)
    if index < 0:
        raise ValueError(f"No K-RUN SUMMARY found in {path}")
    return text[index + len(marker) :]


def partition_block(summary: str, partition: str, path: Path) -> str:
    match = re.search(
        rf"(?ms)^\s*{re.escape(partition)}\s*$\n(?P<body>.*?)(?=^\S|\Z)",
        summary,
    )
    if not match:
        raise ValueError(
            f"Partition {partition!r} not found in final K-RUN SUMMARY of {path}"
        )
    return match.group("body")


def metric_mean_std(block: str, metric: str, path: Path) -> Tuple[float, float]:
    match = re.search(
        rf"(?mi)^\s*{re.escape(metric)}\s*:\s*({FLOAT_RE})\s*{PM_RE}\s*({FLOAT_RE})\s*$",
        block,
    )
    if not match:
        raise ValueError(f"Metric {metric!r} not found in test_unseen summary of {path}")
    return float(match.group(1)), float(match.group(2))


def parse_noise(filename: str) -> str:
    for noise in NOISE_ORDER:
        if f"_{noise}_" in filename:
            return noise
    raise ValueError(f"Could not infer noise setting from filename: {filename}")


def parse_x(case: str, filename: str) -> int:
    match = re.search(rf"sens_ba2_{re.escape(case)}_(\d+)_", filename)
    if not match:
        raise ValueError(f"Could not infer {case} value from filename: {filename}")
    return int(match.group(1))


def parse_fpa_master_tag(filename: str) -> Optional[str]:
    match = re.search(r"_(masterfpa\d+)(?:_|\.)", filename)
    return match.group(1) if match else None


def infer_last_run_progress(text: str) -> Optional[Tuple[int, int]]:
    matches = re.findall(r"==========\s*RUN\s+(\d+)\s*/\s*(\d+)\s*==========", text)
    if not matches:
        return None
    current, total = matches[-1]
    return int(current), int(total)


def single_run_test_unseen(text: str, path: Path) -> Tuple[float, float]:
    progress = infer_last_run_progress(text)
    if progress != (1, 1):
        if progress is None:
            raise ValueError(
                f"No K-RUN SUMMARY and no explicit RUN 1/1 marker found in {path}"
            )
        raise ValueError(
            f"No K-RUN SUMMARY in {path}; last run marker is RUN "
            f"{progress[0]}/{progress[1]}, so this is not accepted as K=1"
        )

    run_markers = list(re.finditer(r"==========\s*RUN\s+1\s*/\s*1\s*==========", text))
    run_text = text[run_markers[-1].end() :] if run_markers else text
    pattern = re.compile(
        rf"(?mi)^\s*Test\s+unseen\s*\|\s*"
        rf"Loss:\s*({FLOAT_RE})\s*;.*?"
        rf"Acc@0\.1:\s*({FLOAT_RE})(?:\s*;|\s*$)"
    )
    matches = list(pattern.finditer(run_text))
    if not matches:
        raise ValueError(
            f"RUN 1/1 found but no completed Test unseen line found in {path}"
        )
    match = matches[-1]
    return float(match.group(1)), float(match.group(2))


def parse_log(case: str, path: Path) -> Result:
    text = path.read_text(encoding="utf-8", errors="replace")
    progress = infer_last_run_progress(text)
    k_runs = progress[1] if progress is not None else 0
    parse_source = "k_run_summary"

    try:
        summary = last_k_run_summary(text, path)
        block = partition_block(summary, "test_unseen", path)
        loss_mean, loss_std = metric_mean_std(block, "loss", path)
        acc_mean, acc_std = metric_mean_std(block, "acc@0.1", path)
    except ValueError as summary_error:
        if "No K-RUN SUMMARY" not in str(summary_error):
            raise
        loss_mean, acc_mean = single_run_test_unseen(text, path)
        loss_std = 0.0
        acc_std = 0.0
        k_runs = 1
        parse_source = "single_run_test_unseen_fallback"

    values = (loss_mean, loss_std, acc_mean, acc_std)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite final metric in {path}")

    return Result(
        case=case,
        x=parse_x(case, path.name),
        noise=parse_noise(path.name),
        smooth_l1_mean=loss_mean,
        smooth_l1_std=loss_std,
        acc01_mean=acc_mean,
        acc01_std=acc_std,
        log_file=path.resolve(),
        fpa_master_tag=parse_fpa_master_tag(path.name) if case == "fpa" else None,
        k_runs=k_runs,
        parse_source=parse_source,
    )


def collect_case(case: str, root: Path) -> Tuple[List[Result], List[str]]:
    log_dir = root / case / "logs"
    if not log_dir.is_dir():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    results: List[Result] = []
    warnings: List[str] = []
    for path in sorted(log_dir.glob(f"sens_ba2_{case}_*.log")):
        try:
            results.append(parse_log(case, path))
        except Exception as exc:
            warnings.append(f"SKIP {path.name}: {exc}")

    if not results:
        detail = "\n".join(warnings[:10])
        raise RuntimeError(
            f"No completed {case} logs could be parsed in {log_dir}\n{detail}"
        )
    return results, warnings


def apply_fpa_master_filter(results: List[Result], requested: str) -> List[Result]:
    if requested != "auto":
        filtered = [result for result in results if result.fpa_master_tag == requested]
        if not filtered:
            available = sorted({result.fpa_master_tag for result in results}, key=str)
            raise ValueError(
                f"No FPA logs match --fpa-master-tag={requested!r}. "
                f"Detected tags: {available}"
            )
        print(f"FPA master filter: {requested}")
        return filtered

    detected = sorted({result.fpa_master_tag for result in results}, key=str)
    if len(detected) > 1:
        raise ValueError(
            "Multiple FPA master datasets were detected: "
            f"{detected}. Select one with --fpa-master-tag."
        )
    if detected:
        print(f"FPA master detected: {detected[0]}")
    return results


def validate_unique(results: Sequence[Result]) -> None:
    seen: Dict[Tuple[str, int, str], Path] = {}
    for result in results:
        key = (result.case, result.x, result.noise)
        if key in seen:
            raise ValueError(
                "Duplicate completed logs for the same sensitivity point:\n"
                f"  {seen[key]}\n  {result.log_file}"
            )
        seen[key] = result.log_file


def check_noise_pairing(
    results: Sequence[Result], case: str, *, strict: bool
) -> List[str]:
    by_noise = {
        noise: {result.x for result in results if result.noise == noise}
        for noise in NOISE_ORDER
    }
    for noise, values in by_noise.items():
        if not values:
            raise ValueError(f"No {NOISE_LABELS[noise]} results found for {case}.")

    if by_noise["nop"] == by_noise["p020_q060"]:
        return []

    only_nop = sorted(by_noise["nop"] - by_noise["p020_q060"])
    only_noisy = sorted(by_noise["p020_q060"] - by_noise["nop"])
    message = (
        f"Noise curves are not fully paired for {case}. "
        f"Only noise-free: {only_nop}; only noisy: {only_noisy}."
    )
    if strict:
        raise ValueError(message)
    return [message]


def write_csv(results: Sequence[Result], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sensitivity",
                "value",
                "noise",
                "smooth_l1_mean",
                "smooth_l1_std",
                "acc_at_0.1_mean",
                "acc_at_0.1_std",
                "fpa_master_tag",
                "k_runs",
                "parse_source",
                "log_file",
            ]
        )
        for result in sorted(results, key=lambda item: (item.case, item.x, item.noise)):
            writer.writerow(
                [
                    result.case,
                    result.x,
                    result.noise,
                    f"{result.smooth_l1_mean:.10g}",
                    f"{result.smooth_l1_std:.10g}",
                    f"{result.acc01_mean:.10g}",
                    f"{result.acc01_std:.10g}",
                    result.fpa_master_tag or "",
                    result.k_runs,
                    result.parse_source,
                    str(result.log_file),
                ]
            )


def configure_matplotlib() -> None:
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


def tick_rotation(case: str, n_values: int) -> int:
    # Dense sweeps need vertical labels; sparse node sweeps read better horizontally.
    if case == "nodes" and n_values <= 5:
        return 0
    if n_values >= 7:
        return 90
    return 45


def metric_config(metric: str) -> Mapping[str, str]:
    configs = {
        "smooth_l1": {
            "mean": "smooth_l1_mean",
            "std": "smooth_l1_std",
            "ylabel": "Smooth L1 error",
            "suptitle": "Sensitivity analysis — Smooth L1 error",
            "stem": "sensitivity_all_smooth_l1",
        },
        "acc01": {
            "mean": "acc01_mean",
            "std": "acc01_std",
            "ylabel": "Accuracy@0.1",
            "suptitle": "Sensitivity analysis — Accuracy@0.1",
            "stem": "sensitivity_all_acc01",
        },
    }
    return configs[metric]


def global_y_limits(results: Sequence[Result], metric: str) -> Tuple[float, float]:
    cfg = metric_config(metric)
    mean_field = cfg["mean"]
    std_field = cfg["std"]

    lows = [getattr(result, mean_field) - getattr(result, std_field) for result in results]
    highs = [getattr(result, mean_field) + getattr(result, std_field) for result in results]
    low = min(lows)
    high = max(highs)
    span = max(high - low, 1e-6)
    pad = 0.08 * span

    if metric == "acc01":
        return max(0.0, low - pad), min(1.005, high + pad)
    return max(0.0, low - pad), high + pad



def categorical_axis_data(
    case_results: Sequence[Result],
) -> Tuple[List[int], np.ndarray, Dict[int, float]]:
    ordered_values = sorted({result.x for result in case_results})
    positions = np.arange(len(ordered_values), dtype=float)
    x_lookup = {value: position for value, position in zip(ordered_values, positions)}
    return ordered_values, positions, x_lookup


def apply_categorical_ticks(ax, case: str, ordered_values: Sequence[int], positions: np.ndarray) -> None:
    ax.set_xticks(positions)
    rotation = tick_rotation(case, len(ordered_values))
    ax.set_xticklabels(
        [str(value) for value in ordered_values],
        rotation=rotation,
        ha="center" if rotation == 90 else ("right" if rotation else "center"),
        va="top" if rotation else "center_baseline",
    )
    if len(positions):
        ax.set_xlim(-0.45, len(positions) - 0.55)


def add_noise_curves(
    ax,
    *,
    case_results: Sequence[Result],
    x_lookup: Mapping[int, float],
    metric: str,
    collect_legend: bool = False,
):
    cfg = metric_config(metric)
    mean_field = cfg["mean"]
    std_field = cfg["std"]
    noise_styles = {
        "nop": {"marker": "o", "linestyle": "-"},
        "p020_q060": {"marker": "s", "linestyle": "-"},
    }

    handles = []
    labels = []
    for noise in NOISE_ORDER:
        group = sorted(
            (result for result in case_results if result.noise == noise),
            key=lambda result: result.x,
        )
        if not group:
            continue

        xs = np.asarray([x_lookup[result.x] for result in group], dtype=float)
        means = np.asarray([getattr(result, mean_field) for result in group], dtype=float)
        stds = np.asarray([getattr(result, std_field) for result in group], dtype=float)

        container = ax.errorbar(
            xs,
            means,
            yerr=stds,
            marker=noise_styles[noise]["marker"],
            linestyle=noise_styles[noise]["linestyle"],
            linewidth=1.7,
            markersize=4.8,
            capsize=3.0,
            elinewidth=1.0,
            label=NOISE_LABELS[noise],
            zorder=3,
        )
        line = container.lines[0]
        line_color = line.get_color()
        ax.fill_between(
            xs,
            means - stds,
            means + stds,
            color=line_color,
            alpha=0.10,
            linewidth=0,
            zorder=1,
        )
        if collect_legend:
            handles.append(line)
            labels.append(NOISE_LABELS[noise])

    return handles, labels


def local_y_limits(case_results: Sequence[Result], metric: str) -> Tuple[float, float]:
    cfg = metric_config(metric)
    mean_field = cfg["mean"]
    std_field = cfg["std"]
    lows = [getattr(result, mean_field) - getattr(result, std_field) for result in case_results]
    highs = [getattr(result, mean_field) + getattr(result, std_field) for result in case_results]
    low = min(lows)
    high = max(highs)
    span = max(high - low, 1e-6)
    pad = 0.10 * span
    if metric == "acc01":
        return max(0.0, low - pad), min(1.005, high + pad)
    return max(0.0, low - pad), high + pad


def plot_single_metric(
    *,
    case_results: Sequence[Result],
    case: str,
    metric: str,
    output_dir: Path,
    dpi: int,
    show: bool,
) -> Tuple[Path, Path]:
    cfg = metric_config(metric)
    ordered_values, positions, x_lookup = categorical_axis_data(case_results)

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    handles, labels = add_noise_curves(
        ax,
        case_results=case_results,
        x_lookup=x_lookup,
        metric=metric,
        collect_legend=True,
    )

    ax.set_title(f"{CASES[case]['panel_title']} sensitivity")
    ax.set_xlabel(CASES[case]["xlabel"], labelpad=8)
    ax.set_ylabel(cfg["ylabel"])
    apply_categorical_ticks(ax, case, ordered_values, positions)
    ax.set_ylim(*local_y_limits(case_results, metric))
    ax.grid(True, which="major", axis="both", alpha=0.22, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(handles, labels, title="Noise", frameon=False, loc="best")

    bottom = 0.30 if tick_rotation(case, len(ordered_values)) == 90 else 0.18
    fig.subplots_adjust(left=0.16, right=0.97, bottom=bottom, top=0.88)

    stem = f"sensitivity_{case}_{cfg['stem'].replace('sensitivity_all_', '')}"
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=dpi)

    if show:
        plt.show()
    plt.close(fig)
    return pdf_path, png_path


def plot_case_combined(
    *,
    case_results: Sequence[Result],
    case: str,
    output_dir: Path,
    dpi: int,
    show: bool,
) -> Tuple[Path, Path]:
    ordered_values, positions, x_lookup = categorical_axis_data(case_results)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5), sharex=True)

    legend_handles = []
    legend_labels = []
    for panel_index, metric in enumerate(("smooth_l1", "acc01")):
        ax = axes[panel_index]
        cfg = metric_config(metric)
        handles, labels = add_noise_curves(
            ax,
            case_results=case_results,
            x_lookup=x_lookup,
            metric=metric,
            collect_legend=(panel_index == 0),
        )
        if panel_index == 0:
            legend_handles = handles
            legend_labels = labels

        ax.set_title("Smooth L1" if metric == "smooth_l1" else "Acc@0.1")
        ax.set_ylabel(cfg["ylabel"])
        apply_categorical_ticks(ax, case, ordered_values, positions)
        ax.set_ylim(*local_y_limits(case_results, metric))
        ax.grid(True, which="major", axis="both", alpha=0.22, linewidth=0.7)
        ax.set_axisbelow(True)

    fig.suptitle(f"Sensitivity to {CASES[case]['panel_title'].lower()}", y=0.985, fontsize=13)
    fig.supxlabel(CASES[case]["xlabel"], y=0.035)
    fig.legend(
        legend_handles,
        legend_labels,
        title="Noise",
        loc="center left",
        bbox_to_anchor=(0.985, 0.52),
        frameon=False,
    )

    bottom = 0.28 if tick_rotation(case, len(ordered_values)) == 90 else 0.18
    fig.subplots_adjust(left=0.09, right=0.85, bottom=bottom, top=0.83, wspace=0.28)

    stem = f"sensitivity_{case}_combined"
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=dpi)

    if show:
        plt.show()
    plt.close(fig)
    return pdf_path, png_path

def plot_four_panel_metric(
    *,
    results: Sequence[Result],
    metric: str,
    output_dir: Path,
    dpi: int,
    show: bool,
    independent_y: bool,
) -> Tuple[Path, Path]:
    cfg = metric_config(metric)
    mean_field = cfg["mean"]
    std_field = cfg["std"]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(15.8, 4.7),
        sharey=not independent_y,
    )

    # Use Matplotlib's default color cycle; the two conditions stay consistent
    # across all panels because the plotting order is fixed.
    noise_styles = {
        "nop": {"marker": "o", "linestyle": "-"},
        "p020_q060": {"marker": "s", "linestyle": "-"},
    }

    legend_handles = []
    legend_labels = []

    for panel_index, case in enumerate(CASE_ORDER):
        ax = axes[panel_index]
        case_results = [result for result in results if result.case == case]
        ordered_values = sorted({result.x for result in case_results})

        # Equal/categorical positions are the key alignment fix.
        positions = np.arange(len(ordered_values), dtype=float)
        x_lookup = {value: position for value, position in zip(ordered_values, positions)}

        for noise in NOISE_ORDER:
            group = sorted(
                (result for result in case_results if result.noise == noise),
                key=lambda result: result.x,
            )
            if not group:
                continue

            xs = np.asarray([x_lookup[result.x] for result in group], dtype=float)
            means = np.asarray([getattr(result, mean_field) for result in group], dtype=float)
            stds = np.asarray([getattr(result, std_field) for result in group], dtype=float)

            container = ax.errorbar(
                xs,
                means,
                yerr=stds,
                marker=noise_styles[noise]["marker"],
                linestyle=noise_styles[noise]["linestyle"],
                linewidth=1.7,
                markersize=4.8,
                capsize=3.0,
                elinewidth=1.0,
                label=NOISE_LABELS[noise],
                zorder=3,
            )

            line = container.lines[0]
            line_color = line.get_color()
            ax.fill_between(
                xs,
                means - stds,
                means + stds,
                color=line_color,
                alpha=0.10,
                linewidth=0,
                zorder=1,
            )

            if panel_index == 0:
                legend_handles.append(line)
                legend_labels.append(NOISE_LABELS[noise])

        ax.set_title(CASES[case]["panel_title"])
        ax.set_xlabel(CASES[case]["xlabel"], labelpad=8)
        ax.set_xticks(positions)
        rotation = tick_rotation(case, len(ordered_values))
        ax.set_xticklabels(
            [str(value) for value in ordered_values],
            rotation=rotation,
            ha="center" if rotation == 90 else ("right" if rotation else "center"),
            va="top" if rotation else "center_baseline",
        )

        # Categorical padding keeps the first and last markers away from the frame.
        if len(positions):
            ax.set_xlim(-0.45, len(positions) - 0.55)

        ax.grid(True, which="major", axis="both", alpha=0.22, linewidth=0.7)
        ax.set_axisbelow(True)

        if independent_y:
            if metric == "acc01":
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(max(0.0, ymin), min(1.005, max(1.0, ymax)))
        elif panel_index == 0:
            ax.set_ylabel(cfg["ylabel"])

    if independent_y:
        axes[0].set_ylabel(cfg["ylabel"])
    else:
        ymin, ymax = global_y_limits(results, metric)
        for ax in axes:
            ax.set_ylim(ymin, ymax)

    fig.suptitle(cfg["suptitle"], y=0.985, fontsize=13)
    fig.legend(
        legend_handles,
        legend_labels,
        title="Noise",
        loc="center left",
        bbox_to_anchor=(0.995, 0.52),
        frameon=False,
    )

    # Reserve bottom space for vertical tick labels and right space for legend.
    fig.subplots_adjust(left=0.065, right=0.89, bottom=0.28, top=0.84, wspace=0.17)

    pdf_path = output_dir / f"{cfg['stem']}.pdf"
    png_path = output_dir / f"{cfg['stem']}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=dpi)

    if show:
        plt.show()
    plt.close(fig)
    return pdf_path, png_path


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / "final_plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()

    all_results: List[Result] = []
    all_warnings: List[str] = []

    for case in CASE_ORDER:
        results, warnings = collect_case(case, root)
        if case == "fpa":
            results = apply_fpa_master_filter(results, args.fpa_master_tag)
        validate_unique(results)
        pairing_warnings = check_noise_pairing(
            results, case, strict=args.strict_noise_pairs
        )

        all_results.extend(results)
        all_warnings.extend(f"[{case}] {warning}" for warning in warnings)
        all_warnings.extend(f"[{case}] {warning}" for warning in pairing_warnings)

        values = sorted({result.x for result in results})
        print(f"{case:12s}: {values}")

    csv_path = output_dir / "sensitivity_test_unseen_summary.csv"
    write_csv(all_results, csv_path)

    created: List[Path] = [csv_path]

    if not args.skip_singles:
        for case in CASE_ORDER:
            case_results = [result for result in all_results if result.case == case]
            for metric in ("smooth_l1", "acc01"):
                pdf_path, png_path = plot_single_metric(
                    case_results=case_results,
                    case=case,
                    metric=metric,
                    output_dir=output_dir,
                    dpi=args.dpi,
                    show=args.show,
                )
                created.extend([pdf_path, png_path])

    if not args.skip_case_combined:
        for case in CASE_ORDER:
            case_results = [result for result in all_results if result.case == case]
            pdf_path, png_path = plot_case_combined(
                case_results=case_results,
                case=case,
                output_dir=output_dir,
                dpi=args.dpi,
                show=args.show,
            )
            created.extend([pdf_path, png_path])

    if not args.skip_all_combined:
        for metric in ("smooth_l1", "acc01"):
            pdf_path, png_path = plot_four_panel_metric(
                results=all_results,
                metric=metric,
                output_dir=output_dir,
                dpi=args.dpi,
                show=args.show,
                independent_y=args.independent_y,
            )
            created.extend([pdf_path, png_path])

    print("\nCreated:")
    for path in created:
        print(f"  {path}")

    if all_warnings:
        print("\nWarnings / skipped non-final logs:")
        for warning in all_warnings:
            print(f"  {warning}")

    print("\nMetrics plotted: final K-RUN SUMMARY -> test_unseen only;")
    print("completed K=1 logs without a summary use the strict RUN 1/1 fallback.")
    print("X positions are categorical/equally spaced for clean tick alignment.")
    print("Default output includes singles, per-case 2-panel combined figures, and all-case combined figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
