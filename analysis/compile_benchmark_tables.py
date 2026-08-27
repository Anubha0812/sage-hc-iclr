#!/usr/bin/env python3
"""Compile benchmark performance and uncertainty tables from training logs.

This script is tailored to log files named like:
    sens_ba2_assignments_5000_nop_..._uncertainty.log

It reads the LAST completed K-run summary in each log, so older failed/retried
blocks earlier in an appended log do not affect the table.

Performance table (test_unseen only):
    - Smooth L1 = logged `loss`
    - Acc@0.1
    - mean and std across K runs

Uncertainty table (test_unseen only):
    - mean_ci_width
    - mean_prediction_std
    - mean and std across K runs

Median uncertainty metrics are intentionally ignored.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


GRAPH_ORDER = ["tree", "karate", "ba2", "ba3", "ba4"]
GRAPH_LABEL = {
    "tree": "Tree",
    "karate": "Karate",
    "ba2": "BA(2)",
    "ba3": "BA(3)",
    "ba4": "BA(4)",
}

NOISE_ORDER = ["nop", "p020_q060"]
NOISE_LABEL = {
    "nop": "No noise",
    "p020_q060": "Noisy",
}

FLOAT = r"[-+0-9.eE]+"

FILENAME_RE = re.compile(
    r"^sens_(?P<graph>tree|karate|ba2|ba3|ba4)_"
    r"assignments_(?P<assignments>\d+)_"
    r"(?P<noise>nop|p020_q060)_.*\.log$"
)

PERFORMANCE_RE = re.compile(
    rf"^test_unseen\s*$\n"
    rf"\s*loss\s*:\s*(?P<loss_mean>{FLOAT})\s*\+-\s*(?P<loss_std>{FLOAT})\s*$\n"
    rf"\s*l1\s*:\s*{FLOAT}\s*\+-\s*{FLOAT}\s*$\n"
    rf"\s*acc@0\.1\s*:\s*(?P<acc01_mean>{FLOAT})\s*\+-\s*(?P<acc01_std>{FLOAT})\s*$",
    re.MULTILINE,
)

UNCERTAINTY_RE = re.compile(
    rf"^uncertainty_test_unseen\s*$\n"
    rf"\s*assignments\s*:\s*(?P<test_assignments>\d+)\s*$\n"
    rf"\s*mean_ci_width\s*:\s*(?P<ci_mean>{FLOAT})\s*\+-\s*(?P<ci_std>{FLOAT})\s*$\n"
    rf"\s*median_ci_width\s*:\s*{FLOAT}\s*\+-\s*{FLOAT}\s*$\n"
    rf"\s*mean_prediction_std\s*:\s*(?P<predstd_mean>{FLOAT})\s*\+-\s*(?P<predstd_std>{FLOAT})\s*$",
    re.MULTILINE,
)

K_RUN_HEADER = "========== K-RUN SUMMARY =========="


@dataclass(frozen=True)
class Result:
    graph: str
    assignments: int
    noise: str
    log_file: Path
    smooth_l1_mean: float
    smooth_l1_std: float
    acc01_mean: float
    acc01_std: float
    uncertainty_assignments: int
    mean_ci_width_mean: float
    mean_ci_width_std: float
    mean_prediction_std_mean: float
    mean_prediction_std_std: float


def natural_sort_key(result: Result) -> Tuple[int, int, int]:
    return (
        result.assignments,
        GRAPH_ORDER.index(result.graph),
        NOISE_ORDER.index(result.noise),
    )


def last_k_run_section(text: str) -> str:
    """Return text starting at the final K-run summary in an appended log."""
    pos = text.rfind(K_RUN_HEADER)
    if pos < 0:
        raise ValueError("No K-RUN SUMMARY found")
    return text[pos:]


def parse_log(path: Path) -> Result:
    filename_match = FILENAME_RE.match(path.name)
    if not filename_match:
        raise ValueError(f"Filename does not match expected benchmark pattern: {path.name}")

    text = path.read_text(encoding="utf-8", errors="replace")
    section = last_k_run_section(text)

    perf_match = PERFORMANCE_RE.search(section)
    if not perf_match:
        raise ValueError("Could not find test_unseen K-run performance summary")

    unc_match = UNCERTAINTY_RE.search(section)
    if not unc_match:
        raise ValueError("Could not find uncertainty_test_unseen K-run summary")

    return Result(
        graph=filename_match.group("graph"),
        assignments=int(filename_match.group("assignments")),
        noise=filename_match.group("noise"),
        log_file=path,
        smooth_l1_mean=float(perf_match.group("loss_mean")),
        smooth_l1_std=float(perf_match.group("loss_std")),
        acc01_mean=float(perf_match.group("acc01_mean")),
        acc01_std=float(perf_match.group("acc01_std")),
        uncertainty_assignments=int(unc_match.group("test_assignments")),
        mean_ci_width_mean=float(unc_match.group("ci_mean")),
        mean_ci_width_std=float(unc_match.group("ci_std")),
        mean_prediction_std_mean=float(unc_match.group("predstd_mean")),
        mean_prediction_std_std=float(unc_match.group("predstd_std")),
    )


def collect_results(log_dir: Path, assignments: Optional[int]) -> Tuple[List[Result], List[str]]:
    results: List[Result] = []
    warnings: List[str] = []

    for path in sorted(log_dir.glob("sens_*_assignments_*_*.log")):
        match = FILENAME_RE.match(path.name)
        if not match:
            continue
        if assignments is not None and int(match.group("assignments")) != assignments:
            continue

        try:
            results.append(parse_log(path))
        except Exception as exc:  # keep other completed logs usable
            warnings.append(f"SKIP {path.name}: {exc}")

    results.sort(key=natural_sort_key)
    return results, warnings


def fmt_pm(mean: float, std: float, digits: int = 5) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def write_performance_csv(results: Iterable[Result], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "assignments",
                "graph",
                "noise",
                "smooth_l1_mean",
                "smooth_l1_std",
                "acc01_mean",
                "acc01_std",
                "smooth_l1_mean_pm_std",
                "acc01_mean_pm_std",
                "log_file",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.assignments,
                    GRAPH_LABEL[r.graph],
                    NOISE_LABEL[r.noise],
                    f"{r.smooth_l1_mean:.8g}",
                    f"{r.smooth_l1_std:.8g}",
                    f"{r.acc01_mean:.8g}",
                    f"{r.acc01_std:.8g}",
                    fmt_pm(r.smooth_l1_mean, r.smooth_l1_std),
                    fmt_pm(r.acc01_mean, r.acc01_std),
                    str(r.log_file),
                ]
            )


def write_uncertainty_csv(results: Iterable[Result], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "assignments",
                "graph",
                "noise",
                "test_unseen_assignments",
                "mean_ci_width_mean",
                "mean_ci_width_std",
                "mean_prediction_std_mean",
                "mean_prediction_std_std",
                "mean_ci_width_mean_pm_std",
                "mean_prediction_std_mean_pm_std",
                "log_file",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.assignments,
                    GRAPH_LABEL[r.graph],
                    NOISE_LABEL[r.noise],
                    r.uncertainty_assignments,
                    f"{r.mean_ci_width_mean:.8g}",
                    f"{r.mean_ci_width_std:.8g}",
                    f"{r.mean_prediction_std_mean:.8g}",
                    f"{r.mean_prediction_std_std:.8g}",
                    fmt_pm(r.mean_ci_width_mean, r.mean_ci_width_std),
                    fmt_pm(r.mean_prediction_std_mean, r.mean_prediction_std_std),
                    str(r.log_file),
                ]
            )


def result_map(results: Iterable[Result], assignments: int) -> Dict[Tuple[str, str], Result]:
    return {
        (r.graph, r.noise): r
        for r in results
        if r.assignments == assignments
    }


def latex_pm(mean: float, std: float, digits: int = 5) -> str:
    return f"${mean:.{digits}f} \\pm {std:.{digits}f}$"


def write_performance_latex(results: List[Result], assignments: int, path: Path) -> None:
    by_key = result_map(results, assignments)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Test-unseen performance for the benchmark setting. Smooth L1 corresponds to the logged loss. Values are mean $\pm$ standard deviation across independent training runs.}",
        rf"\label{{tab:benchmark_performance_{assignments}}}",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"& \multicolumn{2}{c}{No noise} & \multicolumn{2}{c}{Noisy} \\",
        r"Graph & Smooth L1 & Acc@0.1 & Smooth L1 & Acc@0.1 \\",
        r"\hline",
    ]

    for graph in GRAPH_ORDER:
        cells = [GRAPH_LABEL[graph]]
        for noise in NOISE_ORDER:
            r = by_key.get((graph, noise))
            if r is None:
                cells.extend(["--", "--"])
            else:
                cells.extend(
                    [
                        latex_pm(r.smooth_l1_mean, r.smooth_l1_std),
                        latex_pm(r.acc01_mean, r.acc01_std),
                    ]
                )
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_uncertainty_latex(results: List[Result], assignments: int, path: Path) -> None:
    by_key = result_map(results, assignments)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Test-unseen uncertainty for the benchmark setting. Only mean CI width and mean prediction standard deviation are reported. Values are mean $\pm$ standard deviation across independent training runs.}",
        rf"\label{{tab:benchmark_uncertainty_{assignments}}}",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"& \multicolumn{2}{c}{No noise} & \multicolumn{2}{c}{Noisy} \\",
        r"Graph & Mean CI width & Mean pred. std & Mean CI width & Mean pred. std \\",
        r"\hline",
    ]

    for graph in GRAPH_ORDER:
        cells = [GRAPH_LABEL[graph]]
        for noise in NOISE_ORDER:
            r = by_key.get((graph, noise))
            if r is None:
                cells.extend(["--", "--"])
            else:
                cells.extend(
                    [
                        latex_pm(r.mean_ci_width_mean, r.mean_ci_width_std),
                        latex_pm(r.mean_prediction_std_mean, r.mean_prediction_std_std),
                    ]
                )
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def print_tables(results: List[Result], assignments: int) -> None:
    by_key = result_map(results, assignments)

    print("\nPERFORMANCE: TEST UNSEEN")
    print("Graph    Noise       Smooth L1 (mean ± std)    Acc@0.1 (mean ± std)")
    print("-" * 76)
    for graph in GRAPH_ORDER:
        for noise in NOISE_ORDER:
            r = by_key.get((graph, noise))
            if r is None:
                continue
            print(
                f"{GRAPH_LABEL[graph]:8s} {NOISE_LABEL[noise]:10s} "
                f"{fmt_pm(r.smooth_l1_mean, r.smooth_l1_std):24s} "
                f"{fmt_pm(r.acc01_mean, r.acc01_std)}"
            )

    print("\nUNCERTAINTY: TEST UNSEEN")
    print("Graph    Noise       Mean CI width (mean ± std) Mean pred std (mean ± std)")
    print("-" * 82)
    for graph in GRAPH_ORDER:
        for noise in NOISE_ORDER:
            r = by_key.get((graph, noise))
            if r is None:
                continue
            print(
                f"{GRAPH_LABEL[graph]:8s} {NOISE_LABEL[noise]:10s} "
                f"{fmt_pm(r.mean_ci_width_mean, r.mean_ci_width_std):26s} "
                f"{fmt_pm(r.mean_prediction_std_mean, r.mean_prediction_std_std)}"
            )


def expected_keys(assignments: int) -> List[Tuple[str, str]]:
    return [(graph, noise) for graph in GRAPH_ORDER for noise in NOISE_ORDER]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dir",
        type=Path,
        required=True,
        help="Directory containing sens_* benchmark .log files.",
    )
    parser.add_argument(
        "--assignments",
        type=int,
        default=5000,
        help="Assignment count to include in the paper tables (default: 5000).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_tables"),
        help="Output directory for CSV and LaTeX tables.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any of the expected 5 graphs x 2 noise settings is missing.",
    )
    args = parser.parse_args()

    if not args.log_dir.is_dir():
        raise SystemExit(f"Log directory does not exist: {args.log_dir}")

    results, warnings = collect_results(args.log_dir, args.assignments)
    if not results:
        raise SystemExit(
            f"No completed matching logs found for assignments={args.assignments} "
            f"under {args.log_dir}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    performance_csv = args.output_dir / f"benchmark_performance_{args.assignments}.csv"
    uncertainty_csv = args.output_dir / f"benchmark_uncertainty_{args.assignments}.csv"
    performance_tex = args.output_dir / f"benchmark_performance_{args.assignments}_table.tex"
    uncertainty_tex = args.output_dir / f"benchmark_uncertainty_{args.assignments}_table.tex"

    write_performance_csv(results, performance_csv)
    write_uncertainty_csv(results, uncertainty_csv)
    write_performance_latex(results, args.assignments, performance_tex)
    write_uncertainty_latex(results, args.assignments, uncertainty_tex)
    print_tables(results, args.assignments)

    found = {(r.graph, r.noise) for r in results if r.assignments == args.assignments}
    missing = [key for key in expected_keys(args.assignments) if key not in found]

    if warnings:
        print("\nWarnings:")
        for message in warnings:
            print(f"  {message}")

    if missing:
        print("\nMissing expected benchmark logs:")
        for graph, noise in missing:
            print(f"  {GRAPH_LABEL[graph]} / {NOISE_LABEL[noise]}")
        if args.strict:
            raise SystemExit(2)

    print("\nSaved:")
    for path in (performance_csv, uncertainty_csv, performance_tex, uncertainty_tex):
        print(f"  {path}")


if __name__ == "__main__":
    main()
