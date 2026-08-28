from pathlib import Path
from fire import Fire
import os
import re
import sys
from dataclasses import dataclass
from typing import List
import json
import datetime
from statistics import mean

from plotnine import (
    ggplot,
    aes,
    geom_line,
    geom_point,
    facet_grid,
    facet_wrap,
    scale_x_continuous,
    scale_y_continuous,
    geom_hline,
    position_dodge,
    geom_errorbar,
    scale_y_discrete,
    theme,
    element_text,
    ylab,
    xlab,
    scale_color_discrete,
)
import pandas as pd
from pandas import Categorical, DataFrame
from plotnine.scales.limits import ylim
from plotnine.scales.scale_xy import scale_x_discrete


def plot_single_frame(frame, output_file_name):

    plot = (
        ggplot(frame)
        + aes(x="Edges Added", y="Acc-0.1")
        + geom_point(
            size=1.5,
        )
        # + scale_x_continuous(limits=(0.2, 0.8))
        # + scale_y_discrete(limits=model_type_order)
    )

    # plot = plot + theme(figure_size=(6, 4), strip_text_x=element_text(size=5), axis_text_x=element_text(size=5),)

    plot.save(str(output_file_name), dpi=600)



def plot_combined_frame(frame, output_file_name):

    plot = (
        ggplot(frame)
        + aes(x="Edges Added", y="Acc-0.1", color="Noise")
        + geom_point(
            size=1.5,
        )
        + geom_line(
            size=0.5,
        )
        # + scale_x_continuous(limits=(0.2, 0.8))
        # + scale_y_discrete(limits=model_type_order)
    )

    # plot = plot + theme(figure_size=(6, 4), strip_text_x=element_text(size=5), axis_text_x=element_text(size=5),)

    plot.save(str(output_file_name), dpi=600)



def main(
    root="./results/progressive/uncertainty",
    prefix="rggcn_pr_tree",
):

    glob_pattern_no_noise = f"{prefix}_nop*summary.json"
    glob_pattern_noisy = f"{prefix}_p*summary.json"

    files_nop = list(Path(root).rglob(glob_pattern_no_noise))
    files_p = list(Path(root).rglob(glob_pattern_noisy))
    print(f"Found {len(files_nop)} files for no noise and {len(files_p)} files for noisy.")

    df_noisy = None
    df_no_noise = None

    for name, files in [("no noise", files_nop), ("noisy", files_p)]:
        if len(files) == 0:
            print(f"No files found for {name} with pattern {glob_pattern_no_noise if name == 'no noise' else glob_pattern_noisy}.")
            continue

        all_results = []
        for file in files:
            with open(file, "r") as f:
                data = json.load(f)
                experiment_name = data.get("experiment", "None")
                acc_01 = data["runs"][0].get("test_unseen", {}).get("acc@0.1", None)
                acc_02 = data["runs"][0].get("test_unseen", {}).get("acc@0.2", None)
                l1 = data["runs"][0].get("test_unseen", {}).get("l1", None)
                loss = data["runs"][0].get("test_unseen", {}).get("loss", None)
                all_results.append({
                    "Experiment": experiment_name,
                    "Edges Added": int(experiment_name.split("_")[-1]),
                    "Acc-0.1": acc_01,
                    "Acc-0.2": acc_02,
                    "L1": l1,
                    "Loss": loss,
                })

        df = DataFrame(all_results)
        output_file_name = Path(root) / f"summary_{name.replace(' ', '_')}.csv"
        df.sort_values(by=["Edges Added"], ascending=True, inplace=True)

        print(df.to_string())

        df.to_csv(output_file_name, index=False)
        print(f"Saved summary for {name} to {output_file_name}")

        if name == "no noise":
            df_no_noise = df
        else:
            df_noisy = df

        plot_single_frame(df, Path(root) / f"summary_{name.replace(' ', '_')}.png")

    df_no_noise["Noise"] = "No Noise"
    df_noisy["Noise"] = "Noisy"

    if len(df_no_noise) > len(df_noisy):
        df_no_noise = df_no_noise[df_no_noise["Edges Added"].isin(df_noisy["Edges Added"])]
    elif len(df_noisy) > len(df_no_noise):
        df_noisy = df_noisy[df_noisy["Edges Added"].isin(df_no_noise["Edges Added"])]

    df_combined = DataFrame(pd.concat([df_no_noise, df_noisy], ignore_index=True))

    for edges_addeed in df_combined["Edges Added"].unique():
        df_subset = df_combined[df_combined["Edges Added"] == edges_addeed]
        acc_01_diff = df_subset[df_subset["Noise"] == "No Noise"]["Acc-0.1"].values[0] - df_subset[df_subset["Noise"] == "Noisy"]["Acc-0.1"].values[0]
        print(f"Edges Added: {edges_addeed}, Acc-0.1 Difference: {acc_01_diff:.4f}")


    print(df_combined.to_string())

    plot_combined_frame(df_combined, Path(root) / f"summary_combined.png")


if __name__ == "__main__":
    Fire(main)