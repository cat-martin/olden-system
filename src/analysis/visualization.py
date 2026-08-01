import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from src.util.config import *
from src.models.jansenrit import simulate_jr
from src.simulations.hetero import set_v_vals
from src.analysis.math import calculate_relative_fragility_scores, cut_transient

# DATA STRUCTURE REFERENCE
#
# plot_data: list of trace dictionaries returned by one hetero_sweep()
# [
#     {
#         "model": "FHN" or "JR",
#         "parameter": "a", "tau", "v0", or "q",
#         "h": heterogeneity level,
#         "t": complete time array,
#         "homo_trace": homogeneous baseline trace,
#         "hetero_trace": heterogeneous population-mean trace,
#     },
#     ...one dictionary per h level
# ]
#
# four_panel_data: groups plot_data by model-parameter test
# {
#     "FHN a":   plot_data,
#     "FHN tau": plot_data,
#     "JR v0":   plot_data,
#     "JR q":    plot_data,
# }
#
# dataframes: groups feature-result DataFrames by model-parameter test
# {
#     "FHN a":   DataFrame,
#     "FHN tau": DataFrame,
#     "JR v0":   DataFrame,
#     "JR q":    DataFrame,
# }
#
# Each DataFrame (results_df) contains one row per h level and these columns:
# model, parameter, h,
# homo_mean, homo_std, homo_dom_freq, homo_peak_to_peak,
# hetero_mean, hetero_std, hetero_dom_freq, hetero_peak_to_peak,
# delta_mean, delta_std, delta_dom_freq, delta_peak_to_peak


def plot_h_vs_std(results_df):
    # transient is removed upstream in the stats function, don't need to do it here
    plt.figure()
    plt.title("h vs std change")
    plt.plot(results_df["h"], results_df["delta_std"], marker="o", label="std delta")
    plt.plot(
        results_df["h"],
        results_df["delta_peak_to_peak"],
        marker="o",
        label="delta peak to peak",
    )
    plt.xlabel("h level")
    plt.ylabel("delta")
    plt.legend()
    plt.show()


def plot_homo_vs_hetero(
    model, param_to_vary, h, t, V_traces, pop_mean_V, V, unit_traces=False
):
    plt.figure()

    # make the title dynamic & paper worthy
    model_labels = {
    "JR": "Jansen-Rit",
    "FHN": "FitzHugh-Nagumo",
    }

    parameter_labels = {
        "a": r"$a$",
        "tau": r"$\tau_w$",
        "v0": r"$v_0$",
        "q": r"$q$",
    }

    model_label = model_labels.get(model, model)
    parameter_label = parameter_labels.get(param_to_vary, param_to_vary)

    plt.title(
        rf"{model_label}: {parameter_label} heterogeneity ($h={h:g}$)"
    )

    plt.xlabel("Time (s)")
    plt.ylabel(r"EEG proxy, $y_1-y_2$ (mV)")

    mask = cut_transient(t, 0.2)

    if unit_traces:
        for i in range(5):
            plt.plot(
                t[mask],
                V_traces[i][mask],
                color="#2ca02c",
                alpha=0.45,
                linewidth=1.0,
                label=(
                    "Representative units"
                    if i == 0
                    else "_nolegend_"
                ),
            )

    plt.plot(
        t[mask],
        pop_mean_V[mask],
        color="#8c564b",
        linewidth=2.0,
        label="Heterogeneous mean",
    )

    plt.plot(
        t[mask],
        V[mask],
        color="#e377c2",
        linewidth=2.0,
        label="Homogeneous mean",
    )

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        fontsize=8,
        frameon=False,
    )

    plt.gcf().subplots_adjust(bottom=0.24)
    plt.savefig(
        "src/data/visuals/frozen_dephase.pdf",
        bbox_inches="tight",
    )
    plt.show()


def plot_homo_vs_hetero_ax(
    ax,
    model,
    param_to_vary,
    h,
    t,
    pop_mean_V,
    V,
):

    mask = cut_transient(t, 0.2)

    ax.plot(t[mask], pop_mean_V[mask], label="Heterogeneous mean")
    ax.plot(t[mask], V[mask], label="Homogeneous mean")
    ax.set_title(f"{model}: {param_to_vary} heterogeneity")
    ax.set_xlabel("Time")
    ax.set_ylabel("Proxy signal")


def plot_four_panel_hetero(plot_data):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8, 6),
    )

    axes = axes.flatten()

    for ax, result in zip(axes, plot_data):
        plot_homo_vs_hetero_ax(
            ax=ax,
            model=result["model"],
            param_to_vary=result["parameter"],
            h=result["h"],
            t=result["t"],
            pop_mean_V=result["hetero_trace"],
            V=result["homo_trace"],
        )

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(handles, labels, loc="upper center", ncol=2)

    fig.suptitle(
        "Effect of parameter heterogeneity on aggregate dynamics",
        y=1.04,
    )

    plt.show()


def find_run_at_h(plot_data, target_h):
    """
    Return the plot-data record corresponding to target_h
    """

    for run in plot_data:
        # use isclose just in case of floating point error
        if np.isclose(run["h"], target_h):
            return run

    available_h = [run["h"] for run in plot_data]
    raise ValueError(
        f"No run found for h={target_h}. "
        f"Available heterogeneity levels: {available_h}"
    )


def plot_cross_model_comparison(four_panel_data, target_h=1.0):
    """
    Plot homogeneous and heterogeneous population means for
    the four model-parameter tests at one heterogeneity level.
    """

    # each tuple stores dictionary key, title for subplot
    # r makes it a raw string
    # $$ for mathtext
    panel_specs = [
        ("FHN a", r"(a) FHN threshold: $a$"),
        ("JR v0", r"(b) JR threshold: $v_0$"),
        ("FHN tau", r"(c) FHN timescale: $\tau_w$"),
        ("JR q", r"(d) JR timescale: $q$"),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8, 6),
    )

    axes = axes.flatten()

    for ax, (test_key, title) in zip(axes, panel_specs):

        run = find_run_at_h(four_panel_data[test_key], target_h)

        mask = cut_transient(run["t"], 0.2)

        ax.plot(
            run["t"][mask],
            run["homo_trace"][mask],
            label="Homogeneous mean",
            linestyle="--",
            linewidth=1.5,
            color="#303030",
        )

        ax.plot(
            run["t"][mask],
            run["hetero_trace"][mask],
            label="Heterogeneous mean",
            linewidth=1.5,
            linestyle="-",
            color="#009E8E",
        )

        ax.set_title(title)

        # dynamically set axis titles
        if run["model"] == "FHN":
            ax.set_xlabel("Time (model units)")
            ax.set_ylabel(r"Mean membrane variable, $V$")

        elif run["model"] == "JR":
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(r"Mean EEG proxy, $y_1-y_2$ (mV)")

        else:
            ax.set_xlabel("Time")
            ax.set_ylabel("Population mean")

    handles, labels = axes[0].get_legend_handles_labels()

    fig.text(
        0.5,
        0.075,
        rf"Heterogeneity level: $h={target_h}$",
        ha="center",
        va="center",
        fontsize=10,
    )

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.005),
        frameon=False,
        handlelength=3,
    )

    fig.subplots_adjust(
        top=0.95, bottom=0.18, left=0.10, right=0.98, hspace=0.42, wspace=0.30
    )

    plt.savefig(
        "src/data/visuals/frozen_4_panel_trace.pdf",
        bbox_inches="tight",
    )
    plt.show()


def plot_degradation_panels(dataframes):
    """

    Plot relative degradation in standard deviation and peak-to-peak

    amplitude across heterogeneity levels for all four tests.

    """

    panel_specs = [
        ("FHN a", r"(a) FHN threshold: $a$"),
        ("JR v0", r"(b) JR threshold: $v_0$"),
        ("FHN tau", r"(c) FHN timescale: $\tau_w$"),
        ("JR q", r"(d) JR timescale: $q$"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharey=True)

    axes = axes.flatten()

    for ax in axes:
        ax.set_ylim(0, 1.0)

    for ax, (test_key, title) in zip(axes, panel_specs):

        if test_key not in dataframes:

            raise KeyError(
                f"Missing '{test_key}' from dataframes. "
                f"Available keys: {list(dataframes.keys())}"
            )

        # Copy so sorting and adding columns do not modify

        # the original DataFrame stored in the dictionary.

        df = dataframes[test_key].copy()

        df = df.sort_values("h")

        # Relative deviation from the homogeneous baseline.

        std_deviation = (df["hetero_std"] - df["homo_std"]).abs() / df["homo_std"].abs()

        ptp_deviation = (
            df["hetero_peak_to_peak"] - df["homo_peak_to_peak"]
        ).abs() / df["homo_peak_to_peak"].abs()

        ax.plot(
            df["h"],
            std_deviation,
            label="Standard deviation",
            marker="o",
            linewidth=1.7,
            color="#0C9937",
        )

        ax.plot(
            df["h"],
            ptp_deviation,
            label="Peak-to-peak",
            marker="s",
            linewidth=1.7,
            color="#764D89",
        )

        ax.set_title(title)

        ax.set_xlabel(r"Heterogeneity level, $h$")

        ax.yaxis.set_major_formatter(PercentFormatter(1.0))

        ax.set_xlim(0, 1)

        ax.set_ylim(bottom=0)

    fig.supylabel("Relative deviation from baseline")

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.015),
        frameon=False,
        handlelength=3,
    )

    fig.subplots_adjust(
        top=0.95,
        bottom=0.15,
        left=0.10,
        right=0.98,
        hspace=0.42,
        wspace=0.30,
    )

    plt.savefig(
        "src/data/visuals/frozen_4_panel_stats.pdf",
        bbox_inches="tight",
    )
    plt.show()


def plot_fragility_scores(dataframes):
    raw_scores, scores = calculate_relative_fragility_scores(dataframes)
    

        # Group the tests by matched functional role.
    group_labels = [
        "Threshold / excitability",
        "Intrinsic timescale",
    ]

    fhn_values = [
        scores["FHN a"],
        scores["FHN tau"],
    ]

    jr_values = [
        scores["JR v0"],
        scores["JR q"],
    ]

    x = np.arange(len(group_labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7, 4.5))

    fhn_bars = ax.bar(
        x - width / 2,
        fhn_values,
        width,
        label="FHN",
        color="#2A9D8F",
        edgecolor="#303030",
        linewidth=0.7,
    )

    jr_bars = ax.bar(
        x + width / 2,
        jr_values,
        width,
        label="JR",
        color="#B565A7",
        edgecolor="#303030",
        linewidth=0.7,
    )

    ax.set_title(
        "Relative Fragility by Functional Probe",
        pad=12,
    )

    ax.set_xlabel("Functional probe")
    ax.set_ylabel("Relative fragility score")
    ax.set_ylim(0, 10.8)

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)

    ax.legend(
        frameon=False,
        loc="upper left",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.bar_label(
        fhn_bars,
        labels=[f"{value:.1f}" for value in fhn_values],
        padding=4,
        fontsize=10,
    )

    ax.bar_label(
        jr_bars,
        labels=[f"{value:.1f}" for value in jr_values],
        padding=4,
        fontsize=10,
    )

    fig.tight_layout()

    fig.savefig(
        "src/data/visuals/frozen_fragility_scores.pdf",
        bbox_inches="tight",
    )

    plt.show()

    return raw_scores, scores


# one visual showing multiple individual traces and phase degredation

# one four panel visual showing degredation over all four tests

# one four panel feature degredation curve, with std and peak to peak

# one fragility score comparison
