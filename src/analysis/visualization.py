import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from src.util.config import *
from src.models.jansenrit import simulate_jr
from src.simulations.hetero import set_v_vals
from src.analysis.math import calculate_relative_fragility_scores


def plot_h_vs_std(results_df):
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
    plt.title(f"{model} Heterogeneity Sweep: Deviation on '{param_to_vary}' | h={h}")
    plt.xlabel("Time")
    plt.ylabel("Proxy Signal")

    if unit_traces:
        for i in range(5):
            plt.plot(t, V_traces[i], alpha=0.6)

    plt.plot(t, pop_mean_V, label="Heterogeneous Population Mean")
    plt.plot(t, V, label="Homogeneous Population Mean")
    plt.legend()
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

    ax.plot(t, pop_mean_V, label="Heterogeneous mean")
    ax.plot(t, V, label="Homogeneous mean")
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
    """Return the plot-data record corresponding to target_h."""

    for run in plot_data:
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
    # $$ for mathjax
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

        ax.plot(
            run["t"],
            run["homo_trace"],
            label="Homogeneous mean",
            linestyle="--",
            linewidth=1.5,
            color="#303030",
        )

        ax.plot(
            run["t"],
            run["hetero_trace"],
            label="Heterogeneous mean",
            linewidth=1.5,
            linestyle="-",
            color="#009E8E",
        )

        ax.set_title(title)
        ax.set_xlabel("Time")

        if run["model"] == "FHN":
            ax.set_ylabel(r"Mean membrane variable, $V$")

        elif run["model"] == "JR":
            ax.set_ylabel(r"Mean EEG proxy, $y_1-y_2$")

        else:
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
        ax.set_ylim(0, 0.85)

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

    plt.show()


def plot_fragility_scores(dataframes):
    _, scores = calculate_relative_fragility_scores(dataframes)

    test_order = [
        "FHN a",
        "JR v0",
        "FHN tau",
        "JR q",
    ]

    labels = [
        r"FHN $a$",
        r"JR $v_0$",
        r"FHN $\tau_w$",
        r"JR $q$",
    ]

    values = [scores[test] for test in test_order]

    colors = [
        "#2A9D8F",  # teal
        "#B565A7",  # magenta
        "#5E81AC",  # slate blue
        "#E76F51",  # coral
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    bars = ax.bar(
        labels,
        values,
        color=colors,
        width=0.65,
        edgecolor="#303030",
        linewidth=0.7,
    )

    ax.set_title(
        "Relative Fragility Across Model–Parameter Tests",
        pad=12,
    )

    ax.set_xlabel("Model–parameter test")
    ax.set_ylabel("Relative fragility score")
    ax.set_ylim(0, 10.8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.bar_label(
        bars,
        labels=[f"{value:.1f}" for value in values],
        padding=4,
        fontsize=10,
    )

    fig.tight_layout()
    plt.show()


# one visual showing multiple individual traces and phase degredation

# one four panel visual showing degredation over all four tests

# one four panel feature degredation curve, with std and peak to peak

# one fragility score comparison, perhaps a mock radar chart
