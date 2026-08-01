import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.simulations.hetero import hetero_sim
from src.analysis.visualization import (
    plot_homo_vs_hetero,
    plot_four_panel_hetero,
)
from src.analysis.math import cut_transient

def stats(t, y_axis, transient_frac=0.2):
    """
    A function that takes the output of a sim and produces a record of the sim's stats.
    The transient_frac parameter calculates the percentage of the simulation data to remove from the beginning.
    """
    # assumes sim doesn't start at zero just in case
    mask = cut_transient(t, transient_frac)
    # cut out the transient
    t_cut = t[mask]
    y_cut = y_axis[mask]

    y_mean = np.mean(y_cut)
    y_std = np.std(y_cut)

    # dominant frequency
    dt = t_cut[1] - t_cut[0]
    freqs = np.fft.rfftfreq(len(y_cut), d=dt)  # create possible frequencies
    spectrum = np.abs(
        np.fft.rfft(y_cut - y_mean)
    )  # remove vertical offset, return strength of frequency contributions
    dominant_idx = (
        np.argmax(spectrum[1:]) + 1
    )  # ignores frequency 0, returns largest of remaining values, +1 since we skip frq 0
    dominant_freq = freqs[dominant_idx]

    amplitude = np.ptp(y_cut)

    return {
        "mean": y_mean,
        "std": y_std,
        "dom_freq": dominant_freq,
        "peak to peak": amplitude,
    }


def persistent_parameter_sweep(
    baseline_params,
    parameter_vals,
    parameter_name,
    set_fn,
    sim_fn,
):

    '''
    model agnostic improved sweep fn
    call it like:
    jr_v_results = persistent_parameter_sweep(
        baseline_params=base_jr_params,
        parameter_vals=np.linspace(5.5, 6.5, 51),
        parameter_name="v0",
        set_fn=set_v_vals,
        sim_fn=lambda params: simulate_jr(params, t_end=10.48),
    )
    '''
    records = []

    for value in parameter_vals:
        sweep_params = set_fn(baseline_params, value)
        t, trace = sim_fn(sweep_params)

        n = len(trace)

        prev = trace[int(0.60 * n) : int(0.80 * n)]
        final = trace[int(0.80 * n) :]

        # peak to peak range of 2nd to last and last 20% of the sim
        prev_ptp = np.ptp(prev)
        final_ptp = np.ptp(final)

        # not using this yet, but want a number to tell me how much oscillations are changing so i can refine operating intervals later
        # avoiding zero issue
        stability_ratio = (
            final_ptp / prev_ptp
            if prev_ptp > 1e-12
            else np.nan
        )

        records.append(
            {
                parameter_name: value,
                "previous_peak_to_peak": prev_ptp,
                "final_peak_to_peak": final_ptp,
                "stability_ratio": stability_ratio,
            }
        )

    results_df = pd.DataFrame(records)

    model = ""

    if parameter_name in ("a", "tau"):
        model = "FHN"
    else:
        model = "JR"

    # get rid of any stale figure state
    plt.close("all")

    fig, ax = plt.subplots()

    ax.plot(
        results_df[parameter_name],
        results_df["final_peak_to_peak"],
        marker="o",
    )

    ax.set_xlabel(parameter_name)
    ax.set_ylabel("Late-window peak-to-peak")
    ax.set_title(
        f"Late Window Oscillation Persistence Sweep for "
        f"{model} {parameter_name}"
    )

    # this next part gets the windows to behave correctly - they are chaotic without it
    # don't rely on default blocking behavior. doesn't always work
    plt.show(block=False)

    # force it to wait until the figure is closed
    while plt.fignum_exists(fig.number):
        plt.pause(0.1)

    # refresh the figure state again
    plt.close("all")

    return results_df


def hetero_sweep(
    baseline_params,
    h_vals,
    sim_fn,
    set_fn,
    half_widths,
    param_to_vary,
    unit_traces=True,
    four_panel=False,
):
    """
    Model agnostic heterogeneous sweep fn.
    """

    if param_to_vary == "a" or param_to_vary == "tau":
        model = "FHN"
    else:
        model = "JR"

    plot_data = []

    t_homo, V = sim_fn(baseline_params)
    homo_stats = stats(t_homo, V)

    records = []

    for h in h_vals:
        t, pop_mean_V, V_traces, _ = hetero_sim(
            baseline_params=baseline_params,
            h=h,
            half_widths=half_widths,
            sim_fn=sim_fn,
            set_fn=set_fn,
            param_to_vary=param_to_vary,
        )

        hetero_stats = stats(t, pop_mean_V)

        record = {
            "model": model,
            "parameter": param_to_vary,
            "h": h,
            "homo_mean": homo_stats["mean"],
            "homo_std": homo_stats["std"],
            "homo_dom_freq": homo_stats["dom_freq"],
            "homo_peak_to_peak": homo_stats["peak to peak"],
            "hetero_mean": hetero_stats["mean"],
            "hetero_std": hetero_stats["std"],
            "hetero_dom_freq": hetero_stats["dom_freq"],
            "hetero_peak_to_peak": hetero_stats["peak to peak"],
        }

        record["delta_mean"] = abs(hetero_stats["mean"] - homo_stats["mean"])

        record["delta_std"] = abs(hetero_stats["std"] - homo_stats["std"])

        record["delta_dom_freq"] = abs(
            hetero_stats["dom_freq"] - homo_stats["dom_freq"]
        )

        record["delta_peak_to_peak"] = abs(
            hetero_stats["peak to peak"] - homo_stats["peak to peak"]
        )

        records.append(record)

        if unit_traces:
            plot_homo_vs_hetero(
                model=model,
                param_to_vary=param_to_vary,
                h=h,
                t=t,
                V_traces=V_traces,
                pop_mean_V=pop_mean_V,
                V=V,
                unit_traces=unit_traces,
            )
        else:
            plot_data.append(
                {
                    "model": model,
                    "parameter": param_to_vary,
                    "h": h,
                    "t": t,
                    "homo_trace": V,
                    "hetero_trace": pop_mean_V,
                }
            )

    if not unit_traces and four_panel:
        plot_four_panel_hetero(plot_data=plot_data)

    results_df = pd.DataFrame(records)

    return results_df, plot_data
