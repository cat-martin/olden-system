import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.config import fhn_a_base, fhn_tau_base, base_fhn_params, base_jr_params
from src.models.fhn import simulate_fhn

def stats(t, y_axis, transient=0.5):
    # a function that takes the output of a sim and produces a record of the sim's stats

    mask = t >= transient
    # cut out the transient
    t_cut = t[mask]
    y_cut = y_axis[mask]

    y_mean = np.mean(y_cut)
    y_std = np.std(y_cut)

    # dominant frequency
    dt = t_cut[1] - t_cut[0]
    freqs = np.fft.rfftfreq(len(y_cut), d=dt)   # create possible frequencies
    spectrum = np.abs(np.fft.rfft(y_cut - y_mean))  # remove vertical offset, return strength of frequency contributions
    dominant_idx = np.argmax(spectrum[1:]) + 1   # ignores frequency 0, returns largest of remaining values, +1 since we skip frq 0
    dominant_freq = freqs[dominant_idx]

    amplitude = np.ptp(y_cut)

    return {
        "mean": y_mean,
        "std": y_std,
        "dom_freq": dominant_freq,
        "peak to peak": amplitude
    }

def fhn_a_sweep():
    '''
    Sweeps values of parameter 'a' to determine the edges of the baseline dynamical regime for the FHN model while varying excitability.

    Using a symmetric interval convention, the course bounds for the dynamic regime appear to be -0.27, 0.07. This uses the shortest distance from the baseline to an edge, and applies that distance on the other side of the baseline to create an interval. 
    '''
    records = []

    a_values = np.linspace(0.4, 0.8, 50)
    # a_values = np.linspace(-0.31, -0.25, 31)
    for a_value in a_values:
        # run simulation using this parameter value
        t, V, w = simulate_fhn(a=a_value)

        # calculate output statistics
        record = stats(t, V)

        # add identifying information
        record["model"] = "FHN"
        record["parameter"] = "a"
        record["parameter_value"] = a_value
        record["scale"] = a_value / fhn_a_base

        records.append(record)

    results_df = pd.DataFrame(records)
    print(results_df)

    # results_df.to_csv("fhn_a_sweep.csv", index=False)

    plt.plot(results_df["parameter_value"], results_df["peak to peak"], marker="o")
    plt.xlabel("a")
    plt.ylabel("peak to peak")
    plt.title("FHN homogeneous sweep: a")
    plt.show()

def fhn_t_sweep(base_fhn_params):
    '''
    Sweeps values of 1/c while keeping b/c fixed, to determine the edges of the baseline dynamical regime for the FHN model while varying the parameter values controlling the timescale.

    for baseline params, lower bound appears to be tau - 43.0
    upper bound for testing purposes will be tau = 57, so as to center an interval around the baseline value of 50.
    '''
    records = []

    tau_vals = np.linspace(2, 200, 100)

    kappa = base_fhn_params["b"] / base_fhn_params["c"]

    for tau in tau_vals:
        # run simulation using this parameter value
        c_val = 1.0 / tau
        b_val = kappa * c_val

        sweep_params = base_fhn_params.copy()
        sweep_params["b"] = b_val
        sweep_params["c"] = c_val

        t, V, w = simulate_fhn(**sweep_params)

        # calculate output statistics
        record = stats(t, V)

        # add identifying information
        record["model"] = "FHN"
        record["parameter"] = "a"
        record["parameter_value"] = tau
        record["scale"] = tau / fhn_tau_base

        records.append(record)

    results_df = pd.DataFrame(records)
    print(results_df)

    # results_df.to_csv("fhn_a_sweep.csv", index=False)

    plt.plot(results_df["parameter_value"], results_df["peak to peak"], marker="o")
    plt.xlabel("tau")
    plt.ylabel("peak to peak")
    plt.title("FHN homogeneous sweep: tau (1/c)")
    plt.show()


def jr_v_sweep(base_jr_params):
    records = []

    tau_vals = np.linspace(2, 200, 100)

    kappa = base_fhn_params["b"] / base_fhn_params["c"]

    for tau in tau_vals:
        # run simulation using this parameter value
        c_val = 1.0 / tau
        b_val = kappa * c_val

        sweep_params = base_fhn_params.copy()
        sweep_params["b"] = b_val
        sweep_params["c"] = c_val

        t, V, w = simulate_fhn(**sweep_params)

        # calculate output statistics
        record = stats(t, V)

        # add identifying information
        record["model"] = "FHN"
        record["parameter"] = "a"
        record["parameter_value"] = tau
        record["scale"] = tau / fhn_tau_base

        records.append(record)

    results_df = pd.DataFrame(records)
    print(results_df)

    # results_df.to_csv("fhn_a_sweep.csv", index=False)

    plt.plot(results_df["parameter_value"], results_df["peak to peak"], marker="o")
    plt.xlabel("tau")
    plt.ylabel("peak to peak")
    plt.title("FHN homogeneous sweep: tau (1/c)")
    plt.show()


fhn_t_sweep(base_fhn_params)