import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.config import fhn_a_base, fhn_tau_base, base_fhn_params, base_jr_params
from src.models.fhn import simulate_fhn
from src.models.jansenrit import simulate_jr

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

        t, V, _ = simulate_fhn(**sweep_params)

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
    '''
    Sweep to discover the edges of the dynamical regime that the baseline parameter set creates, varying v0. The coarse bounds appear to be 4.8, 6.6.

    The baseline value is 6.0, so we use 5.5, 6.5 as the interval by the symmetric interval convention.

    Note there is a weird amplitude change that is not smooth between 6.1 and 6.5. Oscillations until 6.55, where there is a sharp spike in amplitude and then a collapse to zero.
    '''
    records = []

    # v_vals = np.linspace(2.0, 8.0, 50)
    # v_vals = np.linspace(4.7, 7, 100)
    v_vals = np.linspace(6.0, 6.6, 100)


    for v in v_vals:

        sweep_params = base_jr_params.copy()
        sweep_params["v0"] = v

        t, proxy = simulate_jr(params=sweep_params)

        # calculate output statistics
        record = stats(t, proxy)

        # add identifying information
        record["model"] = "JR"
        record["parameter"] = "v0"
        record["parameter_value"] = v
        record["scale"] = v / base_jr_params["v0"]

        records.append(record)

    results_df = pd.DataFrame(records)
    print(results_df)

    # results_df.to_csv("fhn_a_sweep.csv", index=False)

    plt.plot(results_df["parameter_value"], results_df["peak to peak"], marker="o")
    plt.xlabel("v_0")
    plt.ylabel("peak to peak")
    plt.title("JR homogeneous sweep: v_0")
    plt.show()

def jr_q_sweep(base_jr_params):
    '''
    Sweeps a dimensionless timescale multiplier q to determine a working interval for the baseline dynamical regime.

    Define tau_e = 1/a and tau_i = 1/b s.t.
    tau_e' = qtau_e & tau_i' = qtau_i

    There is a jump at 0.95, and then smooth rise until basically forever. Working interval is 0.95 to 1.05.
    '''
    records = []

    q_vals = np.linspace(0.8, 100, 101)


    for q in q_vals:

        sweep_params = base_jr_params.copy()
        sweep_params["a"] = base_jr_params["a"] / q
        sweep_params["b"] = base_jr_params["b"] / q

        t, proxy = simulate_jr(params=sweep_params)

        # calculate output statistics
        record = stats(t, proxy)

        # add identifying information
        record["model"] = "JR"
        record["parameter"] = "timescale multiplier"
        record["parameter_value"] = q

        records.append(record)

    results_df = pd.DataFrame(records)
    print(results_df)

    plt.plot(results_df["parameter_value"], results_df["peak to peak"], marker="o")
    plt.xlabel("q")
    plt.ylabel("peak to peak")
    plt.title("JR homogeneous sweep: q (timescale multiplier)")
    plt.show()



# fhn_t_sweep(base_fhn_params)
# jr_v_sweep(base_jr_params=base_jr_params)
jr_q_sweep(base_jr_params=base_jr_params)