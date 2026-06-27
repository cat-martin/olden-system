import numpy as np
import matplotlib.pyplot as plt
from src.models.fhn import simulate_fhn
from src.config import base_fhn_params


'''
want to create a heterogeneous parameter population with an exact baseline mean

h in [0,1] is the heterogeneity level, where h=0.5 allows deviations that extend halfway toward the interval boundaries
'''

def hetero_vals(baseline, half_width, h, num_units=100, seed=None):
    
    if not 0 <= h <= 1:
        raise ValueError("h must be between 0 and 1")
    
    rng = np.random.default_rng(seed)

    # create positive and negative pairs of deviations so the mean stays zero
    num_pairs = num_units // 2
    pos_dev = rng.uniform(0.0, 1.0, size=num_pairs)

    norm_dev = np.concatenate([pos_dev, -pos_dev,])

    # if an odd number of units, append one that's exactly on the baseline value so we don't skew the mean
    if num_units % 2 == 1:
        norm_dev = np.append(norm_dev, 0.0)

    rng.shuffle(norm_dev)

    params = (baseline + h * half_width * norm_dev)

    return params

def sim_fhn_hetero_pop_a(baseline_params, h, half_width=0.17, num_units=100, seed=37):
    '''
    Simulates a population of FHN neurons of the specified heterogeneity level.
    '''
    a_vals = hetero_vals(baseline=baseline_params["a"], half_width=half_width, h=h, num_units=num_units, seed=seed)

    V_traces = []

    # simulate and store traces for each unit
    for a_val in a_vals:
        unit_params = baseline_params.copy()
        unit_params["a"] = a_val

        t, V, _ = simulate_fhn(**unit_params)
    
        V_traces.append(V)

    # turn list of arrays into 2D array
    V_traces = np.array(V_traces)

    # average along the 0 axis, aka take average of V across all units at each time point
    pop_mean_V = np.mean(V_traces, axis=0)

    return t, pop_mean_V, V_traces, a_vals


