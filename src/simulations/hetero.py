import numpy as np

def hetero_vals(baseline, half_width, h, num_units=100, seed=None):
    '''
    Creates a heterogeneous parameter population with an exact baseline mean

    h in [0,1] is the heterogeneity level, where h=0.5 allows deviations that extend halfway toward the interval boundaries
    '''
    
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

    # applies the calc to every point in the vector of samples
    params = (baseline + h * half_width * norm_dev)

    return params

# these functions just allow the creation of a new parameter set that contains the heterogeneous parameter to simulate a single unit with 
def set_a_vals(baseline_params, a_val):
    unit_params = baseline_params.copy()
    unit_params['a'] = a_val
    return unit_params

def set_tau_vals(baseline_params, tau):
    unit_params = baseline_params.copy()
    kappa = baseline_params['b'] / baseline_params['c']
    
    unit_params["c"] = 1.0/tau
    unit_params["b"] = kappa * unit_params["c"]
    return unit_params

def set_q_vals(baseline_params, q):
    unit_params = baseline_params.copy()
    unit_params["a"] = baseline_params["a"] / q
    unit_params["b"] = baseline_params["b"] / q
    return unit_params

def set_v_vals(baseline_params, v):
    unit_params = baseline_params.copy()
    unit_params["v0"] = v
    return unit_params

# internals of hetero_sweep
def hetero_sim(
        baseline_params, 
        h, 
        half_widths, 
        param_to_vary,
        sim_fn, 
        set_fn, 
        num_units=100, 
        seed=37):
    '''Generates hetero ensembles, informs the user of what's going on, simulates the hetero units, collects the traces, and returns them along with the ensemble pointwise mean.'''
    
    # handle special cases where hetero params are derived
    if param_to_vary == "tau":
        baseline = 1/baseline_params["c"]
    elif param_to_vary == "q":
        baseline = 1
    else:
        baseline = baseline_params[param_to_vary]

    # generate hetero parameters
    vals = hetero_vals(
        baseline=baseline, 
        half_width=half_widths[param_to_vary], 
        h=h, 
        num_units=num_units, 
        seed=seed)

    V_traces = []

    print(
    f"Simulating {num_units} units for "
    f"{param_to_vary} heterogeneity at h={h:.2f}...",
    flush=True,
    )

    # create new parameter sets & simulate individual units
    for val in vals:
        unit_params = set_fn(baseline_params, val)
        t, V = sim_fn(unit_params)

        V_traces.append(V)

    V_traces = np.array(V_traces)
    pop_mean_V = np.mean(V_traces, axis=0)

    return t, pop_mean_V, V_traces, vals




