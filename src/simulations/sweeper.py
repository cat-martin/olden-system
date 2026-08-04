import numpy as np

from src.analysis.sweeps import persistent_parameter_sweep
from src.util.config import base_jr_params, base_fhn_params
from src.models.jansenrit import simulate_jr
from src.models.fhn import simulate_fhn
from src.simulations.hetero import set_a_vals, set_q_vals, set_tau_vals, set_v_vals


def main():

    # keep the sweep results for refined interval selection in the future

    # shows refined interval of [0.985, 1.015] or half width=0.015
    jr_q_results = persistent_parameter_sweep(
        baseline_params=base_jr_params,
        parameter_vals=np.linspace(0.95, 1.05, 51),
        parameter_name="q",
        set_fn=set_q_vals,
        sim_fn=lambda params: simulate_jr(params, t_end=10.48),
    )

    # shows refined interval of [5.9, 6.1] or half width=0.1
    jr_v_results = persistent_parameter_sweep(
        baseline_params=base_jr_params,
        parameter_vals=np.linspace(5.5, 6.5, 51),
        parameter_name="v0",
        set_fn=set_v_vals,
        sim_fn=lambda params: simulate_jr(params, t_end=10.48),
    )

    # supports original range, [-0.27, 0.07], half width = 0.17
    fhn_a_results = persistent_parameter_sweep(
        baseline_params=base_fhn_params,
        parameter_vals=np.linspace(-0.35, 0.20, 51),
        parameter_name="a",
        set_fn=set_a_vals,
        sim_fn=lambda params: simulate_fhn(params, end=2740),
    )

    # need to revise original range
    # oscillations appear before tau=5 and continue on to forseeable future
    # range following convention is [5, 95] with half width=45
    fhn_tau_results = persistent_parameter_sweep(
        baseline_params=base_fhn_params,
        parameter_vals=np.linspace(1, 20, 51),
        parameter_name="tau",
        set_fn=set_tau_vals,
        sim_fn=lambda params: simulate_fhn(params, end=2740),
    )


if __name__ == "__main__":
    main()
