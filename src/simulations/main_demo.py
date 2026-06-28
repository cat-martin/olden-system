from src.analysis.sweeps import hetero_sweep
from src.util.config import *
from src.simulations.hetero import set_a_vals, set_q_vals, set_tau_vals, set_v_vals
from src.models.fhn import simulate_fhn
from src.models.jansenrit import simulate_jr
from src.analysis.visualization import plot_cross_model_comparison

def main():

    four_panel_data = {}

    # print("This is the main demonstration showcasing the comparison \nof FitzHugh-Nagumo and Jansen Rit models on the heterogeneity axis \nof the five dimensional framework.\n")

    # print("\nSweep illustrating differences in JR population mean \nwith and without heterogeneous 'v0' values. Demonstrates \nhow dephasing causes oscillation collapse in heterogeneous population mean:\n")

    # hetero_sweep(
    #     baseline_params=base_jr_params,
    #     h_vals=[0.75],
    #     sim_fn=simulate_jr,
    #     set_fn=set_v_vals,
    #     half_widths=half_widths,
    #     param_to_vary='v0'
    # )

    # print("\nSweep illustrating differences in FHN population mean \nwith and without heterogeneous 'tau' values, representing \nresults in populations with heterogeneous recovery \ntimescales as compared to the homogeneous baseline:\n")

    _, plot_data = hetero_sweep(
        baseline_params=base_fhn_params,
        h_vals=simple_h_vals,
        sim_fn=simulate_fhn,
        set_fn=set_tau_vals,
        half_widths=half_widths,
        param_to_vary='tau',
        unit_traces=False,
        four_panel=False
    )
    four_panel_data["FHN tau"] = plot_data

    # print("\nSweep illustrating differences in JR population mean \nwith and without heterogeneous 'v0' values, representing \nresults in populations with heterogeneous firing \nthresholds as compared to the homogeneous baseline:\n")

    _, plot_data = hetero_sweep(
        baseline_params=base_jr_params,
        h_vals=simple_h_vals,
        sim_fn=simulate_jr,
        set_fn=set_v_vals,
        half_widths=half_widths,
        param_to_vary='v0',
        unit_traces=False,
        four_panel=False
    )
    four_panel_data["JR v0"] = plot_data

    # print("\nSweep illustrating differences in JR population mean \nwith and without heterogeneous 'q' values, representing \nresults in populations with heterogeneous \ntime constants as compared to the homogeneous baseline:\n")

    _, plot_data = hetero_sweep(
        baseline_params=base_jr_params,
        h_vals=simple_h_vals,
        sim_fn=simulate_jr,
        set_fn=set_q_vals,
        half_widths=half_widths,
        param_to_vary='q',
        unit_traces=False,
        four_panel=False
    )
    four_panel_data["JR q"] = plot_data

    _, plot_data = hetero_sweep(
        baseline_params=base_fhn_params,
        h_vals=simple_h_vals,
        sim_fn=simulate_fhn,
        set_fn=set_a_vals,
        half_widths=half_widths,
        param_to_vary='a',
        unit_traces=False,
        four_panel=False
    )
    four_panel_data["FHN a"] = plot_data

    plot_cross_model_comparison(
        four_panel_data,
        target_h=1.0
    )

    # print("\nDemonstration complete.")



if __name__ == "__main__":
    main()