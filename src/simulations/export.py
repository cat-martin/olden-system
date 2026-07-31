import pandas as pd
from pathlib import Path

from src.analysis.sweeps import hetero_sweep
from src.util.config import (
    base_fhn_params,
    base_jr_params,
    half_widths,
    simple_h_vals,
)
from src.simulations.hetero import set_a_vals, set_q_vals, set_tau_vals, set_v_vals
from src.models.fhn import simulate_fhn
from src.models.jansenrit import simulate_jr
from src.analysis.math import calculate_relative_fragility_scores


def main():

    output_path = Path("src/data/")
    output_path.mkdir(parents=True, exist_ok=True)

    # casting to a string removes the trailing slash
    output_path = str(output_path) + '/'

    four_panel_data = {}
    dataframes = {}

    dataframe, plot_data = hetero_sweep(
        baseline_params=base_fhn_params,
        h_vals=simple_h_vals,
        sim_fn=simulate_fhn,
        set_fn=set_tau_vals,
        half_widths=half_widths,
        param_to_vary="tau",
        unit_traces=False,
        four_panel=False,
    )
    four_panel_data["FHN tau"] = plot_data
    dataframes["FHN tau"] = dataframe

    dataframe, plot_data = hetero_sweep(
        baseline_params=base_jr_params,
        h_vals=simple_h_vals,
        sim_fn=simulate_jr,
        set_fn=set_v_vals,
        half_widths=half_widths,
        param_to_vary="v0",
        unit_traces=False,
        four_panel=False,
    )
    four_panel_data["JR v0"] = plot_data
    dataframes["JR v0"] = dataframe

    dataframe, plot_data = hetero_sweep(
        baseline_params=base_jr_params,
        h_vals=simple_h_vals,
        sim_fn=simulate_jr,
        set_fn=set_q_vals,
        half_widths=half_widths,
        param_to_vary="q",
        unit_traces=False,
        four_panel=False,
    )
    four_panel_data["JR q"] = plot_data
    dataframes["JR q"] = dataframe

    dataframe, plot_data = hetero_sweep(
        baseline_params=base_fhn_params,
        h_vals=simple_h_vals,
        sim_fn=simulate_fhn,
        set_fn=set_a_vals,
        half_widths=half_widths,
        param_to_vary="a",
        unit_traces=False,
        four_panel=False,
    )
    four_panel_data["FHN a"] = plot_data
    dataframes["FHN a"] = dataframe

    for name, sims_list in four_panel_data.items():
        frames = []
        for sim in sims_list:

            trace_df = pd.DataFrame(
                {
                    "model": sim["model"],
                    "parameter": sim["parameter"],
                    "h": sim["h"],
                    "time": sim["t"],
                    "homo_trace": sim["homo_trace"],
                    "hetero_trace": sim["hetero_trace"],
                }
            )

            frames.append(trace_df)

        all_frames = pd.concat(frames, ignore_index=True)

        filename = f"{output_path}{sims_list[0]['model']}_" f"{sims_list[0]['parameter']}_trace.csv"

        all_frames.to_csv(
            filename,
            index=False,
        )

    raw_scores, scores = calculate_relative_fragility_scores(dataframes)

    score_rows = []

    for test_key in raw_scores:
        score_rows.append(
            {
                "test_key": str(test_key),
                "raw_fragility": raw_scores[test_key],
                "relative_fragility": scores[test_key],
            }
        )

    scores_df = pd.DataFrame(score_rows)

    scores_df.to_csv(
        "src/data/fragility_scores.csv",
        index=False,
    )

    for name, df in dataframes.items():
        df.to_csv(f"{output_path + name}.csv", index=False)


if __name__ == "__main__":
    main()
