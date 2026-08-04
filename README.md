A prototype framework that allows informed neuronal and neural mass model selection via quantified failure tolerance.

The current implementation is proof-of-concept and includes simulation and export for the heterogeneity axis only.

main_demo.py: a visual demonstration that will produce graphics illustrating the effect of heterogeneity on model output.

export.py: official analysis, will save CSV output into src/data.

sweeper.py: used to conduct homogeneous parameter sweeps that informed operational parameter interval selection.

robustness.py: recalculates scores under the robustness conditions and exports the results.

CSV outputs are written to `src/data`, and generated PDF figures are written to `src/data/visuals`.

Commands should be run from the repository root.

## Setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```
To see the visual demo:

```bash
python3 -m src.simulations.main_demo
```

To execute an analysis run or export data:

```bash
python3 -m src.simulations.export
```

To conduct homogeneous parameter sweeps:

```bash
python3 -m src.simulations.sweeper
```

To execute the robustness analysis:

```bash
python3 -m src.simulations.robustness
```

## Project structure

```text
.
├── README.md
├── requirements.txt
└── src
    ├── analysis
    │   ├── math.py             
    │   ├── sweeps.py           
    │   └── visualization.py    
    ├── models
    │   ├── fhn.py              
    │   └── jansenrit.py        
    ├── simulations
    │   ├── export.py           
    │   ├── hetero.py           
    │   ├── main_demo.py        
    │   ├── robustness.py       
    │   └── sweeper.py          
    ├── util
    │   └── config.py           
    └── data
        ├── *_features.csv      # Extracted feature values
        ├── *_traces.csv        # Homogeneous and heterogeneous traces
        ├── fragility_scores.csv
        ├── model_fragility_scores.csv
        ├── robustness_scores.csv
        ├── model_robustness_scores.csv
        └── visuals
            ├── frozen_dephase.pdf
            ├── frozen_4_panel_trace.pdf
            ├── frozen_4_panel_stats.pdf
            └── frozen_fragility_scores.pdf
```