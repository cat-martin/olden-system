A prototype framework that allows informed neuronal and neural mass model selection via quantified failure tolerance.

The current implementation is proof-of-concept and includes simulation and export for the heterogeneity axis only.

main_demo.py: a visual demonstration that will produce graphics illustrating the effect of heterogeneity on model output.

export.py: official analysis, will save CSV output into src/data.

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