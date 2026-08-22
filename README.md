# SPIQ

SPIQ is a framework for Clifford-based QAOA initialization with multi-start point selection for downstream continuous optimization.

## Quickstart

Requires Python 3.10+.

```bash
pip install -r requirements.txt
pip install -e .
```

Run the example notebooks in `examples/`:

1. [`examples/spiq_maxcut_workflow.ipynb`](examples/spiq_maxcut_workflow.ipynb)
2. [`examples/spiq_knapsack_workflow.ipynb`](examples/spiq_knapsack_workflow.ipynb)
3. [`examples/spiq_biomarker_workflow.ipynb`](examples/spiq_biomarker_workflow.ipynb)

Each notebook builds a small problem instance, runs SPIQ, compares two starting-point selection strategies (fixed-interval and K-GAPS), and prints selected points with energies.

Default notebook settings target a short local run on CPU.

To rerun the paper-style scripts with small test settings:

```bash
cd experiments
bash run_experiments.sh
```

Outputs are written under `experiments/data/<script_name>/`.

## Repository map

- `examples/` — public end-to-end workflow notebooks
- `spiq/` — core library (`qaoa`, `graphs`, `knapsack`, `selection`)
- `clapton/` — Clifford search engine used by SPIQ
- `biomarker_data/` — biomarker PCBO sample data (`samples/n8` … `samples/n20`) and helpers
- `red_qaoa/` — graph reduction used by `experiments/full_opt_maxcut.py`
- `experiments/` — reproducible evaluation scripts from the paper
