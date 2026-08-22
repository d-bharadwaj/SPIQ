# SPIQ

SPIQ is a framework for Clifford-based QAOA initialization with multi-start point selection for downstream continuous optimization.

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .
```

Run the public example notebooks in `examples/`:

1. [`examples/spiq_maxcut_workflow.ipynb`](examples/spiq_maxcut_workflow.ipynb)
2. [`examples/spiq_knapsack_workflow.ipynb`](examples/spiq_knapsack_workflow.ipynb)
3. [`examples/spiq_biomarker_workflow.ipynb`](examples/spiq_biomarker_workflow.ipynb)

Each notebook:

- builds a small problem instance
- runs the SPIQ framework
- compares two starting-point selection strategies:
  - Fixed Interval Selection
  - K-GAPS
- outputs selected points and expectation values

Default settings target a short local run.

## Repository map

- `examples/` — public end-to-end workflows
- `spiq/` — core library
- `clapton/` — Clifford search engine used by SPIQ
- `biomarker_data/` — biomarker PCBO datasets
- `experiments/` — scripts for evaluations done in the paper
