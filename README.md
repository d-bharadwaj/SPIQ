# SPIQ

SPIQ is a framework for Clifford-based QAOA initialization (CAFQA-style search) with multi-start point selection for downstream continuous optimization.

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
- runs SPIQ initialization
- compares two starting-point selection strategies:
  - spaced-out energy selection
  - clustering with gradient filtering
- prints selected points and expectation values

Default settings target a short local run (under about one minute).

For the biomarker example, the notebook loads
`biomarker_data/sampled_8_features_subproblem_1 copy`
(the non-`copy` directories currently contain Git LFS pointer files).

## Repository map

- `examples/` — public end-to-end workflows
- `spiq/` — public Python API used by the example notebooks
- `clapton/` — SPIQ / Clifford search core
- `biomarker_data/` — biomarker PCBO datasets and helpers
- `testing_scripts/` — development and experiment scripts (not required for the public quickstart)

## Required packages

Install dependencies and the local package:

```bash
pip install -r requirements.txt
pip install -e .
```
