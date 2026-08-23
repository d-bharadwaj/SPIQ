# SPIQ

SPIQ is a framework for Clifford-based QAOA initialization for downstream continuous optimization.

## Quickstart

Requires Python 3.10+.

```bash
pip install -r requirements.txt
pip install -e .
```

Run the example notebooks in `examples/`:

1. `examples/spiq_maxcut_workflow.ipynb`
2. `examples/spiq_knapsack_workflow.ipynb`
3. `examples/spiq_biomarker_workflow.ipynb`

Each notebook builds a small problem instance, runs SPIQ, compares two starting-point selection strategies (fixed-interval and K-GAPS), and outputs selected points with energies.

Default notebook settings target a short local run on CPU.

To run evaluations that are in the paper:

```bash
cd experiments
bash run_experiments.sh
```

Outputs are written under `experiments/data/<script_name>/`.

### `experiments/` scripts

These are command-line versions of the notebook workflows, plus the heavier evaluations from the paper:

- `spiq_maxcut_workflow.py`, `spiq_knapsack_workflow.py`, `spiq_biomarker_workflow.py` — build a problem instance, run SPIQ, apply fixed-interval and K-GAPS selection, and save results to `data/`.
- `statevec_reduction_maxcut.py`, `statevec_reduction_knapsack.py` — compare SPIQ vs random initialization by measuring factor of reduction in the final statevector support is. They default to `--shots 1e8`, which is slow; for a quick local test, pass something like `--shots 1e4`.
- `full_opt_maxcut.py` — full MaxCut instantiation: SPIQ initialization, multi-start COBYLA optimization, and comparison against baseline methods.
- `run_experiments.sh` — runs the scripts above with small test settings (edit args or uncomment lines as needed).



## Repository map

- `examples/` — Public end-to-end workflow notebooks
- `spiq/` — Core implementation
- `clapton/` — Clifford search engine used by SPIQ
- `biomarker_data/` — Biomarker PCBO sample data (`samples/n8` … `samples/n20`) and helpers
- `red_qaoa/` — Graph reduction used by `experiments/full_opt_maxcut.py`
- `experiments/` — Reproducible evaluation scripts from the paper

## Citation

If you find our work useful, please cite our paper:

```bibtex
@article{bharadwaj2026scalable,
  title={Scalable Clifford-Based Classical Initialization for the Quantum Approximate Optimization Algorithm},
  author={Bharadwaj, Dhanvi and Hou, Yuewen and Li, Guang-Yi and Ravi, Gokul Subramanian},
  journal={arXiv preprint arXiv:2602.14327},
  year={2026}
}
```

