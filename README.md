# SPIQ
 
Making CAFQA work for QAOA problems with ma-QAOA anstaz.

# Example 

Example on the full workflow using the code for a max-cut problen is given in `testing_scripts/maxcut_qaoa.ipynb`.

## Required Packages

To run this project, you will need a few packages:

```bash
pip install -r requirements.txt
```

# Multi-Start properites

For relevant code on choosing n-best cafqa points, search for `best_cafqa_gen_params`, `best_cafqa_gen_fitness` in `qaoa_utils` module.

A full implementation of multi-start can be found in `testing_scripts/cafqa_diff_points_maxcut.py` 