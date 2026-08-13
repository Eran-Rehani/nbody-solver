# Final project: N-body gravitational simulation

Course 77315, Computational Physics. Barnes-Hut tree with adaptive RK4 integration.

Eran Rehani

## Contents

| File | What it is |
|---|---|
| `report.pdf` | The 15-page report, including the method, checks, results, numerical difficulties, conclusion, and assigned-tolerance appendix. |
| `Final_project.ipynb` | Runnable companion to the report, with saved outputs. It imports `nbody_solver` and displays selected routines from the imported module. |
| `nbody_solver.py` | The main simulation source. Initial conditions, octree, both tree walks, the direct O(N²) reference, adaptive RK4, energy diagnostics, density profile, King fit, every figure, and the self-tests. |
| `build_notebook.py` | Regenerates `Final_project.ipynb`. A normal rebuild leaves code outputs empty; `--preserve-outputs` retains executed outputs after documentation-only changes and refreshes the fitted outputs. |
| `requirements.txt` | Python packages needed to run the solver and execute the notebook. |
| `figures/` | Saved output of the production runs: states, energy histories and escaper records for all six runs (`run_V*.npz`), V=80 snapshots for figures (a) to (c), plus every figure and table in the report. |

The report is the primary document. The notebook provides executable checks and
the same production results.

Install with `python3 -m pip install -r requirements.txt`. The solver requires
`numpy`, `scipy` and `matplotlib`. `numba` is optional at runtime. It compiles
the tree walk and is roughly twenty times faster; the pure-Python walk
performs the same calculation without it. Jupyter/IPython packages are needed
only to execute the notebook.

## Running

```bash
python3 nbody_solver.py --help      # all options
python3 nbody_solver.py --test      # numerical self-checks
```

The self-checks compare the tree against a direct O(N²) sum for both
acceleration and potential, integrate a softened binary over two periods against
its analytic period and separation, and exercise the degenerate geometries that
break naive octrees.

### Redrawing from the saved states

These commands read `figures/*.npz`, repeat no dynamics, and take seconds:

```bash
python3 nbody_solver.py --redraw          # figures (a), (b), (c)  [Figs. 3-5]
python3 nbody_solver.py --refit           # figure (d) + King table [Fig. 6, Table 4]
python3 nbody_solver.py --tolerance-fig   # tolerance comparison    [Fig. 2]
```

Add `--tag=eps01` to any of these to rebuild the Appendix A versions, e.g.
`python3 nbody_solver.py --refit --tag=eps01`. `--refit` prints the fitted King
parameters as it runs, so its output can be checked directly against Table 4.

### Reproducing from scratch

```bash
python3 nbody_solver.py --tol=0.1332              # Section 5 (a few minutes per velocity)
python3 nbody_solver.py --tol=133.2 --tag=eps01   # Appendix A
python3 nbody_solver.py --convergence             # Table 2
python3 nbody_solver.py --theta-table             # Table 1
python3 nbody_solver.py --quick                   # fast sanity run at N = 500, tagged _quick
```

The random seed is fixed, so these reproduce the supplied numerical states and
the numbers quoted in the report. Runtime fields and compressed `.npz` bytes can
vary with hardware and library versions. `--quick` writes with the tag `_quick`
to avoid overwriting the production states; inspect its output
with `--redraw --tag=quick` and `--refit --tag=quick`.

## Integration tolerance

The assignment specifies the RK4 accuracy as ε = 0.1 in units of the calculation
box, or a permitted step error of 133.2 kpc, larger than the cluster being
simulated. Section 4 of the report measures what that costs and explains why the
results in Section 5 use 0.1332 kpc instead. Appendix A gives every requested
output at the assigned value as well, so both are in the report.
