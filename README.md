# GitHub release description — v1.0.0 (PRA submission release)

## Rank-One Access Fisher-Information Reproducibility Package v1.0.0

This release accompanies the manuscript:

**Aishwarya Vishwakarma, “Fisher information under rank-one access: exact information routing, geometric scaling laws, and finite-time limits.”**

The package contains the complete numerical and analytical reproducibility materials used for the Physical Review A submission:

- exact rank-one access / information-routing tests on random passive networks;
- chain scaling and spatial-profile calculations;
- intrinsic-loss and information-routing scans;
- resource-normalized Fisher-information calculations;
- 1D/2D/3D port-codimension simulations;
- finite-duration, hard-deadline, stored-information, and propagation-limit calculations;
- weak-drive finite-Hilbert-space Lindblad validation;
- all processed numerical data;
- regression tests and pinned environment information.

### Reproduce the paper

```bash
python -m pip install -r requirements.txt
./build.sh
```

Use `./build.sh --full` to recompute the expensive scans rather than redraw them from the archived CSV data.

### Scientific scope

The package treats finite-dimensional passive Markovian linear input-output networks and coherent-state quantum Fisher information. It does not claim a nonclassical-probe advantage or a universal strong-drive interacting result.

## v1.1.0 — revision

To reproduce the revised calculations and figures, run these commands
from the repository root:

```bash
python -m pip install -r requirements.txt
bash revision-scripts/run_all.sh
```
