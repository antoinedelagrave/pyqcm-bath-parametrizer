# Pyqcm Bath Parametrization

Symmetry-constrained bath parametrization for Cluster Dynamical Mean-Field Theory (CDMFT) simulations using [`pyqcm`](https://github.com/pyqcm-project/pyqcm).

## Setup

```bash
pip install -e .
```

> Here the option `-e` means that the package is *editable*. In that case, the link from
> your virtual environnment to the package is made with the original folder. So, any
> changes on the original package's files take effect immediately.

## Usage

```python
from bath_parametrization import BathParametrizer
import numpy as np

atoms = np.array([[-1,-1,0],[1,-1,0],[-1,1,0],[1,1,0]], dtype=float)
p = BathParametrizer(atoms, "C2v")
p.show_parametrized_cluster()
salcs = p.get_bath_parametrization()  # {irrep: [salc, ...]}
```

## Supported point groups

`Cs`, `C2`, `C2v`, `C3`, `C3v`, `C4`, `C4v`, `C6` and `C6v` see `src/bath_parametrization/point_groups.py`.

## Todo

- [ ] Examples for 1D and 2D models.
- [ ] Examples for each type of coupling.
