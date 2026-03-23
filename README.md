# gvbpp

**GVB Perfect Pairing for PySCF** -- the first standalone GVB-PP solver running entirely within PySCF.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## What is GVB-PP?

The Generalized Valence Bond Perfect Pairing (GVB-PP) method correlates each electron pair with a bonding and correlating natural orbital:

$$\Psi = \hat{\mathcal{A}}\left[\text{core}\;\prod_p \left(\sigma_{pa}\,\phi_{pa}\bar{\phi}_{pa} - \sigma_{pb}\,\phi_{pb}\bar{\phi}_{pb}\right)\right]$$

Unlike RHF, GVB-PP correctly describes bond dissociation because it includes the antibonding configuration for each pair. Unlike full CASSCF, it scales linearly with the number of pairs (2^n configurations vs combinatorial).

**Reference:** F.W. Bobrowicz and W.A. Goddard III, "The Self-Consistent Field Equations for Generalized Valence Bond and Open-Shell Hartree-Fock Wave Functions," in *Methods of Electronic Structure Theory*, ed. H.F. Schaefer III (Plenum, 1977), pp. 79-127.

## Installation

```bash
# Development install
git clone https://github.com/keithgroup/gvbpp.git
cd gvbpp
pip install -e .

# Or directly
pip install gvbpp
```

**Requires:** PySCF >= 2.0, NumPy

## Quick Start

```python
from pyscf import gto
from gvbpp import gvb_pp

mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='cc-pvdz')
g = gvb_pp(mol, n_pairs=1)
# GVB-PP(1) energy: -1.0685442296
# Matches CASSCF(2,2) to < 1e-9 hartree
```

## Features

- **Single-pair GVB-PP:** Exact, uses CASSCF(2,2) with PP-constrained CI solver
- **Multi-pair GVB-PP:** Sequential pair-by-pair optimization
- **Natural orbital analysis:** Occupation numbers, pair overlaps, CI coefficients
- **Pure PySCF:** No external programs needed (no GAMESS, no Gaussian)
- **Pluggable SCF backend:** Swap RHF for any `mf` object (DFT, semi-empirical, etc.)

## Examples

### H2 dissociation curve

```python
from pyscf import gto, scf
from gvbpp import GVBPP

for R in [1.0, 2.0, 4.0, 6.0]:
    mol = gto.M(atom=f'H 0 0 0; H 0 0 {R}', basis='cc-pvdz', verbose=0)
    g = GVBPP(mol, n_pairs=1)
    g.kernel(verbose=0)
    print(f"R={R:.1f}  E={g.e_tot:.8f}  occ={g.occupations}")
```

### H2O with multiple pairs

```python
from pyscf import gto
from gvbpp import gvb_pp

mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
            basis='cc-pvdz')
g = gvb_pp(mol, n_pairs=4, verbose=3)  # 2 bonds + 2 lone pairs
```

### Custom SCF backend

```python
from pyscf import gto, dft
from gvbpp import GVBPP

mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='cc-pvdz')
mf = dft.RKS(mol)  # Use DFT instead of HF
mf.xc = 'pbe'
mf.kernel()

g = GVBPP(mol, n_pairs=1)
g.kernel(mf=mf)  # GVB-PP on top of DFT orbitals
```

## API

| Function/Class | Description |
|----------------|-------------|
| `gvb_pp(mol, n_pairs)` | Convenience function: run and analyze |
| `GVBPP(mol, n_pairs)` | Full driver class |
| `GVBPPSolver(n_pairs)` | Low-level CI solver (plug into any CASSCF) |

## How It Works

1. Run RHF (or any SCF) to get starting orbitals
2. For each pair: set up CASSCF(2,2) with `GVBPPSolver`
3. `GVBPPSolver` builds the 2x2 PP Hamiltonian (bonding vs correlating occupation) and diagonalizes it
4. CASSCF optimizes the orbitals; `GVBPPSolver` re-solves the CI at each macro iteration
5. Extract natural orbital occupations and pair properties

For multi-pair, pairs are optimized sequentially (pair-by-pair strong orthogonality approximation).

## Citation

If you use this code, please cite:

```bibtex
@incollection{bobrowicz1977,
  author    = {Bobrowicz, Frank W. and Goddard, William A., III},
  title     = {The Self-Consistent Field Equations for Generalized Valence Bond
               and Open-Shell {Hartree-Fock} Wave Functions},
  booktitle = {Methods of Electronic Structure Theory},
  editor    = {Schaefer, Henry F., III},
  publisher = {Plenum Press},
  address   = {New York},
  year      = {1977},
  pages     = {79--127},
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Acknowledgments

- William A. Goddard III (Caltech) for developing GVB theory
- The PySCF development team for the quantum chemistry infrastructure
- Keith Group (University of Pittsburgh)
