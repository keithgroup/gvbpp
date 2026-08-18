# gvbpp

**GVB Perfect Pairing for PySCF** — a standalone GVB-PP solver running entirely within PySCF.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## What is GVB-PP?

The Generalized Valence Bond Perfect Pairing (GVB-PP) method correlates each electron pair with a bonding and a correlating natural orbital:

$$\Psi = \hat{\mathcal{A}}\left[\text{core}\;\prod_p \left(\sigma_{pa}\,\phi_{pa}\bar{\phi}_{pa} - \sigma_{pb}\,\phi_{pb}\bar{\phi}_{pb}\right)\right]$$

Unlike RHF, GVB-PP dissociates bonds correctly, because it carries the antibonding configuration for every pair. Unlike full CASSCF, its CI expansion holds only $2^n$ configurations rather than a combinatorially growing set.

The payoff is interpretive as much as numerical. Each pair comes with an overlap

$$S_{ab} = \frac{1-r}{1+r}, \qquad r = \sqrt{n_\text{corr}/n_\text{bond}}$$

The perfect-pairing function is $\cos^2\theta\,|bb\rangle - \sin^2\theta\,|cc\rangle$, so the CI *coefficients* go as $\cos^2$ and $\sin^2$, not $\cos$ and $\sin$. Writing $S_{ab}$ as $(n_\text{bond}-n_\text{corr})/2$ confuses the two: it is correct at both limits and too large in between (0.976 rather than 0.803 for H₂). That was a real bug here through v0.3.0.

that behaves like a continuous bond order: near 1 for an ordinary covalent bond, falling toward 0 as the bond breaks or becomes strongly correlated.

**Reference:** F. W. Bobrowicz and W. A. Goddard III, "The Self-Consistent Field Equations for Generalized Valence Bond and Open-Shell Hartree–Fock Wave Functions," in *Methods of Electronic Structure Theory*, ed. H. F. Schaefer III (Plenum, 1977), pp. 79–127.

## Installation

```bash
git clone https://github.com/keithgroup/gvbpp.git
cd gvbpp
pip install -e .
```

**Requires:** PySCF ≥ 2.0, NumPy.

## Quick start

```python
from pyscf import gto
from gvbpp import gvb_pp

mol = gto.M(atom='H 0 0 0; H 0 0 1.4', unit='Bohr', basis='cc-pvdz')
res = gvb_pp(mol, n_pairs=1)

res.e_tot        # -1.1469081375   (matches CASSCF(2,2) to 2e-10)
res.overlaps     # array([0.803])
res['energy']    # dictionary access also works
```

Mind the units: `gto.M` defaults to Ångström. Writing `1.4` without `unit='Bohr'` builds a molecule at 1.4 Å, not 1.4 $a_0$.

## Examples

### H₂ dissociation

```python
from pyscf import gto, scf
from gvbpp import GVBPP

for R in [1.0, 2.0, 4.0, 6.0]:
    mol = gto.M(atom=f'H 0 0 0; H 0 0 {R}', unit='Bohr',
                basis='cc-pvdz', verbose=0)
    g = GVBPP(mol, n_pairs=1)
    g.kernel(verbose=0)
    print(f"R={R:.1f}  E={g.e_tot:.8f}  S_ab={g.overlaps[0]:.3f}")
```

```
R=1.0  E=-1.12456739  S_ab=0.987
R=2.0  E=-1.09437840  S_ab=0.938
R=4.0  E=-1.01152827  S_ab=0.484
R=6.0  E=-0.99911084  S_ab=0.113
```

RHF gives −0.816 at R = 6 $a_0$, about 115 kcal/mol too high. GVB-PP reaches −0.9991 against an FCI value of −0.9991.

### Water, four valence pairs

```python
from pyscf import gto
from gvbpp import gvb_pp

mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
            basis='cc-pvdz')
res = gvb_pp(mol, n_pairs=4)   # 2 O-H bonds + 2 lone pairs
```

The two O–H bond pairs come out equivalent, as do the two lone pairs — which is a useful check that the pairing worked.

### Custom SCF backend

```python
from pyscf import gto, dft
from gvbpp import GVBPP

mol = gto.M(atom='H 0 0 0; H 0 0 1.4', unit='Bohr', basis='cc-pvdz')
mf = dft.RKS(mol); mf.xc = 'pbe'; mf.kernel()

g = GVBPP(mol, n_pairs=1)
g.kernel(mf=mf)
```

## API

| Function / class | Description |
|---|---|
| `gvb_pp(mol, n_pairs)` | Convenience wrapper: run, analyze, return the `GVBPP` object |
| `GVBPP(mol, n_pairs)` | Full driver class |
| `GVBPPSolver(n_pairs)` | Low-level PP-restricted CI solver; plugs into any PySCF CASSCF |
| `pair_guess(mf, n_pairs)` | Build a correctly ordered active space |

Results support both attribute and dictionary access:

| Attribute | Key | Meaning |
|---|---|---|
| `.e_tot` | `'energy'`, `'e_tot'` | Total GVB-PP energy |
| `.e_corr` | `'e_corr'` | Energy relative to the SCF reference |
| `.overlaps` | `'overlap'` | $S_{ab}$ per pair |
| `.occ` | `'occ'` | `(n_bond, n_corr)` per pair |
| `.occupations` | `'occupations'` | Flat interleaved occupation array |
| `.mo_coeff` | `'mo_coeff'` | Full MO coefficient matrix |
| `.mo_coeff_pair` | `'mo_coeff_pair'` | List of `(nao, 2)` arrays, one per pair |
| `.converged` | `'converged'` | Orbital optimization converged |

## How it works

1. Run RHF (or any SCF) for starting orbitals.
2. Build the active space: choose which occupied orbital is each pair's bonding orbital and which virtual is its correlating partner, ordered as `[bond_1, corr_1, bond_2, corr_2, ...]`.
3. Optimize orbitals with PySCF's CASSCF driver, with `GVBPPSolver` replacing the full CI solver and re-solving the $2^n$-configuration PP problem at every macro iteration.

Three implementation details are load-bearing, and each fails silently if dropped:

- **Active-orbital ordering.** Handing the solver energy-ordered canonical orbitals places electron pairs in virtual orbitals and returns energies *above* Hartree–Fock.
- **Active–active rotation.** For a full-CI solver these rotations are redundant and PySCF skips them by default. For a restricted PP CI they are not redundant — they are how pairs find their shapes. `internal_rotation` is worth ~15 kcal/mol on water.
- **Two-step optimization.** The PP CI vector does not transform simply under active-orbital rotation, so one-step CASSCF micro-iterations work from an inconsistent CI response and fail to converge. Re-solving the CI every macro iteration is the classical alternating GVB-PP algorithm, and it converges.

### Pair selection

Which pairs you correlate is a modeling choice, and the obvious default is wrong more often than you would like. In F₂ the HOMO is a π\* lone-pair orbital and the σ bond sits three orbitals lower, so asking for one pair and taking the HOMO quietly correlates a lone pair instead of the bond.

`gvbpp` therefore Boys-localizes a window of valence occupied and low-lying virtual orbitals and pairs them by the exchange integral $K_{iv} = (iv|iv)$. This is not a heuristic: $K$ *is* the off-diagonal element of the PP Hamiltonian coupling a pair's two configurations, so the virtual that couples most strongly to a given bond is that bond's correlating orbital. On F₂ the σ/σ\* pair wins on $K$ by a factor of three and is selected correctly.

### Multiple minima

Orbital optimization under the perfect-pairing restriction is not convex. Different starting guesses reach different stationary points — on water they differ by about 6 kcal/mol — and neither systematic guess wins in general. A guess sitting near a bifurcation can even land in different basins on repeated runs of identical input, because threaded BLAS reductions are not bitwise reproducible.

The default `guess='auto'` therefore multi-starts: localized, canonical, and `n_trials` seeded orthogonal perturbations, keeping the lowest energy. This makes the reported minimum robust and reproducible rather than lucky. Pass `n_trials=0` for a faster, less thorough search, or an explicit `guess=` to study the dependence itself.

## Validation

Every release is checked against CASSCF references and variational bounds:

| System | pairs | RHF | GVB-PP | CASSCF(2n,2n) | % corr. |
|---|---|---|---|---|---|
| H₂ (1.4 $a_0$) | 1 | −1.1287094 | −1.1469081 | −1.1469081 | 100 |
| H₂ (6.0 $a_0$) | 1 | −0.8159903 | −0.9991108 | −0.9991108 | 100 |
| LiH | 1 | −7.9836159 | −8.0001416 | −8.0001415 | 100 |
| H₂O | 4 | −76.0267936 | −76.0895426 | −76.1114372 | 74 |
| NH₃ | 4 | −56.1955108 | −56.2587906 | −56.3000587 | 61 |
| CH₄ | 4 | −40.1987072 | −40.2583269 | −40.2797635 | 74 |
| N₂ | 3 | −108.9540866 | −109.0288385 | −109.0900538 | 55 |

For a single pair the PP space *is* the CAS space, so agreement is exact. For several pairs, GVB-PP recovers roughly 55–75% of the CASSCF correlation energy; the remainder is inter-pair correlation, which perfect pairing excludes by construction.

Physically meaningful overlaps fall out directly — CH₄ gives four identical $S_{ab} = 0.835$, and N₂ gives two degenerate π pairs at 0.695 plus a σ pair at 0.894.

```bash
pytest tests/ -v
```

## Limitations

- **Perfect pairing only.** No inter-pair excitations, so roughly a third of the CASSCF correlation energy in the same active space is missing by design. GVB-PP is a model of bonding, not a route to chemical accuracy.
- **Closed-shell references.** Open-shell and spin-coupled GVB (the full GVB-CI of Goddard's original work) are not implemented.
- **Restricted to singlet-coupled pairs.**
- Cost is dominated by the CASSCF orbital optimization, so it grows quickly with basis size.

## Citation

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
