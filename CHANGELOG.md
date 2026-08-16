# Changelog

## 0.3.1 (2026-08-16)

### Fixed

- **`overlaps` reported the wrong quantity.** `S_ab` was computed as
  `(n_bond - n_corr)/2`, which equals `c_b^2 - c_c^2`. But the
  perfect-pairing function is `cos^2(t)|bb> - sin^2(t)|cc>`, so the CI
  coefficients go as `cos^2` and `sin^2`, not `cos` and `sin`. The correct
  overlap is `(1 - r)/(1 + r)` with `r = sqrt(n_corr/n_bond)`.

  The old formula is exact at both limits — 1 for the MO pair, 0 at
  homolytic dissociation — and too large everywhere in between, which is
  why it went unnoticed. For H2/cc-pVDZ at 1.4 a0 it gave 0.976 where the
  GVB orbitals, built explicitly from the CASSCF vector and overlapped
  through the AO metric, give 0.803.

  Affected values, recomputed: H2 0.976 -> 0.803, CH4 0.984 -> 0.835,
  N2 sigma 0.994 -> 0.894 and pi 0.937 -> 0.695, F2 sigma 0.862 -> 0.572.
  **Energies are unaffected**; only the reported overlaps change.

  A regression test now compares `overlaps` against the AO overlap of
  orbitals built explicitly from the CASSCF vector, at three geometries.

## 0.2.0 (2026-08-15)

Multi-pair GVB-PP was wrong in 0.1.0 and is now correct. If you produced any
multi-pair numbers with 0.1.0, they need to be regenerated.

### Fixed

- **Multi-pair energies were not GVB-PP energies.** The 0.1.0 driver
  optimized one pair at a time with all other pairs frozen as closed-shell
  core, then reported the *last* pair's CASSCF(2,2) energy as the n-pair
  result. Consequences: the energy was not variational in the n-pair
  wavefunction, and it was not even monotonic in `n_pairs` — water gave
  GVB-PP(3) *above* GVB-PP(2). All pairs are now correlated simultaneously
  in a single wavefunction.

- **`GVBPPSolver(n)` for n > 1 returned nonsense.** The solver requires
  active orbitals interleaved as `[bond_1, corr_1, bond_2, corr_2, ...]`,
  but was being handed energy-ordered canonical orbitals. That placed
  electron pairs in virtual orbitals; water came out 3.4 Hartree *above*
  RHF. Active-space construction now guarantees the required ordering.

- **Active–active orbital rotations were disabled.** PySCF skips them by
  default because they are redundant for a full-CI solver. They are not
  redundant for a restricted PP CI. Enabling `internal_rotation` is worth
  about 15 kcal/mol on water.

- **One-step CASSCF did not converge.** The PP CI vector does not transform
  simply under active-orbital rotation, so augmented-Hessian micro-iterations
  worked from an inconsistent CI response. The driver now uses the classical
  alternating (two-step) algorithm, which converges reliably.

- **`examples/03_f2_strong_correlation.py` computed the wrong molecule.**
  `gto.M` defaults to Ångström, so an unlabeled `2.668` put F₂ at nearly
  twice its equilibrium bond length while the output described equilibrium
  behavior. `examples/01_h2_dissociation.py` had the same bug, labeling its
  axis "bohr" while computing in Ångström. Both now state `unit='Bohr'`.

### Added

- `gvbpp.pairing` — active-space construction. Boys-localizes a valence
  window and pairs orbitals by the exchange integral K, which is the actual
  PP Hamiltonian coupling rather than a proxy for it. This matters: F₂'s HOMO
  is a π\* lone pair, so energy-ordered selection correlates a lone pair
  instead of the σ bond.

- Multi-start orbital optimization (`guess='auto'`, the default). PP orbital
  optimization is non-convex and a single guess near a bifurcation can land
  in different basins on repeated runs of identical input, since threaded
  BLAS is not bitwise reproducible. The driver now optimizes from localized,
  canonical, and `n_trials` seeded perturbations and keeps the lowest.

- `overlaps` attribute: the GVB pair overlap S_ab for
  every pair, from a single converged wavefunction.

- Dictionary-style access on results (`res['energy']`, `res['overlap']`,
  `res['occ']`, `res['mo_coeff_pair']`) alongside attribute access, plus
  `to_dict()`, `get_pair_orbitals(p)` and `mo_coeff_pair`.

- `npairs` accepted as an alias for `n_pairs`.

- Sanity checks that warn when the GVB-PP energy exceeds the SCF reference
  or a pair's occupations do not sum to 2 — the signatures of a broken
  active space.

### Testing

The 0.1.0 suite asserted only that calculations ran and beat RHF, which the
broken multi-pair code satisfied. The suite now checks the properties that
actually constrain the method:

- GVB-PP(n) never above the SCF reference
- GVB-PP(n) never below the CASSCF(2n,2n) floor
- monotonic decrease with `n_pairs`
- GVB-PP(1) exact against CASSCF(2,2)
- correct H₂ dissociation limit and monotonic S_ab decay
- F₂ selects the σ bond, not the frontier lone pair
- N₂ gives two degenerate π pairs
- `internal_rotation` demonstrably changes the answer
- repeated runs of identical input agree

## 0.1.0 (2026-03-23)

- Initial release
- GVB-PP solver with PySCF CASSCF backend
- Single-pair optimization via CASSCF(2,2) (exact)
- Multi-pair sequential optimization (pair-by-pair) — **incorrect, see 0.2.0**
- Natural orbital analysis and pair property extraction
