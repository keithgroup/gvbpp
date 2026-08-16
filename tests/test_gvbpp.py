"""
Regression tests for GVB-PP.

These tests are deliberately written to fail on the failure modes that a
"does it run?" test suite misses. In particular:

  * A GVB-PP(n) energy must never lie above the SCF reference.
  * It must never lie below the corresponding CASSCF(2n, 2n) energy.
  * It must decrease monotonically as pairs are added.

The v0.1.0 multi-pair implementation passed a run-and-beat-RHF test suite
while violating all three.
"""

import numpy as np
import pytest
from pyscf import gto, scf, mcscf

from gvbpp import GVBPP, GVBPPSolver, gvb_pp, pair_guess

HARTREE2KCAL = 627.5094740631


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope='module')
def h2():
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', unit='Bohr',
                basis='cc-pvdz', verbose=0)
    return mol, scf.RHF(mol).run()


@pytest.fixture(scope='module')
def h2o():
    mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
                basis='cc-pvdz', verbose=0)
    return mol, scf.RHF(mol).run()


# ----------------------------------------------------------------------
# Single pair: GVB-PP(1) is exactly CASSCF(2,2)
# ----------------------------------------------------------------------
def test_h2_matches_casscf(h2):
    """For one pair the PP space is the full CAS space, so this is exact."""
    mol, mf = h2
    g = GVBPP(mol, n_pairs=1)
    g.kernel(mf=mf, verbose=0)

    mc = mcscf.CASSCF(mf, 2, 2)
    mc.verbose = 0
    mc.kernel()

    assert abs(g.e_tot - mc.e_tot) < 1e-8


def test_h2_lower_than_rhf(h2):
    mol, mf = h2
    g = GVBPP(mol, n_pairs=1)
    g.kernel(mf=mf, verbose=0)
    assert g.e_tot < mf.e_tot
    assert g.converged


def test_h2_occupations_sum_to_two(h2):
    mol, mf = h2
    g = GVBPP(mol, n_pairs=1)
    g.kernel(mf=mf, verbose=0)
    assert abs(g.occupations.sum() - 2.0) < 1e-8
    assert np.all(g.occupations >= -1e-10)
    assert np.all(g.occupations <= 2.0 + 1e-10)


def test_h2_dissociation(h2):
    """At long range GVB-PP must reach 2 x E(H), where RHF fails badly."""
    mol = gto.M(atom='H 0 0 0; H 0 0 6.0', unit='Bohr',
                basis='cc-pvdz', verbose=0)
    g = GVBPP(mol, n_pairs=1)
    g.kernel(verbose=0)
    e_2h = -0.99855  # 2 x E(H) in cc-pVDZ
    assert abs(g.e_tot - e_2h) < 5e-3
    # the pair overlap must collapse as the bond breaks
    assert g.overlaps[0] < 0.2


def test_h2_overlap_decreases_with_distance():
    """S_ab is the bond-order indicator used throughout the primer."""
    overlaps = []
    for R in [1.0, 2.0, 4.0, 6.0]:
        mol = gto.M(atom=f'H 0 0 0; H 0 0 {R}', unit='Bohr',
                    basis='cc-pvdz', verbose=0)
        g = GVBPP(mol, n_pairs=1)
        g.kernel(verbose=0)
        overlaps.append(g.overlaps[0])
    assert all(a > b for a, b in zip(overlaps, overlaps[1:])), overlaps


# ----------------------------------------------------------------------
# Multi-pair: the tests v0.1.0 would have failed
# ----------------------------------------------------------------------
def test_multipair_never_above_scf(h2o):
    """A variational wavefunction cannot be worse than its own reference."""
    mol, mf = h2o
    for n in [1, 2, 3, 4]:
        g = GVBPP(mol, n_pairs=n)
        g.kernel(mf=mf, verbose=0, n_trials=1)
        assert g.e_tot < mf.e_tot, f'GVB-PP({n}) is above RHF'


def test_multipair_monotonic(h2o):
    """Adding a pair adds variational freedom, so energy must not rise."""
    mol, mf = h2o
    energies = []
    for n in [1, 2, 3, 4]:
        g = GVBPP(mol, n_pairs=n)
        g.kernel(mf=mf, verbose=0, n_trials=1)
        energies.append(g.e_tot)
    for n, (lo_, hi) in enumerate(zip(energies, energies[1:]), start=1):
        assert hi <= lo_ + 1e-7, (
            f'GVB-PP({n + 1}) = {hi:.8f} is above '
            f'GVB-PP({n}) = {lo_:.8f}'
        )


def test_multipair_above_casscf_floor(h2o):
    """PP is a restriction of CAS, so CASSCF(2n,2n) is a strict lower bound."""
    mol, mf = h2o
    n = 3
    g = GVBPP(mol, n_pairs=n)
    g.kernel(mf=mf, verbose=0)

    mc = mcscf.CASSCF(mf, 2 * n, 2 * n)
    mc.verbose = 0
    mc.kernel()

    assert g.e_tot >= mc.e_tot - 1e-7, 'GVB-PP fell below the CASSCF floor'
    assert g.e_tot <= mf.e_tot


def test_multipair_converges(h2o):
    mol, mf = h2o
    g = GVBPP(mol, n_pairs=4)
    g.kernel(mf=mf, verbose=0)
    assert g.converged


def test_multipair_pair_occupations(h2o):
    """Every pair must hold exactly two electrons."""
    mol, mf = h2o
    n = 4
    g = GVBPP(mol, n_pairs=n)
    g.kernel(mf=mf, verbose=0)
    for p in range(n):
        tot = g.occupations[2 * p] + g.occupations[2 * p + 1]
        assert abs(tot - 2.0) < 1e-7, f'pair {p + 1} holds {tot} electrons'
    assert len(g.overlaps) == n
    assert np.all(g.overlaps >= -1e-8) and np.all(g.overlaps <= 1.0 + 1e-8)


def test_internal_rotation_matters(h2o):
    """
    Guards the fix itself. Freezing active-active rotations must give a
    strictly worse energy; if this test starts passing trivially, the
    internal_rotation flag has been lost.
    """
    mol, mf = h2o
    n = 4
    mo, _ = pair_guess(mf, n, method='localized')
    ncore = mol.nelectron // 2 - n

    def run(internal):
        mc = mcscf.CASSCF(mf, 2 * n, 2 * n)
        mc.ncore = ncore
        mc.verbose = 0
        mc.fcisolver = GVBPPSolver(n)
        mc.internal_rotation = internal
        mc.max_cycle_micro = 1
        mc.max_cycle_macro = 200
        mc.natorb = False
        mc.kernel(mo)
        return mc.e_tot

    assert run(True) < run(False) - 1e-4


# ----------------------------------------------------------------------
# Pair selection
# ----------------------------------------------------------------------
def test_pair_guess_ordering(h2o):
    """The active block must be interleaved [bond, corr, bond, corr, ...]."""
    mol, mf = h2o
    n = 3
    mo, info = pair_guess(mf, n, method='localized')
    assert mo.shape == mf.mo_coeff.shape
    assert len(info['assignment']) == n
    # exchange integrals must be positive and sorted descending by construction
    ks = info['pair_exchange']
    assert all(k > 0 for k in ks)
    assert ks == sorted(ks, reverse=True)


def test_f2_selects_sigma_bond_not_lone_pair():
    """
    F2's HOMO is pi*, not the sigma bond. Selecting on orbital energy gives
    a lone pair; selecting on exchange coupling gives the bond. This test
    pins the behavior that distinguishes the two.
    """
    mol = gto.M(atom='F 0 0 0; F 0 0 2.668', unit='Bohr',
                basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).run()

    g = GVBPP(mol, n_pairs=1)
    g.kernel(mf=mf, verbose=0)

    # HOMO/LUMO CASSCF(2,2) correlates the lone pair and recovers far less
    mc = mcscf.CASSCF(mf, 2, 2)
    mc.verbose = 0
    mc.kernel()

    assert g.e_tot < mc.e_tot - 0.01, (
        'GVB-PP picked the frontier lone pair instead of the sigma bond'
    )
    # F2's sigma bond is the textbook weakly-bound, strongly-correlated case.
    # Window recalibrated when the S_ab formula was corrected: the old
    # (n_b - n_c)/2 gave 0.862 here, the true AO overlap of the GVB orbitals
    # is 0.572 (verified by building them from the CASSCF vector). For
    # contrast the frontier lone pair this test exists to avoid comes out at
    # 0.873, so the two are separated by 0.30 and the BOND is the lower one.
    assert 0.45 < g.overlaps[0] < 0.70
    assert g.pair_info['pair_exchange'][0] > 0.2


def test_n2_triple_bond_pairs():
    """N2 should give two degenerate pi pairs plus one sigma pair."""
    mol = gto.M(atom='N 0 0 0; N 0 0 1.098', basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).run()
    g = GVBPP(mol, n_pairs=3)
    g.kernel(mf=mf, verbose=0)

    assert g.converged
    assert g.e_tot < mf.e_tot
    s = np.sort(g.overlaps)
    # the two pi pairs are symmetry-equivalent and must come out degenerate
    assert abs(s[0] - s[1]) < 1e-3, f'pi pairs not degenerate: {g.overlaps}'


def test_pair_guess_rejects_too_many_pairs(h2o):
    mol, mf = h2o
    with pytest.raises(ValueError):
        pair_guess(mf, 999, method='localized')


def test_auto_guess_takes_the_best(h2o):
    """
    Orbital optimization under the PP restriction is not convex and the two
    guesses land on different stationary points. 'auto' must never be worse
    than either one alone.
    """
    mol, mf = h2o
    auto = GVBPP(mol, 4)
    auto.kernel(mf=mf, verbose=0, guess='auto')
    loc = GVBPP(mol, 4)
    loc.kernel(mf=mf, verbose=0, guess='localized')
    can = GVBPP(mol, 4)
    can.kernel(mf=mf, verbose=0, guess='canonical')

    assert auto.e_tot <= min(loc.e_tot, can.e_tot) + 1e-8
    assert auto.guess_used is not None


def test_auto_is_reproducible(h2o):
    """
    A single guess near a bifurcation lands in different basins on repeated
    runs, because threaded BLAS is not bitwise reproducible. Seeded
    multi-start must remove that.
    """
    mol, mf = h2o
    energies = []
    for _ in range(3):
        g = GVBPP(mol, 4)
        g.kernel(mf=mf, verbose=0)
        energies.append(g.e_tot)
    assert max(energies) - min(energies) < 1e-7, energies


def test_guess_dependence_is_real(h2o):
    """Documents why 'auto' exists; if this ever stops holding, simplify."""
    mol, mf = h2o
    loc = GVBPP(mol, 4)
    loc.kernel(mf=mf, verbose=0, guess='localized')
    can = GVBPP(mol, 4)
    can.kernel(mf=mf, verbose=0, guess='canonical')
    assert abs(loc.e_tot - can.e_tot) > 1e-4


# ----------------------------------------------------------------------
# API surface
# ----------------------------------------------------------------------
def test_dict_and_attribute_access(h2):
    mol, mf = h2
    res = gvb_pp(mol, n_pairs=1, mf=mf, verbose=0, analyze=False)
    assert res['energy'] == res.e_tot
    assert res['e_corr'] == res.e_corr
    assert len(res['occ']) == 1
    assert len(res['occ'][0]) == 2
    assert res['mo_coeff_pair'][0].shape == (mol.nao, 2)
    assert res['mo_coeff'].shape == mf.mo_coeff.shape
    assert res['npairs'] == 1


def test_npairs_alias(h2):
    mol, mf = h2
    res = gvb_pp(mol, npairs=1, mf=mf, verbose=0, analyze=False)
    assert res.n_pairs == 1


def test_bad_key_message(h2):
    mol, mf = h2
    res = gvb_pp(mol, n_pairs=1, mf=mf, verbose=0, analyze=False)
    with pytest.raises(KeyError):
        res['not_a_key']


def test_lih_single_pair():
    mol = gto.M(atom='Li 0 0 0; H 0 0 3.0', unit='Bohr',
                basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).run()
    g = GVBPP(mol, n_pairs=1)
    g.kernel(mf=mf, verbose=0)
    assert g.e_tot < mf.e_tot
    assert g.converged


def test_pair_overlap_matches_explicit_gvb_orbitals():
    """
    The reported S_ab must equal the AO overlap of the GVB orbitals built
    explicitly from the CASSCF vector.

    This guards a real bug: S_ab was computed as (n_b - n_c)/2, which is
    c_b^2 - c_c^2. But the perfect-pairing function is cos^2(t)|bb> -
    sin^2(t)|cc>, so the CI coefficients go as cos^2 and sin^2, not cos and
    sin. The old formula agreed at both limits and was too large in between:
    0.976 against a true 0.803 for H2 at 1.4 a0.
    """
    import numpy as np
    from pyscf import gto, scf, mcscf
    from gvbpp import GVBPP

    for R, ref in [(1.4, 0.80253), (2.0, 0.69701), (3.0, 0.47132)]:
        mol = gto.M(atom=f'H 0 0 0; H 0 0 {R}', basis='cc-pvdz',
                    unit='Bohr', verbose=0)
        g = GVBPP(mol, n_pairs=1)
        g.kernel(verbose=0)

        mf = scf.RHF(mol).run(verbose=0)
        mc = mcscf.CASSCF(mf, 2, 2)
        mc.verbose = 0
        mc.kernel()
        S_ao = mol.intor('int1e_ovlp')
        nc = mc.ncore
        gm, um = mc.mo_coeff[:, nc], mc.mo_coeff[:, nc + 1]
        th = np.arctan(np.sqrt(abs(mc.ci[1, 1] / mc.ci[0, 0])))
        a = np.cos(th) * gm + np.sin(th) * um
        b = np.cos(th) * gm - np.sin(th) * um
        direct = float(a @ S_ao @ b)

        assert abs(g.overlaps[0] - direct) < 1e-4, (
            f'R={R}: reported {g.overlaps[0]:.5f}, direct {direct:.5f}')
        assert abs(g.overlaps[0] - ref) < 1e-3
