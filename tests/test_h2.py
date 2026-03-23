"""Tests for GVB-PP against known results."""

import numpy as np
import pytest
from pyscf import gto, scf, mcscf, fci
from gvbpp import GVBPP


def test_h2_matches_casscf():
    """GVB-PP(1) for H2 must match CASSCF(2,2) to 1e-8."""
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).run()

    g = GVBPP(mol, n_pairs=1)
    g.kernel(verbose=0)

    mc = mcscf.CASSCF(mf, 2, 2)
    mc.verbose = 0
    mc.kernel()

    assert abs(g.e_tot - mc.e_tot) < 1e-8


def test_h2_lower_than_rhf():
    """GVB-PP must give lower energy than RHF."""
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).run()
    g = GVBPP(mol, n_pairs=1)
    g.kernel(verbose=0)
    assert g.e_tot < mf.e_tot


def test_h2_dissociation():
    """At large R, GVB-PP must approach 2*E(H), not the wrong RHF limit."""
    mol = gto.M(atom='H 0 0 0; H 0 0 6.0', basis='cc-pvdz', verbose=0)
    g = GVBPP(mol, n_pairs=1)
    g.kernel(verbose=0)
    assert g.e_tot < -0.99  # 2*E(H) ~ -1.0


def test_h2_occupations_sum_to_two():
    """Natural orbital occupations must sum to 2."""
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='cc-pvdz', verbose=0)
    g = GVBPP(mol, n_pairs=1)
    g.kernel(verbose=0)
    assert abs(sum(g.occupations) - 2.0) < 1e-8
    assert all(0 <= occ <= 2 for occ in g.occupations)


def test_lih_single_pair():
    """GVB-PP(1) for LiH must converge and lower the energy."""
    mol = gto.M(atom='Li 0 0 0; H 0 0 3.0', basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).run()
    g = GVBPP(mol, n_pairs=1)
    g.kernel(verbose=0)
    assert g.e_tot < mf.e_tot
    assert g.mc.converged


def test_h2o_multipair():
    """Multi-pair GVB-PP for H2O must converge and lower the energy."""
    mol = gto.M(
        atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
        basis='cc-pvdz', verbose=0,
    )
    mf = scf.RHF(mol).run()
    g = GVBPP(mol, n_pairs=2)
    g.kernel(verbose=0)
    assert g.e_tot < mf.e_tot
