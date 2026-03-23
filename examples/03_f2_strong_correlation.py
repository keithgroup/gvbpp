"""
F2: Strong Correlation in the Sigma Bond
=========================================

F2 is notorious for strong static correlation in its sigma bond.
GVB-PP reveals this through nearly equal natural orbital occupations.
"""

from pyscf import gto, scf
from gvbpp import gvb_pp

mol = gto.M(atom='F 0 0 0; F 0 0 2.668', basis='cc-pvdz')

g = gvb_pp(mol, n_pairs=1, verbose=4)

# A pair overlap near 0 means strong correlation (nearly equal occupations)
# A pair overlap near 1 means weak correlation (bonding orbital dominant)
print(f"\nF2 sigma bond pair overlap: {g.ci_coeffs[0][0]**2 - g.ci_coeffs[0][1]**2:.4f}")
print("(Compare to H2 at equilibrium where overlap ~ 0.85)")
