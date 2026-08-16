"""
F2: Strong Correlation in the Sigma Bond
========================================

F2 is the textbook case of a bond that Hartree-Fock describes badly. The
sigma bond is weak (D_e ~ 38 kcal/mol) and strongly correlated, and GVB-PP
exposes this directly through the natural orbital occupations: the
correlating orbital picks up far more occupation than in a normal covalent
bond.

Two traps worth knowing about, both of which this example used to fall into.

Units: the F-F equilibrium distance is 1.412 Angstrom = 2.668 Bohr. PySCF's
gto.M defaults to Angstrom, so the unit must be stated explicitly -- writing
2.668 without a unit puts the molecule at nearly twice its equilibrium bond
length and reports a half-dissociated bond as equilibrium behavior.

Pair selection: F2's HOMO is a pi* lone-pair orbital, not the sigma bond,
which sits three orbitals lower. Naively correlating the HOMO/LUMO pair --
which is what a plain CASSCF(2,2) does here -- describes a lone pair and
misses the bond entirely. gvbpp searches the valence window and selects on
the exchange integral K, so it finds the sigma/sigma* pair (K = 0.26,
roughly three times any competitor).
"""

from pyscf import gto, scf
from gvbpp import gvb_pp

R_EQ_BOHR = 2.668  # = 1.412 Angstrom

mol = gto.M(atom=f'F 0 0 0; F 0 0 {R_EQ_BOHR}',
            unit='Bohr', basis='cc-pvdz', verbose=0)
mf = scf.RHF(mol).run()

# One pair: the F-F sigma bond
g = gvb_pp(mol, n_pairs=1, mf=mf, verbose=4)

print()
print(f'F-F distance:            {R_EQ_BOHR * 0.52917721:.4f} Angstrom')
print(f'F2 sigma pair S_ab:      {g.overlaps[0]:.4f}')
print(f'Selected pair K:         {g.pair_info["pair_exchange"][0]:.4f} Hartree')
print()
print('H2 at its equilibrium has S_ab = 0.98. F2 is much lower even at')
print('equilibrium: the sigma pair is already substantially uncoupled while')
print('the molecule is still bound. That is the GVB signature of a weak,')
print('strongly correlated bond, and it is why single-reference methods')
print('struggle with F2 well before it dissociates.')
