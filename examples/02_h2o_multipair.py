"""
H2O with Multiple GVB Pairs
===========================

Water has four valence pairs: two O-H bonds and two oxygen lone pairs.
This example correlates them one at a time and shows two things:

  1. The energy decreases monotonically as pairs are added. (It must --
     each pair adds variational freedom.)
  2. GVB-PP recovers roughly 60% of the CASSCF(8,8) correlation energy at
     a small fraction of the cost, because the perfect-pairing restriction
     keeps only 2^n configurations instead of the full CAS expansion.

The bond pairs and lone pairs are distinguishable by their overlaps: a
lone pair is a tightly localized, weakly correlated pair with S_ab close
to 1, while an O-H bond pair has a visibly smaller overlap.
"""

from pyscf import gto, scf, mcscf
from gvbpp import gvb_pp

HARTREE2KCAL = 627.5094740631

mol = gto.M(
    atom='''
    O  0.000  0.000  0.117
    H  0.000  0.757 -0.469
    H  0.000 -0.757 -0.469
    ''',
    basis='cc-pvdz',
    verbose=0,
)
mf = scf.RHF(mol).run()
print(f'RHF reference: {mf.e_tot:.8f} Hartree\n')

print(f"{'n_pairs':>8s}  {'E (Hartree)':>15s}  {'E_corr (kcal/mol)':>18s}"
      f"  {'converged':>10s}")
print('-' * 58)
results = []
for n in [1, 2, 3, 4]:
    g = gvb_pp(mol, n_pairs=n, mf=mf, verbose=0, analyze=False)
    results.append(g)
    print(f'{n:8d}  {g.e_tot:15.8f}  {g.e_corr * HARTREE2KCAL:18.2f}'
          f'  {str(g.converged):>10s}')

# Full analysis of the four-pair wavefunction
print()
results[-1].analyze()

# CASSCF(8,8) is the variational floor for the same active space
mc = mcscf.CASSCF(mf, 8, 8)
mc.verbose = 0
mc.kernel()
frac = results[-1].e_corr / (mc.e_tot - mf.e_tot)
print()
print(f'CASSCF(8,8):   {mc.e_tot:.8f} Hartree '
      f'({(mc.e_tot - mf.e_tot) * HARTREE2KCAL:.2f} kcal/mol)')
print(f'GVB-PP(4) recovers {100 * frac:.0f}% of the CASSCF correlation energy.')
print()
print('The missing ~40% is inter-pair correlation, which the perfect-pairing')
print('restriction excludes by construction. That is the price of the model,')
print('and knowing its size is part of using it honestly.')
