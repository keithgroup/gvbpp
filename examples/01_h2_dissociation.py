"""
H2 Dissociation Curve: RHF vs GVB-PP vs FCI
=============================================

Demonstrates that GVB-PP correctly dissociates H2 to two hydrogen atoms,
while RHF fails catastrophically at large bond distances.
"""

import numpy as np
from pyscf import gto, scf, fci
from gvbpp import GVBPP

distances = np.arange(0.5, 6.1, 0.25)
e_rhf, e_gvb, e_fci = [], [], []

for R in distances:
    mol = gto.M(atom=f'H 0 0 0; H 0 0 {R}', basis='cc-pvdz', verbose=0)

    mf = scf.RHF(mol).run()
    e_rhf.append(mf.e_tot)

    g = GVBPP(mol, n_pairs=1)
    g.kernel(verbose=0)
    e_gvb.append(g.e_tot)

    ef, _ = fci.FCI(mf).kernel()
    e_fci.append(ef)

# Print table
print(f'{"R(bohr)":>8s}  {"RHF":>12s}  {"GVB-PP(1)":>12s}  {"FCI":>12s}')
print('-' * 50)
for i, R in enumerate(distances):
    print(f'{R:8.2f}  {e_rhf[i]:12.6f}  {e_gvb[i]:12.6f}  {e_fci[i]:12.6f}')

# Optional plot
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(distances, e_rhf, 'b--', label='RHF')
    ax.plot(distances, e_gvb, 'r-', lw=2, label='GVB-PP(1)')
    ax.plot(distances, e_fci, 'k:', label='FCI')
    ax.set_xlabel('R(H-H) / bohr')
    ax.set_ylabel('Energy / hartree')
    ax.set_title('H$_2$ Dissociation Curve')
    ax.legend()
    ax.set_xlim(0.5, 6.0)
    plt.tight_layout()
    plt.savefig('h2_dissociation.png', dpi=150)
    print('\nPlot saved to h2_dissociation.png')
    plt.show()
except ImportError:
    print('\nmatplotlib not available; skipping plot.')
