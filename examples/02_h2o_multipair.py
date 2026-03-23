"""
H2O with Multiple GVB Pairs
============================

Demonstrates multi-pair GVB-PP on water, correlating the two O-H bond
pairs and two oxygen lone pairs sequentially.
"""

from pyscf import gto
from gvbpp import gvb_pp

mol = gto.M(
    atom='''
    O  0.000  0.000  0.117
    H  0.000  0.757 -0.469
    H  0.000 -0.757 -0.469
    ''',
    basis='cc-pvdz',
)

# Correlate 4 pairs: 2 O-H bonds + 2 lone pairs
g = gvb_pp(mol, n_pairs=4, verbose=3)
