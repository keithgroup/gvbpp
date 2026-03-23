"""
GVB Perfect Pairing (GVB-PP) for PySCF
=======================================

A standalone GVB-PP solver that runs entirely within PySCF,
implementing the perfect-pairing wavefunction of Bobrowicz and Goddard.

Reference:
    F.W. Bobrowicz and W.A. Goddard III,
    "The Self-Consistent Field Equations for Generalized Valence Bond
     and Open-Shell Hartree-Fock Wave Functions,"
    in Methods of Electronic Structure Theory, ed. H.F. Schaefer III
    (Plenum, New York, 1977), pp. 79-127.
"""

from gvbpp.solver import GVBPPSolver
from gvbpp.gvbpp import GVBPP, gvb_pp

__version__ = "0.1.0"
__all__ = ["GVBPPSolver", "GVBPP", "gvb_pp"]
