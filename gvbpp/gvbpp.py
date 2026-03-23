"""
GVB-PP Driver
=============

High-level driver for GVB Perfect Pairing calculations using PySCF.

Single-pair: uses CASSCF(2,2) with GVBPPSolver (exact).
Multi-pair: sequential pair-by-pair CASSCF(2,2) optimization.
"""

import numpy as np
from gvbpp.solver import GVBPPSolver


class GVBPP:
    """
    GVB Perfect Pairing calculation.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object.
    n_pairs : int
        Number of GVB pairs to correlate.

    Attributes
    ----------
    e_tot : float
        Total GVB-PP energy after convergence.
    mo_coeff : ndarray
        MO coefficient matrix (AO x MO).
    ci_coeffs : list of tuples
        (sigma_a, sigma_b) for each pair.
    occupations : ndarray
        Natural orbital occupation numbers for active orbitals.

    Examples
    --------
    >>> from pyscf import gto
    >>> from gvbpp import GVBPP
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='cc-pvdz')
    >>> g = GVBPP(mol, n_pairs=1)
    >>> g.kernel(verbose=0)
    >>> print(f"E = {g.e_tot:.10f}")
    """

    def __init__(self, mol, n_pairs):
        self.mol = mol
        self.n_pairs = n_pairs
        self.mf = None
        self.mc = None
        self.e_tot = None
        self.mo_coeff = None
        self.ci_coeffs = None
        self.occupations = None
        self.pair_results = None

    def kernel(self, mf=None, verbose=4, mo_coeff=None, natorb=True):
        """
        Run the GVB-PP calculation.

        Parameters
        ----------
        mf : pyscf.scf.hf.RHF or None
            Pre-converged SCF object. If None, runs RHF internally.
        verbose : int
            Print level (0=silent, 4=normal).
        mo_coeff : ndarray or None
            Initial MO coefficients for CASSCF.
        natorb : bool
            If True, canonicalize to natural orbitals at the end.

        Returns
        -------
        e_tot : float
            Total GVB-PP energy.
        """
        from pyscf import scf, mcscf

        # Step 1: SCF
        if mf is not None:
            self.mf = mf
        else:
            self.mf = scf.RHF(self.mol)
            self.mf.verbose = max(0, verbose - 2)
            self.mf.kernel()
            if not self.mf.converged:
                print("WARNING: RHF did not converge!")

        # Step 2: GVB-PP
        if self.n_pairs == 1:
            self._kernel_single(verbose, mo_coeff, natorb)
        else:
            self._kernel_multipair(verbose, mo_coeff, natorb)

        return self.e_tot

    def _kernel_single(self, verbose, mo_coeff, natorb):
        """Single pair via CASSCF(2,2). Exact for one GVB pair."""
        from pyscf import mcscf

        self.mc = mcscf.CASSCF(self.mf, 2, 2)
        self.mc.verbose = verbose
        self.mc.fcisolver = GVBPPSolver(1)
        self.mc.conv_tol = 1e-9
        self.mc.conv_tol_grad = 1e-5
        self.mc.max_cycle_macro = 150
        if natorb:
            self.mc.natorb = True
        if mo_coeff is not None:
            self.mc.kernel(mo_coeff)
        else:
            self.mc.kernel()

        self.e_tot = self.mc.e_tot
        self.mo_coeff = self.mc.mo_coeff
        self._extract_pair_info()

    def _kernel_multipair(self, verbose, mo_coeff, natorb):
        """
        Multi-pair GVB-PP via sequential CASSCF(2,2).

        For each pair p (counting down from HOMO), runs CASSCF(2,2) with
        that pair active and all other occupied orbitals frozen. Orbitals
        are updated cumulatively between pairs.
        """
        from pyscf import mcscf

        n_occ = self.mol.nelectron // 2
        mo = mo_coeff if mo_coeff is not None else self.mf.mo_coeff

        self.pair_results = []
        self.occupations = np.zeros(2 * self.n_pairs)
        self.ci_coeffs = []
        ncore = n_occ - 1  # freeze all occupied except one

        for p in range(self.n_pairs):
            # Swap target occupied MO into the HOMO position
            mo_work = mo.copy()
            target_occ = n_occ - 1 - p
            if target_occ != ncore:
                mo_work[:, [target_occ, ncore]] = mo_work[:, [ncore, target_occ]]

            mc = mcscf.CASSCF(self.mf, 2, 2)
            mc.ncore = ncore
            mc.verbose = max(0, verbose - 2)
            mc.fcisolver = GVBPPSolver(1)
            mc.conv_tol = 1e-9
            mc.conv_tol_grad = 1e-5
            mc.max_cycle_macro = 100
            if natorb:
                mc.natorb = True
            mc.kernel(mo_work)

            # Swap optimized orbitals back
            mo_out = mc.mo_coeff.copy()
            if target_occ != ncore:
                mo_out[:, [target_occ, ncore]] = mo_out[:, [ncore, target_occ]]
            mo = mo_out

            # Extract pair info
            dm1 = mc.fcisolver.make_rdm1(mc.ci, 2, (1, 1))
            occs = np.diag(dm1)
            self.occupations[2 * p] = occs[0]
            self.occupations[2 * p + 1] = occs[1]
            sigma_a = np.sqrt(max(0, min(2, occs[0])) / 2.0)
            sigma_b = np.sqrt(max(0, min(2, occs[1])) / 2.0)
            self.ci_coeffs.append((sigma_a, sigma_b))

            self.pair_results.append({
                'pair': p + 1,
                'e_tot': mc.e_tot,
                'e_corr': mc.e_tot - self.mf.e_tot,
                'occ_bond': occs[0],
                'occ_corr': occs[1],
            })

            if verbose >= 2:
                e_corr = mc.e_tot - self.mf.e_tot
                print(f'  Pair {p+1}: E={mc.e_tot:.10f}  '
                      f'occ=({occs[0]:.4f}, {occs[1]:.4f})  '
                      f'corr={e_corr*627.51:.2f} kcal/mol')

        self.mc = mc
        self.e_tot = mc.e_tot
        self.mo_coeff = mo

    def _extract_pair_info(self):
        """Extract pair coefficients from a single-pair calculation."""
        ci = self.mc.ci
        ncas = self.mc.ncas
        nelecas = self.mc.nelecas

        dm1 = self.mc.fcisolver.make_rdm1(ci, ncas, nelecas)
        self.occupations = np.diag(dm1)

        self.ci_coeffs = []
        for p in range(self.n_pairs):
            n_bond = max(0, min(2, self.occupations[2 * p]))
            n_corr = max(0, min(2, self.occupations[2 * p + 1]))
            self.ci_coeffs.append((np.sqrt(n_bond / 2.0),
                                   np.sqrt(n_corr / 2.0)))

    def analyze(self):
        """Print analysis of GVB-PP results."""
        print("=" * 65)
        print("  GVB Perfect Pairing Results")
        print("=" * 65)
        print(f"  Total energy:     {self.e_tot:18.10f} Hartree")
        print(f"  HF energy:        {self.mf.e_tot:18.10f} Hartree")
        corr = self.e_tot - self.mf.e_tot
        print(f"  Correlation:      {corr:18.10f} Hartree")
        print(f"                    {corr * 627.51:18.4f} kcal/mol")
        print(f"  Number of pairs:  {self.n_pairs}")
        print()
        print("  Natural Orbital Occupations:")
        print("  " + "-" * 50)
        print(f"  {'Pair':>4s}  {'Orbital':>12s}  {'Occupation':>12s}  {'sigma':>8s}")
        print("  " + "-" * 50)
        if self.occupations is not None and self.ci_coeffs is not None:
            for p in range(self.n_pairs):
                n_b = self.occupations[2 * p]
                n_c = self.occupations[2 * p + 1]
                s_a, s_b = self.ci_coeffs[p]
                print(f"  {p+1:4d}  {'bonding':>12s}  {n_b:12.6f}  {s_a:8.4f}")
                print(f"  {'':4s}  {'correlating':>12s}  {n_c:12.6f}  {s_b:8.4f}")
        print("  " + "-" * 50)
        print()

        print("  GVB Pair Overlaps:")
        print("  " + "-" * 30)
        if self.ci_coeffs is not None:
            for p in range(self.n_pairs):
                s_a, s_b = self.ci_coeffs[p]
                denom = s_a ** 2 + s_b ** 2
                overlap = (s_a ** 2 - s_b ** 2) / denom if denom > 0 else 0
                print(f"  Pair {p+1:2d}:  S = {overlap:.6f}")
        print("  " + "-" * 30)
        print("=" * 65)

    def get_natural_orbitals(self):
        """
        Return active-space natural orbitals and their occupations.

        Returns
        -------
        mo_coeff : ndarray (nao, ncas)
            Active MO coefficients in AO basis.
        occupations : ndarray (ncas,)
            Natural orbital occupation numbers.
        """
        ncore = self.mc.ncore
        ncas = self.mc.ncas
        return self.mo_coeff[:, ncore:ncore + ncas], self.occupations


def gvb_pp(mol, n_pairs, verbose=4, mf=None, **kwargs):
    """
    Run a GVB-PP calculation.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object.
    n_pairs : int
        Number of electron pairs to correlate.
    verbose : int
        Print level.
    mf : pyscf.scf.hf.RHF or None
        Pre-converged SCF. If None, runs RHF internally.

    Returns
    -------
    calc : GVBPP
        Converged GVBPP object.

    Examples
    --------
    >>> from pyscf import gto
    >>> from gvbpp import gvb_pp
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='cc-pvdz')
    >>> g = gvb_pp(mol, n_pairs=1, verbose=0)
    >>> print(f"E = {g.e_tot:.10f}")
    """
    calc = GVBPP(mol, n_pairs)
    calc.kernel(mf=mf, verbose=verbose, **kwargs)
    calc.analyze()
    return calc
