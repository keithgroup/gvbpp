"""
GVB-PP CI Solver
================

Custom FCI solver for PySCF's CASSCF that restricts the CI expansion
to the perfect-pairing (PP) subspace.

For n_pairs electron pairs, the PP space contains 2^n_pairs configurations
(each pair is either bonding-occupied or correlating-occupied), compared to
the full CAS space which grows combinatorially.

The solver builds the PP Hamiltonian using Slater-Condon rules, diagonalizes
it, and returns the result in PySCF's standard FCI CI vector format.

Reference:
    F.W. Bobrowicz and W.A. Goddard III (1977), Eq. 6, 43, 58.
"""

import numpy as np


class GVBPPSolver:
    """
    Custom FCI solver for GVB Perfect Pairing.

    Restricts the CI expansion to the PP subspace: only configurations where
    each pair has either (2,0) or (0,2) occupation in its two natural orbitals.

    The active orbitals must be ordered as:
        [bond_1, corr_1, bond_2, corr_2, ..., bond_n, corr_n]
    where bond_p and corr_p are the bonding and correlating NOs of pair p.

    Parameters
    ----------
    n_pairs : int
        Number of GVB electron pairs.

    Examples
    --------
    >>> from pyscf import gto, scf, mcscf
    >>> from gvbpp import GVBPPSolver
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='cc-pvdz')
    >>> mf = scf.RHF(mol).run()
    >>> mc = mcscf.CASSCF(mf, 2, 2)
    >>> mc.fcisolver = GVBPPSolver(n_pairs=1)
    >>> mc.kernel()
    """

    def __init__(self, n_pairs):
        self.n_pairs = n_pairs
        # Attributes required by PySCF's CASSCF driver
        self.nroots = 1
        self.spin = None
        self.verbose = 0
        self.lindep = 1e-14
        self.max_cycle = 100
        self.conv_tol = 1e-10
        self.level_shift = 0.001
        self.pspace_size = 400
        self.davidson_only = False
        self.max_space = 12
        self.max_memory = 2000
        self.wfnsym = 0
        self.threads = 1
        self.orbsym = None

    # ------------------------------------------------------------------
    # Core interface: kernel
    # ------------------------------------------------------------------
    def kernel(self, h1e, eri, ncas, nelecas, ci0=None, ecore=0, **kwargs):
        """
        Solve the CI problem in the PP-restricted subspace.

        Builds the 2^n_pairs x 2^n_pairs PP Hamiltonian matrix using
        Slater-Condon rules and diagonalizes it.

        Parameters
        ----------
        h1e : ndarray (ncas, ncas)
            One-electron integrals in the active space.
        eri : ndarray
            Two-electron integrals in chemist notation (ij|kl).
        ncas : int
            Number of active orbitals.
        nelecas : tuple (neleca, nelecb)
            Number of active alpha and beta electrons.
        ci0 : ndarray or None
            Initial guess (unused; included for API compatibility).
        ecore : float
            Core energy (nuclear repulsion + frozen core). Added to the
            CI eigenvalue before returning, per PySCF convention.

        Returns
        -------
        e_tot : float
            Total energy = CI eigenvalue + ecore.
        ci : ndarray (na, nb)
            CI vector in PySCF's full FCI format.
        """
        n_pairs = self.n_pairs
        n_configs = 2 ** n_pairs

        # PySCF may pass eri in compressed (triangular) format; restore to 4D
        from pyscf import ao2mo
        if eri.ndim != 4:
            eri = ao2mo.restore(1, eri, ncas)

        # Build Hamiltonian matrix in PP basis
        H_pp = np.zeros((n_configs, n_configs))
        for I in range(n_configs):
            for J in range(I, n_configs):
                hij = self._compute_H_element(I, J, h1e, eri, n_pairs)
                H_pp[I, J] = hij
                H_pp[J, I] = hij

        # Diagonalize
        evals, evecs = np.linalg.eigh(H_pp)
        e_ci = evals[0]
        ci_pp = evecs[:, 0]

        # Convert PP vector to full FCI CI-vector format
        ci_fci = self._pp_to_fci(ci_pp, ncas, nelecas)

        return e_ci + ecore, ci_fci

    # ------------------------------------------------------------------
    # Hamiltonian matrix elements (Slater-Condon rules)
    # ------------------------------------------------------------------
    def _compute_H_element(self, I, J, h1e, eri, n_pairs):
        """
        Compute <I|H|J> between two PP configurations.

        Each configuration is an integer whose binary representation gives
        pair occupations: bit p = 0 means bonding orbital doubly occupied,
        bit p = 1 means correlating orbital doubly occupied.

        Slater-Condon rules for closed-shell determinants:
        - Diagonal: E = sum_i 2*h_ii + sum_{i,j} (2*J_ij - K_ij)
        - Double excitation i^2 -> j^2: <K|H|L> = (ij|ij) [B&G Eq. 6]
        - More than one pair differs: zero
        """
        occ_I = self._get_occ_list(I, n_pairs)
        occ_J = self._get_occ_list(J, n_pairs)

        # Find which pairs differ
        diff_pairs = []
        for p in range(n_pairs):
            if ((I >> p) & 1) != ((J >> p) & 1):
                diff_pairs.append(p)

        if len(diff_pairs) > 1:
            return 0.0

        if len(diff_pairs) == 0:
            # Diagonal: closed-shell determinant energy
            E = 0.0
            for i in occ_I:
                E += 2.0 * h1e[i, i]
            for i in occ_I:
                for j in occ_I:
                    E += 2.0 * eri[i, i, j, j] - eri[i, j, i, j]
            return E

        else:
            # Off-diagonal: one pair differs (double excitation i^2 -> j^2)
            p = diff_pairs[0]
            if (I >> p) & 1 == 0:
                i = 2 * p       # bonding (occupied in I)
                j = 2 * p + 1   # correlating (occupied in J)
            else:
                i = 2 * p + 1
                j = 2 * p
            # Exchange integral K_ij = (ij|ij) in chemist notation
            return eri[i, j, i, j]

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def _get_occ_list(self, config, n_pairs):
        """Return list of occupied orbital indices for a PP configuration."""
        occ = []
        for p in range(n_pairs):
            if (config >> p) & 1:
                occ.append(2 * p + 1)  # correlating
            else:
                occ.append(2 * p)      # bonding
        return occ

    def _get_occupation(self, config, n_pairs):
        """Return occupation array (0 or 1) for each active orbital."""
        occ = np.zeros(2 * n_pairs, dtype=int)
        for p in range(n_pairs):
            if (config >> p) & 1:
                occ[2 * p + 1] = 1
            else:
                occ[2 * p] = 1
        return occ

    # ------------------------------------------------------------------
    # PP <-> FCI vector conversion
    # ------------------------------------------------------------------
    def _pp_to_fci(self, ci_pp, ncas, nelecas):
        """Convert PP CI vector to PySCF's full FCI format (na x nb matrix)."""
        from pyscf.fci import cistring
        neleca, nelecb = nelecas
        na = cistring.num_strings(ncas, neleca)
        nb = cistring.num_strings(ncas, nelecb)
        ci = np.zeros((na, nb))

        n_pairs = self.n_pairs
        for idx in range(len(ci_pp)):
            occ_list = self._get_occ_list(idx, n_pairs)
            alpha_str = 0
            beta_str = 0
            for orb in occ_list:
                alpha_str |= (1 << orb)
                beta_str |= (1 << orb)
            alpha_idx = cistring.str2addr(ncas, neleca, alpha_str)
            beta_idx = cistring.str2addr(ncas, nelecb, beta_str)
            ci[alpha_idx, beta_idx] = ci_pp[idx]
        return ci

    def _fci_to_pp(self, ci, ncas, nelecas):
        """Extract PP coefficients from a full FCI CI vector."""
        from pyscf.fci import cistring
        neleca, nelecb = nelecas
        n_pairs = self.n_pairs
        n_configs = 2 ** n_pairs
        ci_pp = np.zeros(n_configs)

        for idx in range(n_configs):
            occ_list = self._get_occ_list(idx, n_pairs)
            alpha_str = 0
            beta_str = 0
            for orb in occ_list:
                alpha_str |= (1 << orb)
                beta_str |= (1 << orb)
            alpha_idx = cistring.str2addr(ncas, neleca, alpha_str)
            beta_idx = cistring.str2addr(ncas, nelecb, beta_str)
            ci_pp[idx] = ci[alpha_idx, beta_idx]
        return ci_pp

    # ------------------------------------------------------------------
    # Density matrices (required by CASSCF orbital optimizer)
    # ------------------------------------------------------------------
    def make_rdm1(self, ci, ncas, nelecas):
        """
        1-electron reduced density matrix.

        For PP, the 1-RDM is diagonal in the natural orbital basis:
            gamma_ii = 2 * sum_K |C_K|^2 * n_{i,K}
        """
        n_pairs = self.n_pairs
        n_configs = 2 ** n_pairs
        dm1 = np.zeros((ncas, ncas))
        ci_pp = self._fci_to_pp(ci, ncas, nelecas)

        for idx in range(n_configs):
            c2 = ci_pp[idx] ** 2
            for orb in self._get_occ_list(idx, n_pairs):
                dm1[orb, orb] += 2.0 * c2
        return dm1

    def make_rdm12(self, ci, ncas, nelecas):
        """1-RDM and 2-RDM. Delegates to PySCF for correctness."""
        from pyscf.fci import direct_spin1
        return direct_spin1.make_rdm12(ci, ncas, nelecas)

    def make_rdm1s(self, ci, ncas, nelecas):
        """Spin-resolved 1-RDMs. For PP (singlet pairs), alpha = beta."""
        dm1 = self.make_rdm1(ci, ncas, nelecas)
        return dm1 * 0.5, dm1 * 0.5

    # ------------------------------------------------------------------
    # PySCF CASSCF compatibility methods
    # ------------------------------------------------------------------
    def absorb_h1e(self, h1e, eri, ncas, nelecas, fac=1):
        from pyscf.fci import direct_spin1
        return direct_spin1.absorb_h1e(h1e, eri, ncas, nelecas, fac)

    def contract_2e(self, eri, ci, ncas, nelecas, **kwargs):
        from pyscf.fci import direct_spin1
        return direct_spin1.contract_2e(eri, ci, ncas, nelecas, **kwargs)

    def get_init_guess(self, ncas, nelecas, nroots, hdiag):
        """Initial guess: all bonding orbitals occupied."""
        from pyscf.fci import cistring
        neleca, nelecb = nelecas
        na = cistring.num_strings(ncas, neleca)
        nb = cistring.num_strings(ncas, nelecb)
        ci0 = np.zeros((na, nb))
        occ_list = self._get_occ_list(0, self.n_pairs)
        alpha_str = beta_str = 0
        for orb in occ_list:
            alpha_str |= (1 << orb)
            beta_str |= (1 << orb)
        ci0[cistring.str2addr(ncas, neleca, alpha_str),
            cistring.str2addr(ncas, nelecb, beta_str)] = 1.0
        return [ci0]

    def gen_linkstr(self, ncas, nelecas, tril=True):
        from pyscf.fci import cistring
        neleca, nelecb = nelecas
        if tril:
            la = cistring.gen_linkstr_index_trilidx(range(ncas), neleca)
            lb = cistring.gen_linkstr_index_trilidx(range(ncas), nelecb)
        else:
            la = cistring.gen_linkstr_index(range(ncas), neleca)
            lb = cistring.gen_linkstr_index(range(ncas), nelecb)
        return la, lb

    @property
    def large_ci(self):
        return False

    def transform_ci_for_orbital_rotation(self, ci, ncas, nelecas, umat):
        return ci
