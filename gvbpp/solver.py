"""
GVB-PP CI Solver
================

Custom FCI solver for PySCF's CASSCF that restricts the CI expansion
to the perfect-pairing (PP) subspace.

For ``n_pairs`` electron pairs the PP space contains 2^n_pairs configurations
(each pair is either bonding-occupied or correlating-occupied), compared to
the full CAS space which grows combinatorially.

Open shells
-----------
``n_open`` high-spin singly occupied orbitals may be carried alongside the
pairs. They are spectators to the CI: every configuration keeps them singly
occupied with alpha spin, so the configuration count stays 2^n_pairs. They
do contribute to the diagonal energy through their one-electron terms and
their Coulomb/exchange interaction with the pairs and with each other.

This is what makes the solver usable for radicals and non-singlet atoms --
CH, OH, CH2 triplet, BeH, and most of the atoms in the first row -- which is
the majority of the species in a bonding primer.

Active orbital ordering (required):

    [bond_1, corr_1, ..., bond_n, corr_n, open_1, ..., open_m]

Reference:
    F.W. Bobrowicz and W.A. Goddard III (1977), Eq. 6, 43, 58.
"""

import numpy as np


class GVBPPSolver:
    """
    Custom FCI solver for GVB Perfect Pairing.

    Restricts the CI expansion to the PP subspace: each pair has either
    (2,0) or (0,2) occupation in its two natural orbitals, while any open
    shells stay singly occupied and high-spin.

    Parameters
    ----------
    n_pairs : int
        Number of GVB electron pairs.
    n_open : int
        Number of high-spin singly occupied orbitals (default 0).

    Examples
    --------
    >>> from pyscf import gto, scf, mcscf
    >>> from gvbpp import GVBPPSolver
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.4', unit='Bohr', basis='cc-pvdz')
    >>> mf = scf.RHF(mol).run()
    >>> mc = mcscf.CASSCF(mf, 2, 2)
    >>> mc.fcisolver = GVBPPSolver(n_pairs=1)
    >>> mc.kernel()
    """

    def __init__(self, n_pairs, n_open=0):
        self.n_pairs = int(n_pairs)
        self.n_open = int(n_open)
        # Attributes required by PySCF's CASSCF driver
        self.nroots = 1
        self.spin = self.n_open
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
    # Orbital bookkeeping
    # ------------------------------------------------------------------
    def _open_orbitals(self):
        """Indices of the singly occupied (alpha) active orbitals."""
        return [2 * self.n_pairs + u for u in range(self.n_open)]

    def _get_occ_list(self, config, n_pairs=None):
        """Doubly occupied active orbital indices for a PP configuration."""
        n_pairs = self.n_pairs if n_pairs is None else n_pairs
        occ = []
        for p in range(n_pairs):
            occ.append(2 * p + 1 if (config >> p) & 1 else 2 * p)
        return occ

    # ------------------------------------------------------------------
    # Core interface: kernel
    # ------------------------------------------------------------------
    def kernel(self, h1e, eri, ncas, nelecas, ci0=None, ecore=0, **kwargs):
        """
        Solve the CI problem in the PP-restricted subspace.

        Builds the 2^n_pairs x 2^n_pairs PP Hamiltonian using Slater-Condon
        rules and diagonalizes it.

        Returns
        -------
        e_tot : float
            CI eigenvalue plus ecore, per PySCF convention.
        ci : ndarray (na, nb)
            CI vector in PySCF's full FCI format.
        """
        n_configs = 2 ** self.n_pairs

        from pyscf import ao2mo
        if eri.ndim != 4:
            eri = ao2mo.restore(1, eri, ncas)

        H_pp = np.zeros((n_configs, n_configs))
        for I in range(n_configs):
            for J in range(I, n_configs):
                hij = self._compute_H_element(I, J, h1e, eri)
                H_pp[I, J] = hij
                H_pp[J, I] = hij

        evals, evecs = np.linalg.eigh(H_pp)
        e_ci = evals[0]
        ci_pp = evecs[:, 0]

        ci_fci = self._pp_to_fci(ci_pp, ncas, nelecas)
        return e_ci + ecore, ci_fci

    # ------------------------------------------------------------------
    # Hamiltonian matrix elements (Slater-Condon rules)
    # ------------------------------------------------------------------
    def _compute_H_element(self, I, J, h1e, eri):
        """
        <I|H|J> between two PP configurations.

        Bit p of the configuration integer is 0 if pair p has its bonding
        orbital doubly occupied, 1 if its correlating orbital is.

        Diagonal, with D the doubly occupied set and O the singly occupied
        (alpha) set:

            E = 2 sum_{i in D} h_ii + sum_{u in O} h_uu
              + sum_{i,j in D} [2(ii|jj) - (ij|ij)]
              + sum_{i in D} sum_{u in O} [2(ii|uu) - (iu|iu)]
              + sum_{u<v in O} [(uu|vv) - (uv|uv)]

        Off-diagonal, one pair differing (double excitation i^2 -> j^2):

            <I|H|J> = (ij|ij)

        The open shells are spectators to that excitation, so the coupling
        is unchanged by their presence. More than one pair differing gives
        zero, which is what makes the PP Hamiltonian sparse.
        """
        n_pairs = self.n_pairs
        diff = [p for p in range(n_pairs)
                if ((I >> p) & 1) != ((J >> p) & 1)]

        if len(diff) > 1:
            return 0.0

        if len(diff) == 1:
            p = diff[0]
            if (I >> p) & 1 == 0:
                i, j = 2 * p, 2 * p + 1
            else:
                i, j = 2 * p + 1, 2 * p
            return eri[i, j, i, j]

        # Diagonal
        occ = self._get_occ_list(I)
        op = self._open_orbitals()

        E = 0.0
        for i in occ:
            E += 2.0 * h1e[i, i]
        for u in op:
            E += h1e[u, u]

        for i in occ:
            for j in occ:
                E += 2.0 * eri[i, i, j, j] - eri[i, j, i, j]

        for i in occ:
            for u in op:
                E += 2.0 * eri[i, i, u, u] - eri[i, u, i, u]

        for a in range(len(op)):
            for b in range(a + 1, len(op)):
                u, v = op[a], op[b]
                E += eri[u, u, v, v] - eri[u, v, u, v]

        return E

    # ------------------------------------------------------------------
    # PP <-> FCI vector conversion
    # ------------------------------------------------------------------
    def _strings(self, config):
        """(alpha_string, beta_string) bitmasks for a PP configuration."""
        occ = self._get_occ_list(config)
        a = b = 0
        for orb in occ:
            a |= (1 << orb)
            b |= (1 << orb)
        for u in self._open_orbitals():
            a |= (1 << u)          # open shells are alpha-occupied
        return a, b

    def _pp_to_fci(self, ci_pp, ncas, nelecas):
        """Convert a PP CI vector to PySCF's full FCI format."""
        from pyscf.fci import cistring
        neleca, nelecb = nelecas
        na = cistring.num_strings(ncas, neleca)
        nb = cistring.num_strings(ncas, nelecb)
        ci = np.zeros((na, nb))
        for idx in range(len(ci_pp)):
            astr, bstr = self._strings(idx)
            ci[cistring.str2addr(ncas, neleca, astr),
               cistring.str2addr(ncas, nelecb, bstr)] = ci_pp[idx]
        return ci

    def _fci_to_pp(self, ci, ncas, nelecas):
        """Extract PP coefficients from a full FCI CI vector."""
        from pyscf.fci import cistring
        neleca, nelecb = nelecas
        n_configs = 2 ** self.n_pairs
        ci_pp = np.zeros(n_configs)
        for idx in range(n_configs):
            astr, bstr = self._strings(idx)
            ci_pp[idx] = ci[cistring.str2addr(ncas, neleca, astr),
                            cistring.str2addr(ncas, nelecb, bstr)]
        return ci_pp

    # ------------------------------------------------------------------
    # Density matrices (required by the CASSCF orbital optimizer)
    # ------------------------------------------------------------------
    def make_rdm1(self, ci, ncas, nelecas):
        """
        Spin-summed 1-RDM, diagonal in the natural orbital basis:
        pairs carry 2|C_K|^2, open shells carry exactly 1.
        """
        dm1 = np.zeros((ncas, ncas))
        ci_pp = self._fci_to_pp(ci, ncas, nelecas)
        for idx in range(2 ** self.n_pairs):
            c2 = ci_pp[idx] ** 2
            for orb in self._get_occ_list(idx):
                dm1[orb, orb] += 2.0 * c2
        for u in self._open_orbitals():
            dm1[u, u] = 1.0
        return dm1

    def make_rdm12(self, ci, ncas, nelecas):
        """1- and 2-RDM. Delegates to PySCF for correctness."""
        from pyscf.fci import direct_spin1
        return direct_spin1.make_rdm12(ci, ncas, nelecas)

    def make_rdm1s(self, ci, ncas, nelecas):
        """Spin-resolved 1-RDMs. Pairs split evenly; open shells are alpha."""
        dm1 = np.zeros((ncas, ncas))
        ci_pp = self._fci_to_pp(ci, ncas, nelecas)
        for idx in range(2 ** self.n_pairs):
            c2 = ci_pp[idx] ** 2
            for orb in self._get_occ_list(idx):
                dm1[orb, orb] += 2.0 * c2
        dma = dm1 * 0.5
        dmb = dm1 * 0.5
        for u in self._open_orbitals():
            dma[u, u] = 1.0
        return dma, dmb

    # ------------------------------------------------------------------
    # PySCF CASSCF compatibility
    # ------------------------------------------------------------------
    def absorb_h1e(self, h1e, eri, ncas, nelecas, fac=1):
        from pyscf.fci import direct_spin1
        return direct_spin1.absorb_h1e(h1e, eri, ncas, nelecas, fac)

    def contract_2e(self, eri, ci, ncas, nelecas, **kwargs):
        from pyscf.fci import direct_spin1
        return direct_spin1.contract_2e(eri, ci, ncas, nelecas, **kwargs)

    def get_init_guess(self, ncas, nelecas, nroots, hdiag):
        """Initial guess: all pairs in their bonding orbitals."""
        from pyscf.fci import cistring
        neleca, nelecb = nelecas
        na = cistring.num_strings(ncas, neleca)
        nb = cistring.num_strings(ncas, nelecb)
        ci0 = np.zeros((na, nb))
        astr, bstr = self._strings(0)
        ci0[cistring.str2addr(ncas, neleca, astr),
            cistring.str2addr(ncas, nelecb, bstr)] = 1.0
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
