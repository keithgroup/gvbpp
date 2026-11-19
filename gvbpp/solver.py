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

        The 2^n_pairs x 2^n_pairs PP Hamiltonian is assembled from the
        Coulomb and exchange matrices J_ij = (ii|jj), K_ij = (ij|ij) of the
        active orbitals (vectorized; dense up to 11 pairs, sparse beyond).

        Returns
        -------
        e_tot : float
            CI eigenvalue plus ecore, per PySCF convention.
        ci : ndarray (2^n_pairs,)
            CI vector **in the PP basis**: element I is the coefficient of the
            configuration whose bit p says whether pair p has its correlating
            (1) or bonding (0) orbital doubly occupied. It is never expanded
            to PySCF's determinant format, whose size C(ncas, n)^2 becomes
            prohibitive beyond about eight pairs. Use :meth:`_pp_to_fci` if a
            determinant-format vector is needed for a small case.
        """
        from pyscf import ao2mo
        if eri.ndim != 4:
            eri = ao2mo.restore(1, eri, ncas)
        n = self.n_pairs
        J = np.einsum('iijj->ij', eri)
        K = np.einsum('ijij->ij', eri)
        hd = np.diag(h1e)
        ncfg = 2 ** n
        bits = (np.arange(ncfg)[:, None] >> np.arange(n)[None, :]) & 1
        occ = 2 * np.arange(n)[None, :] + bits            # doubly occupied orbital of each pair
        G = 2.0 * J - K
        E = 2.0 * hd[occ].sum(1)
        for p in range(n):                                # O(n^2 2^n) memory-light accumulation
            E = E + G[occ[:, p][:, None], occ].sum(1)
        op = self._open_orbitals()
        if op:
            op = np.array(op)
            E = E + hd[op].sum() + G[occ][:, :, op].sum((1, 2))
            E = E + 0.5 * (J[np.ix_(op, op)] - K[np.ix_(op, op)]).sum()
        rows, cols, vals = [], [], []
        for p in range(n):
            I0 = np.where(bits[:, p] == 0)[0]
            rows.append(I0); cols.append(I0 + (1 << p))
            vals.append(np.full(len(I0), K[2 * p, 2 * p + 1]))
        if n <= 11:
            H_pp = np.diag(E)
            for r_, c_, v_ in zip(rows, cols, vals):
                H_pp[r_, c_] = v_
                H_pp[c_, r_] = v_
            evals, evecs = np.linalg.eigh(H_pp)
            e_ci, ci_pp = evals[0], evecs[:, 0]
        else:
            from scipy.sparse import coo_matrix, diags
            from scipy.sparse.linalg import eigsh
            r_, c_, v_ = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
            off = coo_matrix((v_, (r_, c_)), shape=(ncfg, ncfg)).tocsr()
            H_pp = diags(E) + off + off.T
            v0 = np.zeros(ncfg); v0[0] = 1.0
            if ci0 is not None and np.size(ci0) == ncfg:
                v0 = np.asarray(ci0, float).ravel()
            evals, evecs = eigsh(H_pp, k=1, which='SA', v0=v0, tol=1e-12)
            e_ci, ci_pp = evals[0], evecs[:, 0]
        if ci_pp[np.abs(ci_pp).argmax()] < 0:
            ci_pp = -ci_pp
        return e_ci + ecore, ci_pp

    # PySCF's CASSCF calls this in its micro-iterations when present; it keeps
    # the driver from touching determinant-space routines.
    approx_kernel = kernel

    def _as_pp(self, ci, ncas, nelecas):
        """Accept either a PP vector or (legacy) a determinant-format matrix."""
        ci = np.asarray(ci)
        if ci.ndim == 1 and ci.size == 2 ** self.n_pairs:
            return ci
        return self._fci_to_pp(ci, ncas, nelecas)

    def _occupations(self):
        """alpha and beta occupation patterns, shape (2^n_pairs, ncas)."""
        n, m = self.n_pairs, self.n_open
        ncfg = 2 ** n
        bits = (np.arange(ncfg)[:, None] >> np.arange(n)[None, :]) & 1
        occ = 2 * np.arange(n)[None, :] + bits
        nb = np.zeros((ncfg, 2 * n + m))
        np.put_along_axis(nb, occ, 1.0, axis=1)
        na = nb.copy()
        if m:
            na[:, 2 * n:] = 1.0
        return na, nb, bits

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
        c2 = self._as_pp(ci, ncas, nelecas) ** 2
        na, nb, _ = self._occupations()
        return np.diag(c2 @ (na + nb))

    def make_rdm12(self, ci, ncas, nelecas):
        """
        1- and 2-RDM (PySCF convention, E2 = 1/2 sum dm2[pqrs] (pq|rs)),
        built directly from the PP vector. For a PP wave function the only
        non-zero 2-RDM elements are

            dm2[i,i,j,j] =  sum_I c_I^2 n_i n_j
            dm2[i,j,j,i] = -sum_I c_I^2 (na_i na_j + nb_i nb_j)
            dm2[b,c,b,c] = dm2[c,b,c,b] = 2 sum_{I: pair p bonding} c_I c_{I+p}

        with (b, c) the two orbitals of pair p.
        """
        c = self._as_pp(ci, ncas, nelecas)
        c2 = c * c
        na, nb, bits = self._occupations()
        nt = na + nb
        dm1 = np.diag(c2 @ nt)
        A = (nt * c2[:, None]).T @ nt
        X = (na * c2[:, None]).T @ na + (nb * c2[:, None]).T @ nb
        dm2 = np.zeros((ncas,) * 4)
        ii = np.arange(ncas)
        dm2[ii[:, None], ii[:, None], ii[None, :], ii[None, :]] += A
        dm2[ii[:, None], ii[None, :], ii[None, :], ii[:, None]] -= X
        for p in range(self.n_pairs):
            I0 = np.where(bits[:, p] == 0)[0]
            x = 2.0 * np.sum(c[I0] * c[I0 + (1 << p)])
            b_, c_ = 2 * p, 2 * p + 1
            dm2[b_, c_, b_, c_] += x
            dm2[c_, b_, c_, b_] += x
        return dm1, dm2

    def make_rdm1s(self, ci, ncas, nelecas):
        """Spin-resolved 1-RDMs. Pairs split evenly; open shells are alpha."""
        c2 = self._as_pp(ci, ncas, nelecas) ** 2
        na, nb, _ = self._occupations()
        return np.diag(c2 @ na), np.diag(c2 @ nb)

    # ------------------------------------------------------------------
    # PySCF CASSCF compatibility
    # ------------------------------------------------------------------
    def get_init_guess(self, ncas, nelecas, nroots, hdiag):
        """Initial guess: all pairs in their bonding orbitals (PP basis)."""
        ci0 = np.zeros(2 ** self.n_pairs)
        ci0[0] = 1.0
        return [ci0]

    @property
    def large_ci(self):
        return False

    def transform_ci_for_orbital_rotation(self, ci, ncas, nelecas, umat):
        return ci
